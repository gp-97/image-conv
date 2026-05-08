//! Image Convolution Engine
//!
//! ## What is convolution?
//!
//! ```text
//! Convolution slides a filter kernel over an image, computing a weighted sum
//! at each position. Each output pixel is the dot product of the kernel with
//! the underlying image patch:
//!
//!   Input image patch         Kernel           Output pixel
//!   ┌────┬────┬────┐      ┌────┬────┬────┐
//!   │ a  │ b  │ c  │      │ k₀ │ k₁ │ k₂ │     out = a·k₀ + b·k₁ + c·k₂
//!   ├────┼────┼────┤   ⊙  ├────┼────┼────┤        + d·k₃ + e·k₄ + f·k₅
//!   │ d  │ e  │ f  │      │ k₃ │ k₄ │ k₅ │        + g·k₆ + h·k₇ + i·k₈
//!   ├────┼────┼────┤      ├────┼────┼────┤
//!   │ g  │ h  │ i  │      │ k₆ │ k₇ │ k₈ │     Then clamped to [0, 255]
//!   └────┴────┴────┘      └────┴────┴────┘
//!
//! The kernel "slides" horizontally then vertically across the whole image,
//! producing one output pixel per valid position.
//! ```
//!
//! ## Two Convolution Paths
//!
//! ```text
//!                    ┌──────────────────┐
//!                    │  try_separable()  │
//!                    └────────┬─────────┘
//!                             │
//!               ┌─────────────┴─────────────┐
//!               │                           │
//!          Separable                     Not separable
//!               │                           │
//!               ▼                           ▼
//!   ┌──────────────────────┐    ┌──────────────────────┐
//!   │  separable_convolve  │    │      convolve        │
//!   │  (two 1D passes)     │    │    (one 2D pass)      │
//!   │                      │    │                      │
//!   │  Input ────┬──────── │    │  Input image          │
//!   │            │ˡhorizontal│    │  7×7 kernel = 49 ops │
//!   │            ▼   pass   │    │  per output pixel     │
//!   │        Temp buffer    │    │                      │
//!   │   (width reduced by   │    └──────────────────────┘
//!   │    filter-1+2·pad)    │
//!   │            │          │    Legend:
//!   │            │vertical  │    7×7 separable = 14 ops
//!   │            ▼  pass    │    per pixel — 3.5× faster
//!   │        Output image   │
//!   └──────────────────────┘
//! ```
//!
//! ## Output size formula
//!
//! For input size `W×H`, filter size `Fw×Fh`, padding `P`, stride `S`:
//!
//! ```text
//! output_width  = (W - Fw + 2·P) / S + 1
//! output_height = (H - Fh + 2·P) / S + 1
//! ```

use crate::{Filter, PaddingType};
use photon_rs::transform::padding_uniform as uniform;
use photon_rs::PhotonImage;
use photon_rs::Rgba;

/// Standard 2D convolution: slides the full filter kernel over the image.
///
/// ```text
/// For each output pixel (yc, xc) with stride S:
///
///   Padded image (width=wp)      Filter (fw×fh)     Output pixel
///   ┌───────────────────────┐     ┌───┬───┬───┐
///   │                       │     │   │   │   │
///   │  (row_base, col_base)─┼──┐  │   │   │   │    r = Σ fy Σ fx
///   │  │                    │  │  │   │   │   │      raw[px]·kernel[fy][fx]
///   │  │  fh rows           │  │  ├───┼───┼───┤
///   │  │                    │  │  │   │   │   │
///   │  └── fw cols ────────┘  │  │   │   │   │
///   │                       │  │  └───┴───┴───┘
///   └───────────────────────┘  │
///     px = (row_base+fy)*wp + col_base+fx  (×4 for RGBA)
/// ```
///
/// Output buffer is pre-allocated with `with_capacity` to avoid reallocations.
/// All channels are accumulated as `f32` and clamped to `[0, 255]` at the end.
fn convolve(img_padded: &PhotonImage, filter: &Filter, width_conv: u32, height_conv: u32, stride: u32) -> PhotonImage {
    let raw = img_padded.get_raw_pixels();
    let wp = img_padded.get_width() as usize;
    let fw = filter.width;
    let fh = filter.height;
    let kernel = &filter.kernel;
    let wc = width_conv as usize;
    let hc = height_conv as usize;
    let stride = stride as usize;

    let out_size = wc * hc * 4;
    let mut out = Vec::with_capacity(out_size);

    for yc in 0..hc {
        let row_base = yc * stride;

        for xc in 0..wc {
            let col_base = xc * stride;

            let mut r: f32 = 0.0;
            let mut g: f32 = 0.0;
            let mut b: f32 = 0.0;

            for fy in 0..fh {
                let row_offset = (row_base + fy) * wp;
                let k_row = fy * fw;

                for fx in 0..fw {
                    let px = (row_offset + col_base + fx) * 4;
                    let k = kernel[k_row + fx];

                    r += raw[px] as f32 * k;
                    g += raw[px + 1] as f32 * k;
                    b += raw[px + 2] as f32 * k;
                }
            }

            out.push(r.clamp(0.0, 255.0) as u8);
            out.push(g.clamp(0.0, 255.0) as u8);
            out.push(b.clamp(0.0, 255.0) as u8);
            out.push(255_u8);
        }
    }

    debug_assert_eq!(out.len(), out_size, "output buffer size mismatch");

    #[cfg(debug_assertions)]
    println!("Convolution done...");

    PhotonImage::new(out, width_conv, height_conv)
}

/// Separable convolution: decomposes the 2D kernel into two 1D passes.
///
/// ## How it works
///
/// A separable kernel can be factored as `kernel[i][j] = col[i] × row[j]`.
/// Instead of one 2D pass (fw·fh ops/pixel), we do two 1D passes:
///
/// ```text
/// PASS 1 — Horizontal (row vector applied to every row independently)
/// ──────────────────────────────────────────────────────────────────
///   Input (padded)            temp[row][x] = Σ row[fx] × input[row][x·S + fx]
///   ┌──────────────────┐             fx
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│      ┌──────────────────────┐
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│      │ Row 0 convolved       │
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│  →   │ Row 1 convolved       │
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│      │ ...                   │
///   │ ■ ■ ■ ■ ■ ■ ■ ■ ■│      └──────────────────────┘
///   └──────────────────┘            temp_w = (wp - fw)/S + 1
///
/// PASS 2 — Vertical (column vector applied to temp buffer columns)
/// ────────────────────────────────────────────────────────────────
///                    output[y][x] = Σ col[fy] × temp[y·S + fy][x]
///                                    fy
///   temp buffer (hp rows × temp_w cols)    output (hc × wc)
///   ┌──────────────────────┐              ┌─────────────┐
///   │ r₀ g₀ b₀ r₁ g₁ b₁ ...│              │  convolved  │
///   │ r₀ g₀ b₀ r₁ g₁ b₁ ...│          →   │  pixels     │
///   │        ...            │              └─────────────┘
///   └──────────────────────┘
/// ```
///
/// The temp buffer stores **unclamped f32** RGB values (3 floats per pixel)
/// to avoid precision loss between passes. Clamping only happens at the end
/// of the second pass.
fn separable_convolve(
    img_padded: &PhotonImage,
    row_vec: &[f32],
    col_vec: &[f32],
    width_conv: u32,
    height_conv: u32,
    stride: u32,
) -> PhotonImage {
    let raw = img_padded.get_raw_pixels();
    let wp = img_padded.get_width() as usize;
    let hp = img_padded.get_height() as usize;
    let fw = row_vec.len();
    let fh = col_vec.len();
    let wc = width_conv as usize;
    let hc = height_conv as usize;
    let stride = stride as usize;

    let temp_w = wc;
    let temp_size = hp * temp_w * 3;
    let mut temp: Vec<f32> = vec![0.0; temp_size];

    for y in 0..hp {
        let row_input = y * wp;
        let row_temp = y * temp_w;
        for x in 0..temp_w {
            let col_input = x * stride;
            let mut r: f32 = 0.0;
            let mut g: f32 = 0.0;
            let mut b: f32 = 0.0;
            for fx in 0..fw {
                let px = (row_input + col_input + fx) * 4;
                let k = row_vec[fx];
                r += raw[px] as f32 * k;
                g += raw[px + 1] as f32 * k;
                b += raw[px + 2] as f32 * k;
            }
            let t = (row_temp + x) * 3;
            temp[t] = r;
            temp[t + 1] = g;
            temp[t + 2] = b;
        }
    }

    let out_size = wc * hc * 4;
    let mut out = Vec::with_capacity(out_size);

    for yc in 0..hc {
        let row_base = yc * stride;
        for xc in 0..wc {
            let mut r: f32 = 0.0;
            let mut g: f32 = 0.0;
            let mut b: f32 = 0.0;
            for fy in 0..fh {
                let t = ((row_base + fy) * temp_w + xc) * 3;
                let k = col_vec[fy];
                r += temp[t] * k;
                g += temp[t + 1] * k;
                b += temp[t + 2] * k;
            }
            out.push(r.clamp(0.0, 255.0) as u8);
            out.push(g.clamp(0.0, 255.0) as u8);
            out.push(b.clamp(0.0, 255.0) as u8);
            out.push(255_u8);
        }
    }

    debug_assert_eq!(out.len(), out_size, "output buffer size mismatch");

    #[cfg(debug_assertions)]
    println!("Separable convolution done...");

    PhotonImage::new(out, width_conv, height_conv)
}

/// Computes the output dimension for one axis.
///
/// ```text
/// output = (input_size - filter_size + 2·padding) / stride + 1
///
/// Example: input=500, filter=7, pad=0, stride=1
///          output = (500 - 7 + 0) / 1 + 1 = 494
/// ```
#[inline]
fn output_dim(input_size: u32, filter_size: u32, pad: u32, stride: u32) -> u32 {
    let dim = input_size - filter_size + 2 * pad;
    if dim % stride != 0 {
        eprintln!("[WARNING]: stride value not suitable. Convolution may fail.");
    }
    dim / stride + 1
}

/// Applies convolution to an image using the given filter.
///
/// # Arguments
/// * `img` — The input image (`PhotonImage` from photon-rs).
/// * `filter` — The convolution kernel (e.g. Sobel, Gaussian, Laplacian).
/// * `stride` — Step size between output pixels (1 = dense, >1 = downsample).
/// * `padding` — Border handling: `UNIFORM(n)` pads with black, `NONE` skips padding.
///
/// # Dispatch Logic
///
/// ```text
/// convolution(img, filter, stride, padding)
/// │
/// ├─ stride=0? ───> ERROR: exit
/// │
/// ├─ try_separable()
/// │   ├─ Some(col, row) ──> separable_convolve()  ← 2× 1D pass fast path
/// │   └─ None ───────────> convolve()             ← standard 2D pass
/// │
/// └─ padding
///     ├─ UNIFORM(n) ──> pad image, then convolve
///     └─ NONE ────────> convolve directly (zero-copy, faster)
/// ```
///
/// # Speedup Examples
///
/// | Kernel | Size | 2D ops/px | Separable ops/px | Speedup |
/// |--------|------|-----------|------------------|---------|
/// | Sobel  | 3×3  | 9         | 6                | 1.5×    |
/// | Gauss  | 7×7  | 49        | 14               | 3.5×    |
/// | Gauss  | 15×15| 225       | 30               | 7.5×    |
///
/// The separable fast path is **automatically selected** with zero user effort.
///
/// # Example
///
/// ```no_run
/// use image_conv::conv;
/// use image_conv::{Filter, PaddingType};
///
/// let img = photon_rs::native::open_image("img.jpg").expect("No such file found");
/// let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
/// let filter = Filter::from(sobel_x, 3, 3);
/// let img_conv = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));
///```
pub fn convolution(img: &PhotonImage, filter: Filter, stride: u32, padding: PaddingType) -> PhotonImage {
    if stride == 0 {
        eprintln!("[ERROR]: Stride provided = 0");
        std::process::exit(1);
    }

    let separable = filter.try_separable();

    match &padding {
        PaddingType::UNIFORM(pad_amt) => {
            let img_padded = uniform(img, *pad_amt, Rgba::new(0, 0, 0, 255));
            let wc = output_dim(img.get_width(), filter.width as u32, *pad_amt, stride);
            let hc = output_dim(img.get_height(), filter.height as u32, *pad_amt, stride);

            if let Some((col, row)) = separable {
                separable_convolve(&img_padded, &row, &col, wc, hc, stride)
            } else {
                convolve(&img_padded, &filter, wc, hc, stride)
            }
        }
        PaddingType::NONE => {
            let wc = output_dim(img.get_width(), filter.width as u32, 0, stride);
            let hc = output_dim(img.get_height(), filter.height as u32, 0, stride);

            if let Some((col, row)) = separable {
                separable_convolve(img, &row, &col, wc, hc, stride)
            } else {
                convolve(img, &filter, wc, hc, stride)
            }
        }
    }
}
