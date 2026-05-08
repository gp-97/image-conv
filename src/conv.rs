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
//! ## Three-tier performance
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
//!   │            │horizontal│    │  7×7 kernel = 49 ops │
//!   │            ▼   pass   │    │  per output pixel     │
//!   │        Temp buffer    │    │                      │
//!   │   (width reduced by   │    └──────────────────────┘
//!   │    filter-1+2·pad)    │
//!   │            │          │    Legend:
//!   │            │vertical  │    7×7 separable = 14 ops
//!   │            ▼  pass    │    per pixel — 3.5× faster
//!   │        Output image   │
//!   └──────────────────────┘
//!
//!   Both paths are parallelised over output rows via rayon.
//! ```
//!
//! ## Threading model
//!
//! Each output row is completely independent — zero data dependencies.
//! Rayon splits the output buffer into per-row slices and processes
//! them in parallel across all available cores.
//!
//! ```text
//!   Output buffer (hc rows × wc cols × 4 bytes RGBA)
//!   ┌──────────────────────────────────────┐
//!   │ Row 0 ──────▶ Thread 0               │
//!   │ Row 1 ──────▶ Thread 1               │  Output rows are
//!   │ Row 2 ──────▶ Thread 2               │  processed in
//!   │ Row 3 ──────▶ Thread 3               │  parallel — each
//!   │   ...                                │  thread reads from
//!   │ Row hc-1 ───▶ Thread N               │  the input image
//!   └──────────────────────────────────────┘  (immutable, safe)
//! ```
//!
//! ## Output size formula
//!
//! ```text
//! output_width  = (W - Fw + 2·P) / S + 1
//! output_height = (H - Fh + 2·P) / S + 1
//! ```

use crate::{Filter, PaddingType};
use photon_rs::transform::padding_uniform as uniform;
use photon_rs::PhotonImage;
use photon_rs::Rgba;
use rayon::prelude::*;

/// Standard 2D convolution — parallelised over output rows.
///
/// ```text
/// Each output row is independent, so we split the output buffer
/// into per-row slices and process them in parallel via rayon.
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
/// All channels accumulate as `f32` and clamp to `[0, 255]` at the end.
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
    let mut out = vec![0u8; out_size];

    out.par_chunks_mut(wc * 4).enumerate().for_each(|(yc, row_out)| {
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

            let i = xc * 4;
            row_out[i] = r.clamp(0.0, 255.0) as u8;
            row_out[i + 1] = g.clamp(0.0, 255.0) as u8;
            row_out[i + 2] = b.clamp(0.0, 255.0) as u8;
            row_out[i + 3] = 255_u8;
        }
    });

    debug_assert_eq!(out.len(), out_size);

    #[cfg(debug_assertions)]
    println!("Convolution done (rayon)...");

    PhotonImage::new(out, width_conv, height_conv)
}

/// Separable convolution — each pass is parallelised independently.
///
/// ## How it works
///
/// A separable kernel factors as `kernel[i][j] = col[i] × row[j]`.
/// Two 1D passes replace one 2D pass: O(fw+fw) per pixel vs O(fw·fh).
///
/// ```text
/// PASS 1 — Horizontal (parallel over rows)
/// ─────────────────────────────────────────
///   Input (padded, hp rows)   temp (hp rows × temp_w cols × 3 floats)
///   ┌──────────────────┐      ┌──────────────────────┐
///   │ Row 0 ──▶ T0     │      │ Row 0 convolved       │
///   │ Row 1 ──▶ T1     │  →   │ Row 1 convolved       │  Each row is
///   │ Row 2 ──▶ T2     │      │ ...                   │  independent
///   │ ...               │      └──────────────────────┘
///   └──────────────────┘
///
/// PASS 2 — Vertical (parallel over output rows)
/// ──────────────────────────────────────────────
///   temp buffer (read-only)      output (hc × wc)
///   ┌──────────────────────┐    ┌─────────────┐
///   │ r₀ g₀ b₀ r₁ g₁ b₁ ...│    │ Row 0 ──▶ T0 │
///   │ r₀ g₀ b₀ r₁ g₁ b₁ ...│ →  │ Row 1 ──▶ T1 │
///   │        ...            │    │ ...          │
///   └──────────────────────┘    └─────────────┘
/// ```
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

    // Horizontal pass — each row is independent, process in parallel
    temp.par_chunks_mut(temp_w * 3).enumerate().for_each(|(y, row_temp)| {
        let row_input = y * wp;
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
            let t = x * 3;
            row_temp[t] = r;
            row_temp[t + 1] = g;
            row_temp[t + 2] = b;
        }
    });

    // Vertical pass — output rows are independent, process in parallel
    let out_size = wc * hc * 4;
    let mut out = vec![0u8; out_size];

    out.par_chunks_mut(wc * 4).enumerate().for_each(|(yc, row_out)| {
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
            let i = xc * 4;
            row_out[i] = r.clamp(0.0, 255.0) as u8;
            row_out[i + 1] = g.clamp(0.0, 255.0) as u8;
            row_out[i + 2] = b.clamp(0.0, 255.0) as u8;
            row_out[i + 3] = 255_u8;
        }
    });

    debug_assert_eq!(out.len(), out_size);

    #[cfg(debug_assertions)]
    println!("Separable convolution done (rayon)...");

    PhotonImage::new(out, width_conv, height_conv)
}

/// Computes the output dimension for one axis.
///
/// ```text
/// output = (input_size - filter_size + 2·padding) / stride + 1
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
/// Each path (separable 1D and standard 2D) is parallelised across output
/// rows via rayon — no data dependencies between rows, trivially parallel.
///
/// # Speedup Examples
///
/// | Kernel  | Size  | Sequential | Rayon (8 cores) |
/// |---------|-------|------------|-----------------|
/// | Gauss   | 7×7   | 61 ms      | ~8 ms           |
/// | Gauss   | 15×15 | 94 ms      | ~12 ms          |
/// | Laplacian|3×3   | 47 ms      | ~6 ms           |
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
