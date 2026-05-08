//! # image-conv — High-Performance Image Convolution
//!
//! This library applies convolution filters to images. A convolution slides a small
//! matrix (the **kernel**, or filter) over every pixel of the image, multiplying
//! overlapping values and summing the results to produce each output pixel.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐     ┌─────────────────────┐     ┌──────────────┐
//! │  Input Image  │────▶│  convolution()       │────▶│ Output Image │
//! │ (PhotonImage) │     │  ├─ try_separable()  │     │ (PhotonImage)│
//! └──────────────┘     │  │  ├─ Y: separable   │     └──────────────┘
//!                      │  │  └─ N: 2D fallback │
//! ┌──────────────┐     │  ├─ padding           │
//! │  Filter       │────▶│  └─ output dimensions│
//! └──────────────┘     └─────────────────────┘
//! ```
//!
//! The `Filter` struct holds a convolution kernel (also called a "mask" or "window").
//! `convolution()` auto-detects whether the kernel is **separable** — if so, it
//! decomposes the 2D convolution into two 1D passes for a major speedup (e.g.
//! a 15×15 Gaussian goes from O(225) to O(30) per pixel, ~7.5× faster).
//!
//! ## Example
//! ```no_run
//! use image_conv::conv;
//! use image_conv::{Filter, PaddingType};
//! use photon_rs::native::{open_image, save_image};
//!
//! fn main() {
//!     let mut img = open_image("img.jpg").expect("No such file found");
//!
//!     // Sobel-X edge detection kernel (3×3)
//!     let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
//!     let filter = Filter::from(sobel_x, 3, 3);
//!
//!     // Automatically detected as separable: col=[1,2,1] × row=[1,0,-1]
//!     let img_conv = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));
//!     save_image(img_conv, "img_conv.jpg");
//! }
//! ```

pub mod conv;

use image::DynamicImage::{self, ImageRgba8};
use photon_rs::{helpers, PhotonImage};
use prettytable::{Cell, Row, Table};

/// A 2D convolution kernel (filter / mask / window).
///
/// ## Kernel Layout
///
/// The kernel is stored in **row-major** order as a flat `Vec<f32>`.
/// For a 3×3 Sobel-X kernel:
///
/// ```text
///      j=0  j=1  j=2         Flat storage:
/// i=0 [  1    0   -1  ]      index: 0  1  2  3  4  5  6  7  8
/// i=1 [  2    0   -2  ]      value: 1, 0,-1, 2, 0,-2, 1, 0,-1
/// i=2 [  1    0   -1  ]
/// ```
///
/// Access formula: `kernel[i * width + j]`
#[derive(Clone)]
pub struct Filter {
    width: usize,
    height: usize,
    kernel: Vec<f32>,
}

impl Filter {
    /// Creates a zero-initialised filter of given dimensions.
    ///
    /// ```text
    /// Filter::new(3, 3) → [0, 0, 0, 0, 0, 0, 0, 0, 0]
    /// ```
    pub fn new(width: usize, height: usize) -> Self {
        let mut kernel = Vec::<f32>::new();
        Vec::resize(&mut kernel, width * height, 0_f32);
        Self { width, height, kernel }
    }

    /// Creates a filter from a pre-computed flat kernel buffer.
    ///
    /// The buffer must have exactly `width × height` elements, stored
    /// row-major (row 0 first, then row 1, etc.).
    ///
    /// # Panics
    /// Exits the process if `kernel_buffer.len() != width * height`.
    pub fn from(kernel_buffer: Vec<f32>, width: usize, height: usize) -> Self {
        let kernel_size = kernel_buffer.len();
        if width * height != kernel_size {
            eprintln!("[ERROR]: Invalid dimensions provided");
            std::process::exit(1);
        }
        Self {
            width,
            height,
            kernel: kernel_buffer,
        }
    }

    /// Returns the filter width (number of columns).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the filter height (number of rows).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns a clone of the kernel data as a flat `Vec<f32>`.
    pub fn kernel(&self) -> Vec<f32> {
        self.kernel.clone()
    }

    /// Returns the element at `(row, col)`, or `None` if out of bounds.
    pub fn get_element(&self, x: usize, y: usize) -> Option<f32> {
        let element_pos = x * self.width + y;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            None
        } else {
            Some(self.kernel[element_pos])
        }
    }

    /// Sets the element at `(row, col)` to the given value.
    ///
    /// # Panics
    /// Exits the process if the position is out of bounds.
    pub fn set_value_at_pos(&mut self, val: f32, position: (usize, usize)) {
        let element_pos = position.0 * self.width + position.1;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            eprintln!("[ERROR]: Index out of bound");
            std::process::exit(1);
        } else {
            self.kernel[element_pos] = val;
        }
    }

    /// Tests whether this kernel is **separable** — i.e. can be expressed
    /// as the outer product of a column vector and a row vector.
    ///
    /// ## The Concept
    ///
    /// ```text
    /// A 2D kernel K can sometimes be factored:  K = col × rowᵀ
    ///
    ///   [ a ]                 [ a·c  a·d  a·e ]
    ///   [ b ] × [ c  d  e ] = [ b·c  b·d  b·e ]
    ///
    ///   col 2×1  row 1×3      kernel 2×3
    /// ```
    ///
    /// When separable, convolution can use two 1D passes instead of one 2D pass:
    /// O(fw·fh) → O(fw + fh) per output pixel.
    ///
    /// ## Decomposition Algorithm
    ///
    /// 1. Find the largest absolute value in the kernel — this is the **pivot**.
    /// 2. Extract the pivot's **row** as the row vector.
    /// 3. Extract the pivot's **column**, divided by the pivot value, as the
    ///    column vector.
    /// 4. Verify that for every element: `col[i] × row[j] ≈ kernel[i][j]`.
    ///    If any element deviates beyond `1e-4`, the kernel is not separable.
    ///
    /// ```text
    /// Example: Sobel-X = [1,0,-1; 2,0,-2; 1,0,-1]
    ///
    ///   Pivot = |2| at (1,0)          col = col₀ / 2 = [1/2, 2/2, 1/2]
    ///   row   = row₁ = [2, 0, -2]     row = row₁      = [2, 0, -2]
    ///
    ///   Verify: [0.5] × [2, 0, -2] = [1,0,-1; 2,0,-2; 1,0,-1]  ✓
    ///           [1.0]
    ///           [0.5]
    /// ```
    ///
    /// ## Returns
    /// * `Some((col_vec, row_vec))` — col has `height` elements, row has `width` elements.
    /// * `None` — kernel is not separable (or is all zeros).
    pub fn try_separable(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        let fw = self.width;
        let fh = self.height;
        let kernel = &self.kernel;

        let (pi, pj) = (0..fh)
            .flat_map(|i| (0..fw).map(move |j| (i, j)))
            .max_by(|(i1, j1), (i2, j2)| {
                let v1 = kernel[i1 * fw + j1].abs();
                let v2 = kernel[i2 * fw + j2].abs();
                v1.partial_cmp(&v2).unwrap_or(std::cmp::Ordering::Equal)
            })?;

        let pivot = kernel[pi * fw + pj];

        if pivot.abs() < 1e-10 {
            return None;
        }

        let row_vec: Vec<f32> = (0..fw).map(|j| kernel[pi * fw + j]).collect();
        let col_vec: Vec<f32> = (0..fh).map(|i| kernel[i * fw + pj] / pivot).collect();

        for i in 0..fh {
            for j in 0..fw {
                if (col_vec[i] * row_vec[j] - kernel[i * fw + j]).abs() > 1e-4 {
                    return None;
                }
            }
        }

        Some((col_vec, row_vec))
    }

    /// Pretty-prints the kernel as a formatted table to stdout.
    pub fn display(&self) {
        let mut table = Table::new();

        for x in 0..self.height {
            let mut row_vec = Vec::<Cell>::new();
            for y in 0..self.width {
                let pos = x * self.width + y;
                let element = &self.kernel[pos];
                row_vec.push(Cell::new(element.to_string().as_str()));
            }
            table.add_row(Row::new(row_vec));
        }
        table.printstd();
    }
}

/// Specifies how border pixels are handled during convolution.
///
/// ```text
/// Without padding, the output shrinks:  out = in - (filter - 1)
///
///   Input (5×5)    Filter 3×3   Output (3×3)
///   ┌─────────┐    ┌───┐       ┌─────┐
///   │ * * * * *│    │###│       │ * * *│
///   │ * * * * *│    │###│       │ * * *│
///   │ * * * * *│ ×  │###│  =    │ * * *│
///   │ * * * * *│    └───┘       └─────┘
///   │ * * * * *│
///   └─────────┘
///
/// With UNIFORM(1), a border of 1 is added all around:
///
///   Input (5×5)       Padded (7×7)     Output (5×5)
///   ┌─────────┐      ┌·············┐  ┌─────────┐
///   │ * * * * *│      ┆ 0 0 0 0 0 0 0┆  │ * * * * *│
///   │ * * * * *│      ┆ 0 * * * * * 0┆  │ * * * * *│
///   │ * * * * *│  →   ┆ 0 * * * * * 0┆ →│ * * * * *│  (same size as input)
///   │ * * * * *│      ┆ 0 * * * * * 0┆  │ * * * * *│
///   │ * * * * *│      ┆ 0 * * * * * 0┆  │ * * * * *│
///   └─────────┘      ┆ 0 0 0 0 0 0 0┆  └─────────┘
///                     └·············┘
/// ```
pub enum PaddingType {
    /// Pad with `n` black pixels on all four sides.
    /// Output size = `(input_size - filter_size + 2·n) / stride + 1`.
    UNIFORM(u32),
    /// No padding. Output shrinks by `filter_size - 1` in each dimension.
    /// Faster than UNIFORM since no padding buffer is allocated.
    NONE,
}

/// Converts a `PhotonImage` (photon-rs native format) into an
/// `image::DynamicImage` (image crate format).
pub fn photon_to_dynamic(photon_image: &PhotonImage) -> DynamicImage {
    let mut img = helpers::dyn_image_from_raw(photon_image);
    img = ImageRgba8(img.to_rgba8());
    img
}

/// Converts an `image::DynamicImage` into a `PhotonImage`.
pub fn dynamic_to_photon(dynamic_image: &DynamicImage) -> PhotonImage {
    let image_buffer: Vec<u8> = (*dynamic_image).clone().into_bytes();
    PhotonImage::new(image_buffer, dynamic_image.width(), dynamic_image.height())
}
