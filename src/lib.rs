//! A high performance Rust library for image convolution.
//!
//!
//! ## Example
//! ```no_run
//! use image_conv::conv;
//! use image_conv::Filter;
//! use photon_rs::native::{open_image, save_image};
//!
//! fn main() {
//!     // Open an image
//!     let mut img = open_image("img.jpg").expect("No such file found");
//!
//!     // Create a filter
//!     let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
//!     let filter = Filter::from(sobel_x, 3, 3);
//!     
//!     // Apply convolution    
//!     let img_conv = conv::convolution(&img, filter, 1, "uniform", 1);
//!     save_image(img_conv, "img_conv.jpg");
//! }
//! ```

use image::{
    DynamicImage::{self, ImageRgba8},
    GenericImageView,
};
use photon_rs::{helpers, PhotonImage};
use prettytable::{Cell, Row, Table};

/// Filter is a struct for holding a kernel
/// and its associated meta-data
///
/// # Struct members
/// * `width` - Holds Filter's width. (usize)
/// * `height` - Holds Filter's height. (usize)
/// * `kernel` - Holds Filter's kernel/data. (32-bit floating point Vector)
pub struct Filter {
    width: usize,
    height: usize,
    kernel: Vec<f32>,
}

impl Filter {
    /// Initializes and returns a new Filter object
    pub fn new(width: usize, height: usize) -> Self {
        let mut kernel = Vec::<f32>::new();
        Vec::resize(&mut kernel, width * height, 0_f32);
        Self {
            width,
            height,
            kernel,
        }
    }
    /// Creates and returns a Filter object from a vector of kernel data
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
    /// Getter for Filter's width
    pub fn width(&self) -> usize {
        self.width
    }
    /// Getter for Filter's height
    pub fn height(&self) -> usize {
        self.height
    }
    /// Getter for Filter's kernel/data
    pub fn kernel(&self) -> Vec<f32> {
        self.kernel.clone()
    }
    /// Returns data element present at (x, y) location of Filter, as an Option Enum
    pub fn get_element(&self, x: usize, y: usize) -> Option<f32> {
        let element_pos = x * self.width + y;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            None
        } else {
            Some(self.kernel[element_pos])
        }
    }
    /// Sets data element at (x, y) location of Filter.
    pub fn set_value_at_pos(&mut self, val: f32, position: (usize, usize)) {
        let element_pos = position.0 * self.width + position.1;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            eprintln!("[ERROR]: Index out of bound");
            std::process::exit(1);
        } else {
            self.kernel[element_pos] = val;
        }
    }
    /// Displays the Filter as a pretty 2D-matrix
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

/// For convertion of PhotonImage into Dynamic image
pub fn photon_to_dynamic(photon_image: &PhotonImage) -> DynamicImage {
    let mut img = helpers::dyn_image_from_raw(photon_image);
    img = ImageRgba8(img.to_rgba8());
    img
}

/// For convertion of Dynamic image into PhotonImage
pub fn dynamic_to_photon(dynamic_image: &DynamicImage) -> PhotonImage {
    let image_buffer: Vec<u8> = (*dynamic_image).clone().into_bytes();
    PhotonImage::new(image_buffer, dynamic_image.width(), dynamic_image.height())
}

pub mod conv;
pub mod padding;
