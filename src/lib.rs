use image::{
    DynamicImage::{self, ImageRgba8},
    GenericImageView,
};
use photon_rs::{helpers, PhotonImage};
use prettytable::{Cell, Row, Table};

pub struct Filter {
    width: usize,
    height: usize,
    kernel: Vec<f32>,
}

impl Filter {
    pub fn new(width: usize, height: usize) -> Self {
        let mut kernel = Vec::<f32>::new();
        Vec::resize(&mut kernel, width * height, 0_f32);
        Self {
            width,
            height,
            kernel,
        }
    }
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

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn kernel(&self) -> Vec<f32> {
        self.kernel.clone()
    }

    pub fn get_element(&self, x: usize, y: usize) -> Option<f32> {
        let element_pos = x * self.width + y;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            None
        } else {
            Some(self.kernel[element_pos])
        }
    }
    pub fn set_value_at_pos(&mut self, val: f32, position: (usize, usize)) {
        let element_pos = position.0 * self.width + position.1;
        if self.kernel.is_empty() || element_pos >= self.kernel.len() {
            eprintln!("[ERROR]: Index out of bound");
            std::process::exit(1);
        } else {
            self.kernel[element_pos] = val;
        }
    }

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

pub fn photon_to_dynamic(photon_image: &PhotonImage) -> DynamicImage {
    let mut img = helpers::dyn_image_from_raw(photon_image);
    img = ImageRgba8(img.to_rgba8());
    img
}

pub fn dynamic_to_photon(dynamic_image: &DynamicImage) -> PhotonImage {
    let image_buffer: Vec<u8> = (*dynamic_image).clone().into_bytes();
    PhotonImage::new(image_buffer, dynamic_image.width(), dynamic_image.height())
}

pub mod conv;
