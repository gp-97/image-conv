//! Image convolution

use crate::padding::uniform;
use crate::{Filter, PaddingType};
use photon_rs::PhotonImage;

fn convolve(img_padded: &PhotonImage, filter: &Filter, width_conv: u32, height_conv: u32, stride: u32) -> PhotonImage {
    let raw_pixel_padded = img_padded.get_raw_pixels();
    let width_padded = img_padded.get_width() as usize;
    let height_padded = img_padded.get_height() as usize;
    let mut img_conv = vec![];

    let filter_width = filter.width();
    let filter_height = filter.height();

    let mut pixel = 0_usize;
    let image_end = (width_padded * height_padded * 4) as usize;
    let step = 4 * stride as usize;

    while pixel < image_end - 4 {
        if pixel != 0 && ((pixel / 4) % width_padded) > (width_padded - filter_width) {
            pixel = ((pixel / 4) / width_padded + stride as usize) * width_padded * 4;

            if (pixel / 4) / width_padded + filter_height > height_padded {
                break;
            }
        }
        let mut img_conv_r: f32 = 0_f32;
        let mut img_conv_g: f32 = 0_f32;
        let mut img_conv_b: f32 = 0_f32;

        for x in 0..filter_width {
            for y in 0..filter_height {
                let kernel_element_val = filter
                    .get_element(x, y)
                    .expect("[ERROR]: Tried to access out-of-bounds value in the filter");
                let img_pixel_r = raw_pixel_padded[x * width_padded * 4 + pixel + y * 4];
                let img_pixel_g = raw_pixel_padded[x * width_padded * 4 + pixel + y * 4 + 1];
                let img_pixel_b = raw_pixel_padded[x * width_padded * 4 + pixel + y * 4 + 2];

                img_conv_r += img_pixel_r as f32 * kernel_element_val;
                img_conv_g += img_pixel_g as f32 * kernel_element_val;
                img_conv_b += img_pixel_b as f32 * kernel_element_val;
            }
        }

        img_conv_r = f32::clamp(img_conv_r, 0.0, 255.0);
        img_conv_g = f32::clamp(img_conv_g, 0.0, 255.0);
        img_conv_b = f32::clamp(img_conv_b, 0.0, 255.0);

        img_conv.push(img_conv_r as u8);
        img_conv.push(img_conv_g as u8);
        img_conv.push(img_conv_b as u8);
        img_conv.push(255_u8);

        pixel += step;
    }

    for _ in (img_conv.len()..(width_conv * height_conv * 4) as usize).step_by(1) {
        img_conv.push(255_u8);
        img_conv.push(255_u8);
        img_conv.push(255_u8);
        img_conv.push(255_u8);
    }
    
    #[cfg(debug_assertions)]    
    println!("Convolution done...");
    
    PhotonImage::new(img_conv, width_conv, height_conv)
}

fn adjust_convolution_params(
    img: &PhotonImage,
    img_padded: &PhotonImage,
    filter: &Filter,
    stride: u32,
    padding: PaddingType,
) -> PhotonImage {
    let mut img_conv_width: u32;
    let mut img_conv_height: u32;

    match padding {
        PaddingType::UNIFORM(pad_amt) => {
            img_conv_width = img.get_width() - filter.width() as u32 + 2 * pad_amt;
            if img_conv_width % stride != 0 {
                eprintln!("[WARNING]: stride value not suitable. Convolution may fail.");
            }
            img_conv_width /= stride;
            img_conv_width += 1;

            img_conv_height = img.get_height() - filter.height() as u32 + 2 * pad_amt;
            if img_conv_height % stride != 0 {
                eprintln!("[WARNING]: stride value not suitable. Convolution may fail.");
            }
            img_conv_height /= stride;
            img_conv_height += 1;
        }

        PaddingType::NONE => {
            img_conv_width = img.get_width() - filter.width() as u32;
            if img_conv_width % stride != 0 {
                eprintln!("[WARNING]: stride value not suitable. Convolution may fail.");
            }
            img_conv_width /= stride;
            img_conv_width += 1;

            img_conv_height = img.get_height() - filter.height() as u32;
            if img_conv_height % stride != 0 {
                eprintln!("[WARNING]: stride value not suitable. Convolution may fail.");
            }
            img_conv_height /= stride;
            img_conv_height += 1;
        }
    };

    convolve(img_padded, filter, img_conv_width, img_conv_height, stride)
}

/// Applies convoultion on the image
///
/// # Arguments
/// * `img` - A Photon Image.
/// * `filter` - A Filter object.
/// * `stride` - Stride value for convolution.
/// * `padding` - Padding type (Enum defined in lib.rs).
///
/// # Example
///
/// ```no_run
/// // For example, to apply horizontal sobel filter:
/// use image_conv::conv;
/// use image_conv::{Filter, PaddingType};
///
/// let img = photon_rs::native::open_image("img.jpg").expect("No such file found");
/// let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
/// let filter = Filter::from(sobel_x, 3, 3);     
/// let img_conv = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));
///```
pub fn convolution(img: &PhotonImage, filter: Filter, stride: u32, padding: PaddingType) -> PhotonImage {
    match stride {
        0 => {
            eprintln!("[ERROR]: Stride provided = 0");
            std::process::exit(1);
        }

        1 => match padding {
            PaddingType::UNIFORM(padding_amt) => {
                let img_padded = uniform(&img, padding_amt);
                adjust_convolution_params(img, &img_padded, &filter, stride, padding)
            }
            PaddingType::NONE => {
                let img_padded = img.clone();
                adjust_convolution_params(img, &img_padded, &filter, stride, padding)
            }
        },
        _ => match padding {
            PaddingType::UNIFORM(padding_amt) => {
                let img_padded = uniform(&img, padding_amt);
                adjust_convolution_params(img, &img_padded, &filter, stride, padding)
            }
            PaddingType::NONE => {
                let img_padded = img.clone();
                adjust_convolution_params(img, &img_padded, &filter, stride, padding)
            }
        },
    }
}
