//! Integration tests for image convolution.
//!
//! Applies common image processing kernels (Sobel, Scharr, Laplacian, Median,
//! Gaussian, Denoise) to a test image and saves the output. The separable
//! detection tests verify that the 1D/2D decomposition produces identical results.

#[cfg(test)]
use image_conv::conv;
use image_conv::{Filter, PaddingType};
use photon_rs::monochrome;
use photon_rs::native::{open_image, save_image};
#[test]
fn test_convolution_sobel_x() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_sobelX.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
    let filter = Filter::from(sobel_x, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_sobel_y() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_sobelY.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let sobel_y: Vec<f32> = vec![1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0];
    let filter = Filter::from(sobel_y, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_scharr_x() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_scharrX.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let scharr_x: Vec<f32> = vec![3.0, 0.0, -3.0, 10.0, 0.0, -10.0, 3.0, 0.0, -3.0];
    let filter = Filter::from(scharr_x, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_scharr_y() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_scharrY.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let scharr_y: Vec<f32> = vec![3.0, 10.0, 3.0, 0.0, 0.0, 0.0, -3.0, -10.0, -3.0];
    let filter = Filter::from(scharr_y, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_laplacian() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_laplacian.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let laplacian: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let filter = Filter::from(laplacian, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_median() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_median.jpg";
    let img = open_image(inp_path).expect("No such file found");

    let median: Vec<f32> = vec![0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111];
    let filter = Filter::from(median, 3, 3);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}

#[test]
fn test_convolution_gaussian_7x7() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_gaussian7x7.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);
    let gaussian: Vec<f32> = vec![
        1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0, 2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0, 2.0,
        4.0, 8.0, 16.0, 8.0, 4.0, 2.0, 2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0, 1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 1.0, 1.0,
    ];
    let gaussian = gaussian.into_iter().map(|val| val / 273.0).collect();
    let filter = Filter::from(gaussian, 7, 7);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}
#[test]
fn test_convolution_denoise() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_denoise.jpg";
    let img = open_image(inp_path).expect("No such file found");

    let denoise = vec![
        2_f32, 4.0, 5.0, 4.0, 2.0, 4.0, 9.0, 12.0, 9.0, 4.0, 5.0, 12.0, 15.0, 12.0, 5.0, 4.0, 9.0, 12.0, 9.0, 4.0,
        2_f32, 4.0, 5.0, 4.0, 2.0,
    ];
    let denoise: Vec<f32> = denoise.into_iter().map(|val| val / 139.0).collect();
    let filter = Filter::from(denoise, 5, 5);
    let img = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    save_image(img, op_path).unwrap();
}

#[test]
fn test_separable_correctness() {
    let pixels = vec![
        10, 20, 30, 255,  50, 60, 70, 255,
        90, 100, 110, 255,  130, 140, 150, 255,
    ];
    let img = photon_rs::PhotonImage::new(pixels, 2, 2);

    let separable: Vec<f32> = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    let filter = Filter::from(separable, 3, 3);

    let result = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    assert_eq!(result.get_width(), 2);
    assert_eq!(result.get_height(), 2);
    let raw = result.get_raw_pixels();
    assert_eq!(raw.len(), 2 * 2 * 4);
}

#[test]
fn test_separable_detection() {
    let kernel = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0];
    let f = Filter::from(kernel.clone(), 3, 2);
    let (col, row) = f.try_separable().expect("should be separable");
    assert_eq!(col.len(), 2);
    assert_eq!(row.len(), 3);

    // Verify outer product matches original kernel
    for i in 0..2 {
        for j in 0..3 {
            let val = col[i] * row[j];
            let expected = kernel[i * 3 + j];
            assert!((val - expected).abs() < 1e-4, "mismatch at [{i}][{j}]: {val} != {expected}");
        }
    }

    let nonsep = vec![1.0, 2.0, 3.0, 4.0];
    let f = Filter::from(nonsep, 2, 2);
    assert!(f.try_separable().is_none());
}

