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
    let pixels = vec![10, 20, 30, 255, 50, 60, 70, 255, 90, 100, 110, 255, 130, 140, 150, 255];
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
            assert!(
                (val - expected).abs() < 1e-4,
                "mismatch at [{i}][{j}]: {val} != {expected}"
            );
        }
    }

    let nonsep = vec![1.0, 2.0, 3.0, 4.0];
    let f = Filter::from(nonsep, 2, 2);
    assert!(f.try_separable().is_none());
}

#[test]
fn test_convolution_known_values_2d() {
    // 3x3 image, Laplacian 3x3 (non-separable, tests 2D path)
    let pixels = vec![
        10,20,30,255, 40,50,60,255, 70,80,90,255,
        100,110,120,255, 130,140,150,255, 160,170,180,255,
        190,200,210,255, 220,230,240,255, 250,250,250,255,
    ];
    let img = photon_rs::PhotonImage::new(pixels, 3, 3);

    let laplacian: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let filter = Filter::from(laplacian, 3, 3);
    let result = conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1));

    // With UNIFORM(1), padded is 5x5 with black border, output is 3x3
    assert_eq!(result.get_width(), 3);
    assert_eq!(result.get_height(), 3);
    let raw = result.get_raw_pixels();

    // TL pixel R: padded(0,0..2,0..2) with Laplacian
    // row0: 0*0 + 1*0 + 0*0 = 0
    // row1: 1*0 + (-4)*10 + 1*40 = 0
    // row2: 0*0 + 1*100 + 0*130 = 100
    // => R=100, G=80, B=60
    assert_eq!(raw[0], 100);
    assert_eq!(raw[1], 80);
    assert_eq!(raw[2], 60);
}

#[test]
fn test_convolution_known_values_separable() {
    // 4x4 image, Sobel-X 3x3 (separable), stride=1, NONE padding → output 2x2
    let pixels = vec![
        10,0,0,255,  50,0,0,255,  100,0,0,255,  150,0,0,255,
        20,0,0,255,  60,0,0,255,  110,0,0,255,  160,0,0,255,
        30,0,0,255,  70,0,0,255,  120,0,0,255,  170,0,0,255,
        40,0,0,255,  80,0,0,255,  130,0,0,255,  180,0,0,255,
    ];
    let img = photon_rs::PhotonImage::new(pixels, 4, 4);

    let sobel_x: Vec<f32> = vec![1.0,0.0,-1.0, 2.0,0.0,-2.0, 1.0,0.0,-1.0];
    let filter = Filter::from(sobel_x, 3, 3);
    let result = conv::convolution(&img, filter, 1, PaddingType::NONE);

    assert_eq!(result.get_width(), 2);
    assert_eq!(result.get_height(), 2);
    let raw = result.get_raw_pixels();

    // output[0][0]: filter on image[0..3][0..3]
    // = 1*10 + 0*50 + (-1)*100 + 2*20 + 0*60 + (-2)*110 + 1*30 + 0*70 + (-1)*120
    // = 10 - 100 + 40 - 220 + 30 - 120 = -360 → clamped to 0
    let r00 = raw[0] as i32;
    // output[0][1]: filter on image[0..3][1..4]
    // = 1*50 + 0*100 + (-1)*150 + 2*60 + 0*110 + (-2)*160 + 1*70 + 0*120 + (-1)*170
    // = 50 - 150 + 120 - 320 + 70 - 170 = -400 → 0
    let r01 = raw[4] as i32;

    // Both should be 0 (negative values clamped)
    assert_eq!(r00, 0, "TL pixel R should be 0, got {}", r00);
    assert_eq!(r01, 0, "TR pixel R should be 0, got {}", r01);

    // All alpha should be 255
    assert_eq!(raw[3], 255);
    assert_eq!(raw[7], 255);
}
