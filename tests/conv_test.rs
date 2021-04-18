#[cfg(test)]
use image_conv::conv;
use image_conv::Filter;
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
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_sobel_y() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_sobelY.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let sobel_y: Vec<f32> = vec![1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0];
    let filter = Filter::from(sobel_y, 3, 3);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_scharr_x() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_scharrX.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let scharr_x: Vec<f32> = vec![3.0, 0.0, -3.0, 10.0, 0.0, -10.0, 3.0, 0.0, -3.0];
    let filter = Filter::from(scharr_x, 3, 3);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_scharr_y() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_scharrY.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let scharr_y: Vec<f32> = vec![3.0, 10.0, 3.0, 0.0, 0.0, 0.0, -3.0, -10.0, -3.0];
    let filter = Filter::from(scharr_y, 3, 3);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_laplacian() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_laplacian.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);

    let laplacian: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let filter = Filter::from(laplacian, 3, 3);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_median() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_median.jpg";
    let img = open_image(inp_path).expect("No such file found");

    let median: Vec<f32> = vec![
        0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111, 0.1111,
    ];
    let filter = Filter::from(median, 3, 3);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}

#[test]
fn test_convolution_gaussian_7x7() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_gaussian7x7.jpg";
    let mut img = open_image(inp_path).expect("No such file found");
    monochrome::grayscale(&mut img);
    let gaussian: Vec<f32> = vec![
        1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 2.0, 2.0, 1.0, 2.0, 2.0, 4.0, 8.0,
        4.0, 2.0, 2.0, 2.0, 4.0, 8.0, 16.0, 8.0, 4.0, 2.0, 2.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0, 1.0,
        2.0, 2.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0,
    ];
    let gaussian = gaussian.into_iter().map(|val| val / 273.0).collect();
    let filter = Filter::from(gaussian, 7, 7);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
#[test]
fn test_convolution_denoise() {
    let inp_path = "tests/test_assets/t5.jpg";
    let op_path = "tests/test_assets/t5_denoise.jpg";
    let img = open_image(inp_path).expect("No such file found");

    let denoise = vec![
        2_f32, 4.0, 5.0, 4.0, 2.0, 4.0, 9.0, 12.0, 9.0, 4.0, 5.0, 12.0, 15.0, 12.0, 5.0, 4.0, 9.0,
        12.0, 9.0, 4.0, 2_f32, 4.0, 5.0, 4.0, 2.0,
    ];
    let denoise = denoise.into_iter().map(|val| val / 139.0).collect();
    let filter = Filter::from(denoise, 5, 5);
    let img = conv::convolution(&img, filter, 1, "uniform", 1);

    save_image(img, op_path);
}
