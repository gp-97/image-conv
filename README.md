<h1 align="center">image-conv</h1>

<div align="center">
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/gp-97/image-conv.svg)](https://github.com/gp-97/image-conv/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/gp-97/image-conv.svg)](https://github.com/gp-97/image-conv/pulls)
  ![Crates.io (version)](https://img.shields.io/crates/dv/image-conv/0.1.1)

</div>

---

<p align="center">A high performance Rust library for image convolution.</p>

- [Documentation](https://docs.rs/image-conv/0.1.1/image_conv/)
- [crate.io](https://crates.io/crates/image-conv)

### Some example ouputs
|Image|Sobel-X|Sobel-Y|Scharr-X|Scharr-Y|Laplacian|Median|Gaussian|Denoise|
|-----|-------|-------|--------|--------|-------|------|----------|-------|
  

### Example usage
- Apply horizontal Sobel kernel
```rust
  use image_conv::conv;
  use image_conv::Filter;
  use photon_rs::native::{open_image, save_image};

  fn main() {
      // Open an image
      let mut img = open_image("img.jpg").expect("No such file found");

      // Create a filter
      let sobel_x: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
      let filter = Filter::from(sobel_x, 3, 3);

      // Apply convolution    
      let img_conv = conv::convolution(&img, filter, 1, "uniform", 1);
      save_image(img_conv, "img_conv.jpg");
  }
```