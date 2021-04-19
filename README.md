<h1 align="center">image-conv</h1>
<div align="center">

  <img alt="GitHub" src="https://img.shields.io/github/license/gp-97/image-conv">

</div>
<div align="center">
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/gp-97/image-conv.svg)](https://github.com/gp-97/image-conv/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/gp-97/image-conv.svg)](https://github.com/gp-97/image-conv/pulls)
  ![Crates.io (recent)](https://img.shields.io/crates/dr/image-conv?style=plastic)

</div>

---

<p align="center">A high performance Rust library for image convolution.</p>

- [Documentation](https://docs.rs/image-conv/0.1.3/image_conv/index.html)
- [crate.io](https://crates.io/crates/image-conv)

### Example usage
- Apply horizontal Sobel filter:
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


### Some example ouputs
|Original|
|--------|
|![Original](assets/t5.jpg)|

|Sobel-X|Sobel-Y|
|-------|-------|
|![Sobel-X](assets/t5_sobelX.jpg)|![Sobel-Y](assets/t5_sobelY.jpg)|

|Scharr-X|Scharr-Y|
|--------|--------|
![Scharr-X](assets/t5_scharrX.jpg)|![Scharr-Y](assets/t5_scharrY.jpg)|

|Laplacian|Median|
|-------|------|
|![Laplacian](assets/t5_laplacian.jpg)|![Median](assets/t5_median.jpg)|

|Gaussian|Denoise|
|---------|-------|
|![Gaussian](assets/t5_gaussian7x7.jpg)|![Denoise](assets/t5_denoise.jpg)|  
