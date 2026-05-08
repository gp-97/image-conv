//! Criterion benchmarks for convolution performance.
//!
//! Benchmarks compare:
//!   * Separable vs non-separable kernels of the same size
//!   * Various image sizes (100×100, 500×500, 1000×1000)
//!   * Kernel sizes from 3×3 to 15×15
//!
//! Run with: `cargo bench`

use criterion::{criterion_group, criterion_main, Criterion};
use image_conv::conv;
use image_conv::{Filter, PaddingType};
use std::hint::black_box;

fn dummy_image(width: u32, height: u32) -> photon_rs::PhotonImage {
    let size = (width * height * 4) as usize;
    let mut pixels = Vec::with_capacity(size);
    for i in 0..size {
        pixels.push((i % 256) as u8);
    }
    photon_rs::PhotonImage::new(pixels, width, height)
}

/// Build a proper separable Gaussian from binomial coefficients
fn separable_gaussian(size: usize) -> Filter {
    assert!(size % 2 == 1, "size must be odd");
    let n = size - 1;
    let mut coeffs = vec![1.0_f32];
    for k in 0..n {
        coeffs.push(coeffs[k] * (n - k) as f32 / (k + 1) as f32);
    }
    let sum: f32 = coeffs.iter().sum();
    let row: Vec<f32> = coeffs.iter().map(|v| v / sum).collect();

    let mut kernel = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            kernel.push(row[i] * row[j]);
        }
    }
    Filter::from(kernel, size, size)
}

fn bench_conv(c: &mut Criterion) {
    // --- 3x3 separable Sobel-X (auto-detected) ---
    let sobel: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
    let f3x3 = Filter::from(sobel, 3, 3);

    // --- 3x3 non-separable Laplacian ---
    let laplacian: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let f3x3_nonsep = Filter::from(laplacian, 3, 3);

    let mut group = c.benchmark_group("conv_3x3");
    group.bench_function("separable_sobel_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f3x3.clone()), 1, PaddingType::NONE))
    });
    group.bench_function("nonseparable_laplacian_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f3x3_nonsep.clone()), 1, PaddingType::NONE))
    });
    group.finish();

    // --- 7x7 separable Gaussian (auto-detected) ---
    let f7x7_sep = separable_gaussian(7);

    // --- 7x7 non-separable (random-ish) ---
    let nonsep7: Vec<f32> = (0..49).map(|i| ((i % 13) as f32 - 6.0) / 49.0).collect();
    let f7x7_nonsep = Filter::from(nonsep7, 7, 7);

    let mut group = c.benchmark_group("conv_7x7");
    group.bench_function("separable_gaussian_100x100", |b| {
        let img = dummy_image(100, 100);
        b.iter(|| conv::convolution(black_box(&img), black_box(f7x7_sep.clone()), 1, PaddingType::NONE))
    });
    group.bench_function("nonseparable_100x100", |b| {
        let img = dummy_image(100, 100);
        b.iter(|| conv::convolution(black_box(&img), black_box(f7x7_nonsep.clone()), 1, PaddingType::NONE))
    });
    group.bench_function("separable_gaussian_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f7x7_sep.clone()), 1, PaddingType::NONE))
    });
    group.bench_function("nonseparable_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f7x7_nonsep.clone()), 1, PaddingType::NONE))
    });
    group.finish();

    // --- 9x9 separable Gaussian ---
    let f9x9_sep = separable_gaussian(9);

    let mut group = c.benchmark_group("conv_9x9");
    group.bench_function("separable_gaussian_500x500", |b| {
        let img = dummy_image(500, 500);
        b.iter(|| conv::convolution(black_box(&img), black_box(f9x9_sep.clone()), 1, PaddingType::NONE))
    });
    group.bench_function("separable_gaussian_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f9x9_sep.clone()), 1, PaddingType::NONE))
    });
    group.finish();

    // --- 15x15 separable Gaussian (big speedup) ---
    let f15x15_sep = separable_gaussian(15);

    let mut group = c.benchmark_group("conv_15x15");
    group.bench_function("separable_gaussian_1000x1000", |b| {
        let img = dummy_image(1000, 1000);
        b.iter(|| conv::convolution(black_box(&img), black_box(f15x15_sep.clone()), 1, PaddingType::NONE))
    });
    group.finish();
}

criterion_group!(benches, bench_conv);
criterion_main!(benches);
