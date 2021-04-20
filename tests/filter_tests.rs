#[cfg(test)]
use image_conv::Filter;
#[test]
fn test_filter_init() {
    let k = Filter::new(5, 5);
    assert_eq!(k.kernel(), vec![0 as f32; 25]);
}

#[test]
fn test_kernel_val_assignment() {
    let mut k = Filter::new(3, 3);
    let val: f32 = 10.0;
    for i in 0..3 {
        for j in 0..3 {
            k.set_value_at_pos(val, (i, j));
        }
    }

    assert_eq!(k.kernel(), vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
}
