use photon_rs::PhotonImage;

pub fn uniform(img: &PhotonImage, padding: u32) -> PhotonImage {
    let image_buffer = img.get_raw_pixels();
    let img_width = img.get_width();
    let img_height = img.get_height();

    let mut img_padded_buffer = Vec::<u8>::new();
    let width_padded: u32 = img_width + 2 * padding;
    let height_padded: u32 = img_height + 2 * padding;

    for _ in 0..((width_padded + 1) * padding) {
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(255);
    }

    for i in (0..image_buffer.len()).step_by(4) {
        if (i / 4) % img_width as usize == 0 && i != 0 {
            for _ in (0..2 * padding).step_by(1) {
                img_padded_buffer.push(0);
                img_padded_buffer.push(0);
                img_padded_buffer.push(0);
                img_padded_buffer.push(255);
            }
        }
        img_padded_buffer.push(image_buffer[i]);
        img_padded_buffer.push(image_buffer[i + 1]);
        img_padded_buffer.push(image_buffer[i + 2]);
        img_padded_buffer.push(image_buffer[i + 3]);
    }

    for _ in 0..((width_padded + 1) * padding) {
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(255);
    }

    PhotonImage::new(img_padded_buffer, width_padded, height_padded)
}

pub fn left(img: &PhotonImage, padding: u32) -> PhotonImage {
    let image_buffer = img.get_raw_pixels();
    let img_width = img.get_width();
    let img_height = img.get_height();

    let mut img_padded_buffer = Vec::<u8>::new();
    let width_padded: u32 = img_width + padding;

    for i in 0..img_height as usize {
        let img_slice =
            image_buffer[(i * 4 * img_width as usize)..(i + 1) * 4 * img_width as usize].to_owned();

        for _ in 0..padding {
            img_padded_buffer.push(0);
            img_padded_buffer.push(0);
            img_padded_buffer.push(0);
            img_padded_buffer.push(255);
        }
        for x in img_slice {
            img_padded_buffer.push(x);
        }
    }
    PhotonImage::new(img_padded_buffer, width_padded, img_height)
}
pub fn right(img: &PhotonImage, padding: u32) -> PhotonImage {
    let image_buffer = img.get_raw_pixels();
    let img_width = img.get_width();
    let img_height = img.get_height();

    let mut img_padded_buffer = Vec::<u8>::new();
    let width_padded: u32 = img_width + padding;

    for i in 0..img_height as usize {
        let img_slice =
            image_buffer[(i * 4 * img_width as usize)..(i + 1) * 4 * img_width as usize].to_owned();
        for x in img_slice {
            img_padded_buffer.push(x);
        }
        for _ in 0..padding {
            img_padded_buffer.push(0);
            img_padded_buffer.push(0);
            img_padded_buffer.push(0);
            img_padded_buffer.push(255);
        }
    }
    PhotonImage::new(img_padded_buffer, width_padded, img_height)
}
pub fn top(img: &PhotonImage, padding: u32) -> PhotonImage {
    let image_buffer = img.get_raw_pixels();
    let img_width = img.get_width();
    let img_height = img.get_height();

    let height_padded: u32 = img_height + padding;
    let mut img_padded_buffer: Vec<u8> = Vec::<u8>::new();

    for _ in 0..(padding * img_width) {
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(255);
    }

    for i in (0..image_buffer.len()).step_by(4) {
        img_padded_buffer.push(image_buffer[i]);
        img_padded_buffer.push(image_buffer[i + 1]);
        img_padded_buffer.push(image_buffer[i + 2]);
        img_padded_buffer.push(image_buffer[i + 3]);
    }

    PhotonImage::new(img_padded_buffer, img_width, height_padded)
}

pub fn bottom(img: &PhotonImage, padding: u32) -> PhotonImage {
    let image_buffer = img.get_raw_pixels();
    let img_width = img.get_width();
    let img_height = img.get_height();

    let height_padded: u32 = img_height + padding;
    let mut img_padded_buffer: Vec<u8> = Vec::<u8>::new();

    for i in (0..image_buffer.len()).step_by(4) {
        img_padded_buffer.push(image_buffer[i]);
        img_padded_buffer.push(image_buffer[i + 1]);
        img_padded_buffer.push(image_buffer[i + 2]);
        img_padded_buffer.push(image_buffer[i + 3]);
    }

    for _ in 0..(padding * img_width) {
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(0);
        img_padded_buffer.push(255);
    }

    PhotonImage::new(img_padded_buffer, img_width, height_padded)
}
