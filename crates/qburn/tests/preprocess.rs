use qburn::vision::ImageProcessor;

#[test]
fn preprocess_tiny_png_produces_aligned_grid() {
    use image::{ImageBuffer, Rgb};
    use std::io::Cursor;

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(48, 32, |x, y| Rgb([(x % 255) as u8, (y % 255) as u8, 80]));
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode png");
    let png = buf.into_inner();

    let proc = ImageProcessor::default();
    let out = proc.preprocess_bytes(&png).expect("preprocess");
    let unit = proc.patch_size * proc.spatial_merge_size;
    let (_t, gh, gw) = out.grid_thw;
    assert_eq!((gh * proc.patch_size) % unit, 0);
    assert_eq!((gw * proc.patch_size) % unit, 0);
    assert_eq!(out.num_patches, gh * gw);
    assert_eq!(
        out.num_merge_tokens,
        (gh / proc.spatial_merge_size) * (gw / proc.spatial_merge_size)
    );
    assert_eq!(
        out.pixel_patches.len(),
        out.num_patches * out.patch_embed_dim
    );
}
