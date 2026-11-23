//! Decode / resize / normalize / patchify uploaded images.
//!
//! Layout matches HuggingFace `Qwen2VLImageProcessorFast` / Qwen3-VL:
//! - smart resize to multiples of `patch_size * merge_size`
//! - rescale `/255` then normalize with mean/std `0.5`
//! - flatten patches in spatial-merge block order with inner dim `C·T·H·W`

use image::GenericImageView;

use crate::config::VisionConfig;

/// Preprocessed image ready for the vision encoder.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Flattened patches `[num_patches * patch_embed_dim]`.
    pub pixel_patches: Vec<f32>,
    pub grid_thw: (usize, usize, usize),
    pub num_merge_tokens: usize,
    pub num_patches: usize,
    pub patch_embed_dim: usize,
}

/// Image processor matching Qwen3-VL defaults.
#[derive(Debug, Clone)]
pub struct ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::from_config(&VisionConfig {
            depth: 24,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_heads: 16,
            in_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 2048,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: vec![5, 11, 17],
        })
    }
}

impl ImageProcessor {
    pub fn from_config(cfg: &VisionConfig) -> Self {
        Self {
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            spatial_merge_size: cfg.spatial_merge_size,
            in_channels: cfg.in_channels,
            // HF shortest_edge is 65536. longest_edge is 16_777_216, but dense
            // vision attention is O(patches²) and OOMs past ~1M pixels on typical
            // desktop GPUs — keep a practical cap until flash/chunked attn lands.
            min_pixels: 65_536,
            max_pixels: 1_003_520,
        }
    }

    fn spatial_unit(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// Decode image bytes (PNG/JPEG/WebP) into patches.
    pub fn preprocess_bytes(&self, bytes: &[u8]) -> Result<ImageInput, String> {
        let img = image::load_from_memory(bytes).map_err(|e| format!("decode image: {e}"))?;
        self.preprocess_dynamic_image(img)
    }

    pub fn preprocess_path(&self, path: &std::path::Path) -> Result<ImageInput, String> {
        let img = image::open(path).map_err(|e| format!("open image: {e}"))?;
        self.preprocess_dynamic_image(img)
    }

    fn preprocess_dynamic_image(
        &self,
        img: image::DynamicImage,
    ) -> Result<ImageInput, String> {
        let (orig_w, orig_h) = img.dimensions();
        let (new_h, new_w) = self.smart_resize(orig_h as usize, orig_w as usize);
        // HF uses bicubic.
        let img = img.resize_exact(
            new_w as u32,
            new_h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let rgb = img.to_rgb8();
        // rescale 1/255 then (x - 0.5) / 0.5  ≡  x/127.5 - 1
        let pixels: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
            .collect();

        // Duplicate frame for temporal_patch_size=2 (image path).
        let frames = vec![pixels.clone(), pixels];
        let patch_embed_dim =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let grid_h = new_h / self.patch_size;
        let grid_w = new_w / self.patch_size;
        let grid_t = 1usize;
        let patches = self.extract_patches(&frames, new_h, new_w);
        let num_patches = grid_t * grid_h * grid_w;
        let merge = self.spatial_merge_size;
        let num_merge_tokens = grid_t * (grid_h / merge) * (grid_w / merge);

        Ok(ImageInput {
            pixel_patches: patches,
            grid_thw: (grid_t, grid_h, grid_w),
            num_merge_tokens,
            num_patches,
            patch_embed_dim,
        })
    }

    /// HF `smart_resize`: keep aspect ratio, dims divisible by `factor`.
    fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = self.spatial_unit();
        let mut h_bar = ((height as f64 / factor as f64).round() as usize).max(1) * factor;
        let mut w_bar = ((width as f64 / factor as f64).round() as usize).max(1) * factor;
        if h_bar * w_bar > self.max_pixels {
            let beta = ((height * width) as f64 / self.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        } else if h_bar * w_bar < self.min_pixels {
            let beta = (self.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
        }
        (h_bar, w_bar)
    }

    /// Extract patches in Qwen-VL merge-block order with inner layout `C·T·ps·ps`.
    fn extract_patches(&self, frames: &[Vec<f32>], height: usize, width: usize) -> Vec<f32> {
        let ps = self.patch_size;
        let tps = self.temporal_patch_size;
        let merge = self.spatial_merge_size;
        let grid_h = height / ps;
        let grid_w = width / ps;
        let channels = self.in_channels;
        let patch_dim = channels * tps * ps * ps;

        let mut out = Vec::with_capacity((grid_h * grid_w) * patch_dim);

        for hb in 0..(grid_h / merge) {
            for wb in 0..(grid_w / merge) {
                for hi in 0..merge {
                    for wi in 0..merge {
                        let gy = hb * merge + hi;
                        let gx = wb * merge + wi;
                        let mut patch = vec![0.0f32; patch_dim];
                        for c in 0..channels {
                            for t in 0..tps {
                                let frame = &frames[t.min(frames.len() - 1)];
                                for py in 0..ps {
                                    for px in 0..ps {
                                        let y = gy * ps + py;
                                        let x = gx * ps + px;
                                        let src = ((y * width + x) * channels + c) as usize;
                                        // HF Conv3d input layout: C, T, H, W
                                        let dst =
                                            (((c * tps + t) * ps + py) * ps + px) as usize;
                                        patch[dst] = frame[src];
                                    }
                                }
                            }
                        }
                        out.extend(patch);
                    }
                }
            }
        }
        out
    }
}
