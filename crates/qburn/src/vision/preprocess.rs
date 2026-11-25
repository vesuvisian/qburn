//! Decode / resize / normalize / patchify uploaded images and video frames.
//!
//! Layout matches HuggingFace `Qwen2VLImageProcessorFast` / Qwen3-VL:
//! - smart resize to multiples of `patch_size * merge_size`
//! - rescale `/255` then normalize with mean/std `0.5`
//! - flatten patches in temporal-group × spatial-merge block order with
//!   inner dim `C·T·H·W`
//!
//! Images duplicate one RGB frame so `temporal_patch_size=2` yields `grid_t=1`.
//! Video samples frames, pads to a multiple of `temporal_patch_size`, and
//! emits `grid_t = n_frames / temporal_patch_size`.

use image::GenericImageView;

use crate::config::VisionConfig;

/// Preprocessed vision input ready for the vision encoder (image or video).
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Flattened patches `[num_patches * patch_embed_dim]`.
    pub pixel_patches: Vec<f32>,
    pub grid_thw: (usize, usize, usize),
    pub num_merge_tokens: usize,
    pub num_patches: usize,
    pub patch_embed_dim: usize,
}

/// Image / video processor matching Qwen3-VL defaults.
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
            // Vision attention uses Burn CubeCL attention (see model/sdpa.rs) but
            // is still O(patches²) FLOPs; keep a practical pixel cap for TTFT.
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
        // Duplicate frame for temporal_patch_size=2 (image path → grid_t=1).
        self.preprocess_video_frames(&[img])
    }

    /// Preprocess an ordered list of decoded frames as one Qwen3-VL video.
    ///
    /// Frames are resized to a shared smart-resize size. If the count is not
    /// divisible by `temporal_patch_size`, the last frame is repeated (HF
    /// temporal pad). A single frame is duplicated (image path).
    pub fn preprocess_video_frames(
        &self,
        frames: &[image::DynamicImage],
    ) -> Result<ImageInput, String> {
        if frames.is_empty() {
            return Err("video requires at least one frame".into());
        }

        let (orig_w, orig_h) = frames[0].dimensions();
        let tps = self.temporal_patch_size.max(1);
        // Pad count up front so video spatial budget sees the temporal length
        // that will actually be patchified.
        let mut n_frames = frames.len();
        while n_frames % tps != 0 {
            n_frames += 1;
        }
        // Multi-frame video: HF total-pixel budget over t×h×w. Single still
        // keeps the image smart_resize caps.
        let (new_h, new_w) = if frames.len() > 1 {
            self.smart_resize_video(n_frames, orig_h as usize, orig_w as usize)
        } else {
            self.smart_resize(orig_h as usize, orig_w as usize)
        };

        let mut rgb_frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);
        for frame in frames {
            let resized = frame.resize_exact(
                new_w as u32,
                new_h as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let rgb = resized.to_rgb8();
            // rescale 1/255 then (x - 0.5) / 0.5  ≡  x/127.5 - 1
            let pixels: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
                .collect();
            rgb_frames.push(pixels);
        }

        // Image path (1 frame) / odd count: pad by repeating the last frame so
        // len % temporal_patch_size == 0 (still → grid_t=1).
        while rgb_frames.len() % tps != 0 {
            let last = rgb_frames.last().expect("non-empty").clone();
            rgb_frames.push(last);
        }

        let patch_embed_dim =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let grid_h = new_h / self.patch_size;
        let grid_w = new_w / self.patch_size;
        let grid_t = rgb_frames.len() / tps;
        let patches = self.extract_patches(&rgb_frames, new_h, new_w);
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

    /// Decode each frame from image bytes (PNG/JPEG/WebP), then patchify as video.
    pub fn preprocess_video_frame_bytes(&self, frames: &[&[u8]]) -> Result<ImageInput, String> {
        if frames.is_empty() {
            return Err("video requires at least one frame".into());
        }
        let mut imgs = Vec::with_capacity(frames.len());
        for (i, bytes) in frames.iter().enumerate() {
            let img = image::load_from_memory(bytes)
                .map_err(|e| format!("decode video frame {i}: {e}"))?;
            imgs.push(img);
        }
        self.preprocess_video_frames(&imgs)
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

    /// HF Qwen3-VL video `smart_resize`: budget is `t_bar × h × w`, so longer
    /// clips shrink spatially instead of stacking full-res frames (which
    /// drives greedy decode into repetitive collapse).
    fn smart_resize_video(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
    ) -> (usize, usize) {
        let factor = self.spatial_unit();
        let tps = self.temporal_patch_size.max(1);
        // Match HF video processor defaults.
        let min_pixels = 128 * 32 * 32;
        let max_pixels = 32 * 32 * 768;

        let mut h = height;
        let mut w = width;
        if h < factor || w < factor {
            let scale = (factor as f64 / h as f64).max(factor as f64 / w as f64);
            h = ((h as f64) * scale) as usize;
            w = ((w as f64) * scale) as usize;
        }
        let mut h_bar = ((h as f64 / factor as f64).round() as usize).max(1) * factor;
        let mut w_bar = ((w as f64 / factor as f64).round() as usize).max(1) * factor;
        let t_bar = ((num_frames as f64 / tps as f64).round() as usize).max(1) * tps;

        if t_bar * h_bar * w_bar > max_pixels {
            let beta = ((num_frames * height * width) as f64 / max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        } else if t_bar * h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (num_frames * height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize).max(1) * factor;
        }
        (h_bar, w_bar)
    }

    /// Extract patches in Qwen-VL order: temporal groups × spatial-merge blocks,
    /// inner layout `C·T·ps·ps`.
    fn extract_patches(&self, frames: &[Vec<f32>], height: usize, width: usize) -> Vec<f32> {
        let ps = self.patch_size;
        let tps = self.temporal_patch_size.max(1);
        let merge = self.spatial_merge_size;
        let grid_h = height / ps;
        let grid_w = width / ps;
        let grid_t = frames.len() / tps;
        let channels = self.in_channels;
        let patch_dim = channels * tps * ps * ps;

        let mut out = Vec::with_capacity(grid_t * grid_h * grid_w * patch_dim);

        for tg in 0..grid_t {
            for hb in 0..(grid_h / merge) {
                for wb in 0..(grid_w / merge) {
                    for hi in 0..merge {
                        for wi in 0..merge {
                            let gy = hb * merge + hi;
                            let gx = wb * merge + wi;
                            let mut patch = vec![0.0f32; patch_dim];
                            for c in 0..channels {
                                for t in 0..tps {
                                    let frame_idx = tg * tps + t;
                                    let frame = &frames[frame_idx.min(frames.len() - 1)];
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
        }
        out
    }
}

/// Uniformly sample frame indices like HF `Qwen3VLVideoProcessor.sample_frames`.
///
/// `fps` and `num_frames` are mutually exclusive (pass one as `None`). When
/// `fps` is set, `num_frames = round(total / src_fps * fps)`, then clamped to
/// `[min_frames, max_frames, total]`.
pub fn sample_frame_indices(
    total_frames: usize,
    src_fps: f64,
    fps: Option<f64>,
    num_frames: Option<usize>,
    min_frames: usize,
    max_frames: usize,
) -> Result<Vec<usize>, String> {
    if total_frames == 0 {
        return Err("video has zero frames".into());
    }
    if fps.is_some() && num_frames.is_some() {
        return Err("`fps` and `num_frames` are mutually exclusive".into());
    }

    let n = if let Some(n) = num_frames {
        n.max(1)
    } else {
        let fps = fps.unwrap_or(2.0);
        let src = if src_fps > 0.0 { src_fps } else { 24.0 };
        let estimated = ((total_frames as f64) / src * fps).round() as usize;
        estimated.max(1)
    };

    let n = n
        .max(min_frames.min(total_frames))
        .min(max_frames)
        .min(total_frames)
        .max(1);

    if n == 1 {
        return Ok(vec![0]);
    }
    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let idx = ((i as f64) * ((total_frames - 1) as f64) / ((n - 1) as f64)).round() as usize;
        indices.push(idx.min(total_frames - 1));
    }
    Ok(indices)
}

/// Default sampling caps matching HF video processor (CLI-oriented).
pub const VIDEO_MIN_FRAMES: usize = 4;
/// Keep temporal depth modest — long greedy decode still loops past ~16 frames.
pub const VIDEO_MAX_FRAMES_CLI: usize = 16;
pub const VIDEO_MAX_FRAMES_WEB: usize = 8;
pub const VIDEO_DEFAULT_FPS: f64 = 2.0;

/// Average frame timestamps within each temporal patch (HF Qwen3-VL).
///
/// Pads the last timestamp when `len % temporal_patch_size != 0`, then returns
/// one time per temporal group: mean of the first/last frame in the patch.
pub fn temporal_group_timestamps(frame_times: &[f64], temporal_patch_size: usize) -> Vec<f64> {
    let tps = temporal_patch_size.max(1);
    if frame_times.is_empty() {
        return Vec::new();
    }
    let mut times = frame_times.to_vec();
    while times.len() % tps != 0 {
        times.push(*times.last().expect("non-empty"));
    }
    times
        .chunks(tps)
        .map(|chunk| (chunk[0] + chunk[chunk.len() - 1]) / 2.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(w: u32, h: u32, v: u8) -> image::DynamicImage {
        let img = RgbImage::from_fn(w, h, |_, _| Rgb([v, v, v]));
        image::DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn temporal_timestamps_average_within_patch() {
        let ts = temporal_group_timestamps(&[0.0, 0.5, 1.0, 1.5], 2);
        assert_eq!(ts.len(), 2);
        assert!((ts[0] - 0.25).abs() < 1e-9);
        assert!((ts[1] - 1.25).abs() < 1e-9);
    }

    #[test]
    fn image_path_grid_t_is_one() {
        let proc = ImageProcessor {
            max_pixels: 65_536,
            min_pixels: 65_536,
            ..ImageProcessor::default()
        };
        // 256×256 → divisible by 32, within min/max after smart_resize.
        let out = proc.preprocess_video_frames(&[solid(256, 256, 128)]).unwrap();
        assert_eq!(out.grid_thw.0, 1);
        assert_eq!(out.num_patches, out.grid_thw.0 * out.grid_thw.1 * out.grid_thw.2);
        assert_eq!(
            out.num_merge_tokens,
            out.grid_thw.0 * (out.grid_thw.1 / 2) * (out.grid_thw.2 / 2)
        );
    }

    #[test]
    fn even_frames_set_grid_t() {
        let proc = ImageProcessor {
            max_pixels: 65_536,
            min_pixels: 65_536,
            ..ImageProcessor::default()
        };
        let frames: Vec<_> = (0..4).map(|i| solid(256, 256, 40 * i)).collect();
        let out = proc.preprocess_video_frames(&frames).unwrap();
        assert_eq!(out.grid_thw.0, 2); // 4 frames / temporal_patch_size 2
        assert_eq!(out.num_patches, 2 * out.grid_thw.1 * out.grid_thw.2);
        assert_eq!(
            out.num_merge_tokens,
            2 * (out.grid_thw.1 / 2) * (out.grid_thw.2 / 2)
        );
    }

    #[test]
    fn odd_frames_pad_to_even() {
        let proc = ImageProcessor {
            max_pixels: 65_536,
            min_pixels: 65_536,
            ..ImageProcessor::default()
        };
        let frames: Vec<_> = (0..3).map(|i| solid(256, 256, 40 * i)).collect();
        let out = proc.preprocess_video_frames(&frames).unwrap();
        assert_eq!(out.grid_thw.0, 2); // 3 → pad to 4 → grid_t=2
    }

    #[test]
    fn sample_indices_fps() {
        let idx = sample_frame_indices(100, 25.0, Some(2.0), None, 4, 64).unwrap();
        // 100/25*2 = 8 frames
        assert_eq!(idx.len(), 8);
        assert_eq!(idx[0], 0);
        assert_eq!(*idx.last().unwrap(), 99);
    }

    #[test]
    fn sample_indices_num_frames() {
        let idx = sample_frame_indices(50, 30.0, None, Some(4), 4, 64).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[3], 49);
    }

    #[test]
    fn sample_rejects_both_fps_and_num() {
        assert!(sample_frame_indices(10, 24.0, Some(2.0), Some(4), 4, 64).is_err());
    }
}
