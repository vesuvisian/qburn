//! qburn — Burn-native Qwen3-VL for native CLI and browser wasm.
//!
//! # Overview
//!
//! Hand-written modules for **Qwen3-VL-2B-Instruct**: image / video preprocess →
//! vision encoder → multimodal splice + M-RoPE → Qwen3 decoder with KV cache.
//!
//! - Native: enable a GPU feature (`wgpu`, `metal`, `cuda`) plus `cli` for the binary.
//! - Browser: the `web` crate enables `webgpu`, calls [`resolve_device_async`], then
//!   [`load::load_from_burnpack_bytes`]. Generation is async (`into_data_async`);
//!   matmul uses Burn 0.23 Hardware-aware CubeCL selection on wasm (see
//!   repo `patches/README.md`).
//!
//! Weights: load Hugging Face safetensors with [`load::load_from_hf_dir`], or converted
//! burnpacks with [`load::load_from_burnpack`] / [`load::load_from_burnpack_bytes`].
//! Offline conversion lives in the `qburn-converter` crate
//! (`WeightQuant::{Fp16,Q4,Q4Mix}`).
//!
//! See the repo `README.md` and `web/static/architecture.html` for end-to-end usage.
#![recursion_limit = "512"]

pub mod config;
pub mod device;
pub mod generate;
pub mod load;
pub mod model;
pub mod quant;
pub mod tokenizer;
pub mod vision;
pub mod vl;

pub use config::Qwen3VlConfig;
pub use device::{backend_name, resolve_device, resolve_device_async};
pub use quant::WeightQuant;
pub use tokenizer::QburnTokenizer;
pub use vision::{
    sample_frame_indices, temporal_group_timestamps, ImageInput, ImageProcessor, VIDEO_DEFAULT_FPS,
    VIDEO_MAX_FRAMES_CLI, VIDEO_MAX_FRAMES_WEB, VIDEO_MIN_FRAMES,
};
pub use vl::{with_length_budget, DescribeTimings, PrefillProbe, Qwen3Vl};

pub use burn::prelude::Device;
