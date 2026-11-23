//! Language decoder and vision encoder modules.
//!
//! Submodules map closely to Hugging Face `Qwen3VL*` building blocks:
//! attention / MLP / RMSNorm / RoPE / M-RoPE / KV cache / ViT + merger.

mod attention;
mod cache;
mod mlp;
mod mrope;
mod rms_norm;
mod rope;
mod transformer;
mod vision;

pub use attention::{AttentionKvCache, MultiHeadAttention};
pub use cache::KvCache;
pub use mlp::FeedForward;
pub use mrope::{build_position_ids, MRopeEmbedding};
pub use rms_norm::RmsNorm;
pub use rope::{apply_rotary, RotaryEmbedding};
pub use transformer::{build_causal_mask, new_caches, new_mrope, TextDecoder, TransformerBlock};
pub use vision::{VisionEncoder, VisionOutput};
