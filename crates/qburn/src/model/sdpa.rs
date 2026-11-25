//! Dense scaled dot-product attention.
//!
//! CubeCL's fused `attention` op (with `is_causal`) was tried for Metal TTFT and
//! left captions OK on Metal, but the WebGPU/wasm path produced greedy repetition
//! collapse. Stay on explicit matmul + softmax + optional additive mask so the
//! browser demo matches the proven numerics.

use burn::prelude::*;
use burn::tensor::activation;

/// Run attention over `[B, H, Q, D]` / `[B, H, K, D]`.
///
/// `mask`, when present, is broadcastable to `[B, H, Q, K]` (typically
/// `[1, 1, Q, K]` with `0` / `-inf` for causal prefill).
pub fn scaled_dot_product_attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f64,
    mask: Option<&Tensor<4>>,
) -> Tensor<4> {
    let mut attn = q.matmul(k.transpose()) * scale;
    if let Some(m) = mask {
        attn = attn + m.clone();
    }
    let attn = activation::softmax(attn, 3);
    attn.matmul(v)
}
