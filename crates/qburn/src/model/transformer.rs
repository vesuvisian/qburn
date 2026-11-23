use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::config::TextConfig;

use super::attention::{AttentionKvCache, MultiHeadAttention};
use super::mlp::FeedForward;
use super::mrope::MRopeEmbedding;
use super::rms_norm::RmsNorm;
use super::rope::RotaryEmbedding;

/// Causal attention mask: `-inf` above the diagonal.
pub fn build_causal_mask(q_len: usize, kv_len: usize, device: &Device) -> Tensor<2> {
    let mut data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let q_pos = kv_len - q_len + i;
        for j in 0..kv_len {
            if j > q_pos {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<1>::from_floats(data.as_slice(), device).reshape([q_len, kv_len])
}

#[derive(Module, Debug)]
pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: MultiHeadAttention,
    post_attention_layernorm: RmsNorm,
    mlp: FeedForward,
}

impl TransformerBlock {
    pub fn new(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            input_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            self_attn: MultiHeadAttention::new(
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
                device,
            ),
            post_attention_layernorm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            mlp: FeedForward::new(cfg.hidden_size, cfg.intermediate_size, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RotaryEmbedding,
        mask: Option<Tensor<4>>,
        cache: &mut AttentionKvCache,
        start_pos: usize,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self.self_attn.forward(h, rope, mask, cache, start_pos);
        let x = h + residual;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(x);
        let h = self.mlp.forward(h);
        h + residual
    }

    pub fn forward_with_cos_sin(
        &self,
        x: Tensor<3>,
        cos: Tensor<2>,
        sin: Tensor<2>,
        mask: Option<Tensor<4>>,
        cache: &mut AttentionKvCache,
    ) -> Tensor<3> {
        let residual = x.clone();
        let h = self.input_layernorm.forward(x);
        let h = self
            .self_attn
            .forward_with_cos_sin(h, cos, sin, mask, cache);
        let x = h + residual;

        let residual = x.clone();
        let h = self.post_attention_layernorm.forward(x);
        let h = self.mlp.forward(h);
        h + residual
    }
}

/// Qwen3 text decoder (dense).
#[derive(Module, Debug)]
pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: Linear,
    #[module(skip)]
    pub config: TextConfig,
}

impl TextDecoder {
    pub fn new(cfg: TextConfig, device: &Device) -> Self {
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| TransformerBlock::new(&cfg, device))
            .collect();
        let embed_tokens = EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device);
        let lm_head = LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
            .with_bias(false)
            .init(device);
        Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, device),
            lm_head,
            config: cfg,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn embed(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.embed_tokens.forward(tokens)
    }

    pub fn forward(
        &self,
        tokens: Tensor<2, Int>,
        rope: &RotaryEmbedding,
        mask: Option<Tensor<2>>,
        caches: &mut [AttentionKvCache],
        start_pos: usize,
    ) -> Tensor<3> {
        let mut x = self.embed(tokens);
        let mask = mask.map(|m| m.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0));
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, rope, mask.clone(), &mut caches[i], start_pos);
        }
        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Forward from precomputed embeddings with M-RoPE cos/sin (VL path).
    pub fn forward_from_embeddings(
        &self,
        embeddings: Tensor<3>,
        cos: Tensor<2>,
        sin: Tensor<2>,
        mask: Option<Tensor<2>>,
        caches: &mut [AttentionKvCache],
        deepstack: Option<(&[Tensor<3>], &Tensor<3>, &[usize])>,
    ) -> Tensor<3> {
        let mut x = embeddings;
        let mask = mask.map(|m| m.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0));
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cos_sin(
                x,
                cos.clone(),
                sin.clone(),
                mask.clone(),
                &mut caches[i],
            );
            if let Some((features, visual_mask, ds_layers)) = deepstack {
                if let Some(ds_idx) = ds_layers.iter().position(|&l| l == i) {
                    if ds_idx < features.len() {
                        x = x + features[ds_idx].clone() * visual_mask.clone();
                    }
                }
            }
        }
        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Share embed weights with `lm_head`.
    ///
    /// Embedding is `[vocab, hidden]`; Burn `Linear` is `[in, out]` = `[hidden, vocab]`,
    /// so the tied copy is transposed.
    pub fn tie_embeddings(&mut self) {
        let w = self.embed_tokens.weight.val().transpose();
        self.lm_head.weight = Param::from_tensor(w);
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut TransformerBlock {
        &mut self.layers[idx]
    }

    pub fn set_embed_weight(&mut self, w: Tensor<2>) {
        self.embed_tokens.weight = Param::from_tensor(w);
    }

    pub fn set_norm_weight(&mut self, w: Tensor<1>) {
        self.norm.set_weight(w);
    }

    pub fn set_lm_head_weight(&mut self, w: Tensor<2>) {
        self.lm_head.weight = Param::from_tensor(w);
    }

    pub fn load_layer_weights(
        &mut self,
        idx: usize,
        qkv: Tensor<2>,
        o: Tensor<2>,
        q_norm: Tensor<1>,
        k_norm: Tensor<1>,
        gate_up: Tensor<2>,
        down: Tensor<2>,
        input_ln: Tensor<1>,
        post_ln: Tensor<1>,
    ) {
        let layer = &mut self.layers[idx];
        layer.self_attn.load_weights(qkv, o, q_norm, k_norm);
        layer.mlp.load_weights(gate_up, down);
        layer.input_layernorm.set_weight(input_ln);
        layer.post_attention_layernorm.set_weight(post_ln);
    }
}

/// Helper to build empty KV caches for a decoder.
pub fn new_caches(
    decoder: &TextDecoder,
    batch: usize,
    max_seq: usize,
    device: &Device,
) -> Vec<AttentionKvCache> {
    (0..decoder.num_layers())
        .map(|_| {
            AttentionKvCache::new(
                batch,
                decoder.config.num_key_value_heads,
                max_seq,
                decoder.config.head_dim,
                device,
            )
        })
        .collect()
}

/// Build M-RoPE for the text config.
pub fn new_mrope(cfg: &TextConfig, mrope_section: [usize; 3], device: &Device) -> MRopeEmbedding {
    MRopeEmbedding::new(mrope_section, cfg.rope_theta, cfg.head_dim, device)
}
