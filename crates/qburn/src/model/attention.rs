use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use super::cache::KvCache;
use super::rms_norm::RmsNorm;
use super::rope::{apply_rotary, RotaryEmbedding};

/// Grouped-query attention KV cache pair.
pub struct AttentionKvCache {
    pub k_cache: KvCache,
    pub v_cache: KvCache,
}

impl AttentionKvCache {
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            k_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
            v_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.reset();
        self.v_cache.reset();
    }
}

/// Multi-head attention with GQA and QK-Norm.
#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    num_kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        device: &Device,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            qkv_proj: LinearConfig::new(d_model, q_dim + 2 * kv_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(q_dim, d_model)
                .with_bias(false)
                .init(device),
            q_norm: RmsNorm::new(head_dim, rms_norm_eps, device),
            k_norm: RmsNorm::new(head_dim, rms_norm_eps, device),
            num_heads,
            num_kv_heads,
            head_dim,
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
        let (q, k, v) = self.project(x);
        let (q, k) = rope.apply(q, k, start_pos);
        self.attend(q, k, v, mask, cache)
    }

    pub fn forward_with_cos_sin(
        &self,
        x: Tensor<3>,
        cos: Tensor<2>,
        sin: Tensor<2>,
        mask: Option<Tensor<4>>,
        cache: &mut AttentionKvCache,
    ) -> Tensor<3> {
        let (q, k, v) = self.project(x);
        let (q, k) = apply_rotary(q, k, cos, sin);
        self.attend(q, k, v, mask, cache)
    }

    fn project(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch, seq_len, _] = x.dims();
        let qkv = self.qkv_proj.forward(x);
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv
            .clone()
            .slice(s![0..batch, 0..seq_len, 0..q_dim])
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = qkv
            .clone()
            .slice(s![0..batch, 0..seq_len, q_dim..q_dim + kv_dim])
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = qkv
            .slice(s![0..batch, 0..seq_len, q_dim + kv_dim..q_dim + 2 * kv_dim])
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        let q = self
            .q_norm
            .forward(q.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(k.reshape([batch * seq_len * self.num_kv_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.swap_dims(1, 2);
        (q, k, v)
    }

    fn attend(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<Tensor<4>>,
        cache: &mut AttentionKvCache,
    ) -> Tensor<3> {
        let [batch, _heads, seq_len, _] = q.dims();
        let k = cache.k_cache.forward(k);
        let v = cache.v_cache.forward(v);

        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);

        let scale = (self.head_dim as f64).sqrt().recip();
        let mut attn = q.matmul(k.transpose()) * scale;
        if let Some(mask) = mask {
            attn = attn + mask;
        }
        let attn = activation::softmax(attn, 3);
        let out = attn.matmul(v);
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward(out)
    }

    pub fn load_weights(
        &mut self,
        qkv: Tensor<2>,
        o: Tensor<2>,
        q_norm: Tensor<1>,
        k_norm: Tensor<1>,
    ) {
        self.qkv_proj.weight = Param::from_tensor(qkv);
        self.o_proj.weight = Param::from_tensor(o);
        self.q_norm.set_weight(q_norm);
        self.k_norm.set_weight(k_norm);
    }
}

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, num_kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([batch, num_kv_heads, n_rep, seq_len, head_dim])
        .reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
}
