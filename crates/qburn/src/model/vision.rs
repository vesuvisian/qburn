use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::TensorData;

use crate::config::VisionConfig;

use super::rope::apply_rotary;
use super::sdpa::scaled_dot_product_attention;

/// CPU-side cache for absolute ViT pos embeds: one GPU→CPU table readback per
/// encoder, then reused bilinear grids keyed by `(gh, gw, gt, merge)`.
/// (GPU Burn interpolate drifted on WebGPU and collapsed greedy captions.)
#[derive(Debug, Default)]
struct PosEmbedCpuCache {
    /// Flattened square table `[side*side*c]` (align_corners bilinear source).
    table: Option<Vec<f32>>,
    table_c: usize,
    table_side: usize,
    grids: HashMap<(usize, usize, usize, usize), Vec<f32>>,
}

/// Output of the vision encoder including DeepStack features.
pub struct VisionOutput {
    /// Merged visual tokens projected to LLM hidden size: `[1, num_tokens, out_hidden]`.
    pub embeddings: Tensor<3>,
    /// DeepStack features (same shape), one per configured index.
    pub deepstack: Vec<Tensor<3>>,
}

#[derive(Module, Debug)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn new(hidden: usize, intermediate: usize, device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(hidden, intermediate).with_bias(true).init(device),
            fc2: LinearConfig::new(intermediate, hidden).with_bias(true).init(device),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let x = activation::gelu(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    #[module(skip)]
    num_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl VisionAttention {
    fn new(hidden: usize, num_heads: usize, device: &Device) -> Self {
        let head_dim = hidden / num_heads;
        Self {
            qkv: LinearConfig::new(hidden, 3 * hidden).with_bias(true).init(device),
            proj: LinearConfig::new(hidden, hidden).with_bias(true).init(device),
            num_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<2>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<2> {
        let [seq, _] = x.dims();
        let qkv = self.qkv.forward(x);
        let q = qkv
            .clone()
            .slice(s![0..seq, 0..self.num_heads * self.head_dim])
            .reshape([1, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .slice(s![0..seq, self.num_heads * self.head_dim..2 * self.num_heads * self.head_dim])
            .reshape([1, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .slice(s![0..seq, 2 * self.num_heads * self.head_dim..3 * self.num_heads * self.head_dim])
            .reshape([1, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let (q, k) = apply_rotary(q, k, cos, sin);
        let scale = (self.head_dim as f64).sqrt().recip();
        let out = scaled_dot_product_attention(q, k, v, scale, None);
        let out = out
            .swap_dims(1, 2)
            .reshape([seq, self.num_heads * self.head_dim]);
        self.proj.forward(out)
    }
}

#[derive(Module, Debug)]
struct VisionBlock {
    norm1: LayerNormSimple,
    attn: VisionAttention,
    norm2: LayerNormSimple,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, device: &Device) -> Self {
        Self {
            norm1: LayerNormSimple::new(cfg.hidden_size, 1e-6, device),
            attn: VisionAttention::new(cfg.hidden_size, cfg.num_heads, device),
            norm2: LayerNormSimple::new(cfg.hidden_size, 1e-6, device),
            mlp: VisionMlp::new(cfg.hidden_size, cfg.intermediate_size, device),
        }
    }

    fn forward(&self, x: Tensor<2>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<2> {
        let h = self.attn.forward(self.norm1.forward(x.clone()), cos, sin);
        let x = x + h;
        let h = self.mlp.forward(self.norm2.forward(x.clone()));
        x + h
    }
}

#[derive(Module, Debug)]
struct LayerNormSimple {
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
    #[module(skip)]
    eps: f64,
}

impl LayerNormSimple {
    fn new(size: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([size], device)),
            bias: Param::from_tensor(Tensor::zeros([size], device)),
            eps,
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let mean = x.clone().mean_dim(1);
        let diff = x - mean;
        let variance = diff.clone().powf_scalar(2.0).mean_dim(1);
        let normed = diff * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze_dim::<2>(0)
            + self.bias.val().unsqueeze_dim::<2>(0)
    }
}

#[derive(Module, Debug)]
struct PatchMerger {
    norm: LayerNormSimple,
    linear_fc1: Linear,
    linear_fc2: Linear,
    #[module(skip)]
    hidden: usize,
    /// HF: norm after spatial merge (DeepStack) vs before (main merger).
    #[module(skip)]
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, use_postshuffle_norm: bool, device: &Device) -> Self {
        let hidden = cfg.merged_hidden_size();
        let norm_size = if use_postshuffle_norm {
            hidden
        } else {
            cfg.hidden_size
        };
        Self {
            norm: LayerNormSimple::new(norm_size, 1e-6, device),
            linear_fc1: LinearConfig::new(hidden, hidden).with_bias(true).init(device),
            linear_fc2: LinearConfig::new(hidden, cfg.out_hidden_size)
                .with_bias(true)
                .init(device),
            hidden,
            use_postshuffle_norm,
        }
    }

    fn forward(&self, x: Tensor<2>, spatial_merge: usize) -> Tensor<2> {
        // Matches HF Qwen3VLVisionPatchMerger:
        //   postshuffle: reshape → LayerNorm(merged) → MLP
        //   otherwise:   LayerNorm(patch) → reshape → MLP
        let [n, c] = x.dims();
        let merge = spatial_merge * spatial_merge;
        assert!(n % merge == 0, "patch count must be divisible by merge unit");
        let h = if self.use_postshuffle_norm {
            self.norm.forward(x.reshape([n / merge, merge * c]))
        } else {
            self.norm
                .forward(x)
                .reshape([n / merge, merge * c])
        };
        let h = activation::gelu(self.linear_fc1.forward(h));
        self.linear_fc2.forward(h)
    }
}

/// Qwen3-VL vision encoder (ViT + merger + DeepStack).
#[derive(Module, Debug)]
pub struct VisionEncoder {
    patch_embed: Linear,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    #[module(skip)]
    pub config: VisionConfig,
    /// Shared across Module clones; cleared when `pos_embed` weights are loaded.
    #[module(skip)]
    pos_cache: Arc<Mutex<PosEmbedCpuCache>>,
}

impl VisionEncoder {
    pub fn new(cfg: VisionConfig, device: &Device) -> Self {
        let blocks = (0..cfg.depth)
            .map(|_| VisionBlock::new(&cfg, device))
            .collect();
        let deepstack_mergers = cfg
            .deepstack_visual_indexes
            .iter()
            .map(|_| PatchMerger::new(&cfg, true, device))
            .collect();
        Self {
            // HF stores this as Conv3d `[out, in, t, h, w]`; we flatten to Linear + bias.
            patch_embed: LinearConfig::new(cfg.patch_embed_dim(), cfg.hidden_size)
                .with_bias(true)
                .init(device),
            pos_embed: EmbeddingConfig::new(cfg.num_position_embeddings, cfg.hidden_size)
                .init(device),
            blocks,
            merger: PatchMerger::new(&cfg, false, device),
            deepstack_mergers,
            config: cfg,
            pos_cache: Arc::new(Mutex::new(PosEmbedCpuCache::default())),
        }
    }

    fn clear_pos_cache(&self) {
        if let Ok(mut cache) = self.pos_cache.lock() {
            *cache = PosEmbedCpuCache::default();
        }
    }

    pub async fn forward(
        &self,
        patches: Tensor<2>,
        grid_thw: (usize, usize, usize),
        device: &Device,
    ) -> VisionOutput {
        let (gt, gh, gw) = grid_thw;
        let mut x = self.patch_embed.forward(patches);

        // Absolute position embeddings via bilinear interp of learned square grid.
        // Must follow spatial-merge block order (same as patchify).
        let pos = interpolate_pos_embed(
            &self.pos_embed,
            self.config.num_position_embeddings,
            gh,
            gw,
            gt,
            self.config.spatial_merge_size,
            device,
            &self.pos_cache,
        )
        .await;
        x = x + pos;

        let (cos, sin) = vision_rope_2d(
            gh,
            gw,
            gt,
            self.config.spatial_merge_size,
            self.config.head_dim(),
            device,
        );
        let ds_indexes = &self.config.deepstack_visual_indexes;
        let mut deepstack = Vec::new();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(x, &cos, &sin);
            if let Some(ds_i) = ds_indexes.iter().position(|&i| i == layer_idx) {
                let feat = self.deepstack_mergers[ds_i].forward(x.clone(), self.config.spatial_merge_size);
                deepstack.push(feat.unsqueeze_dim::<3>(0));
            }
        }

        let merged = self
            .merger
            .forward(x, self.config.spatial_merge_size)
            .unsqueeze_dim::<3>(0);

        VisionOutput {
            embeddings: merged,
            deepstack,
        }
    }

    /// Load HuggingFace `model.visual.*` tensors (best-effort; skips missing keys).
    pub fn load_hf_tensors(
        &mut self,
        tensors: &std::collections::HashMap<String, burn::tensor::TensorData>,
        device: &Device,
    ) {
        let get2 = |key: &str| -> Option<Tensor<2>> {
            let d = tensors.get(key)?;
            // Only true Linear weights are rank-2. Conv3d patch embed is handled separately.
            if d.rank() != 2 {
                return None;
            }
            // HF Linear is [out, in] → Burn row-major [in, out]
            Some(Tensor::<2>::from_data(d.clone(), device).transpose())
        };
        let get1 = |key: &str| -> Option<Tensor<1>> {
            let d = tensors.get(key)?;
            if d.rank() != 1 {
                return None;
            }
            Some(Tensor::<1>::from_data(d.clone(), device))
        };
        let get_emb = |key: &str| -> Option<Tensor<2>> {
            let d = tensors.get(key)?;
            if d.rank() != 2 {
                return None;
            }
            Some(Tensor::<2>::from_data(d.clone(), device))
        };

        // Conv3d `[out, in, t, h, w]` → Linear `[in*t*h*w, out]` (+ bias).
        if let Some(w) = tensors
            .get("model.visual.patch_embed.proj.weight")
            .or_else(|| tensors.get("visual.patch_embed.proj.weight"))
        {
            if let Some(linear) = conv3d_weight_to_linear(w, device) {
                self.patch_embed.weight = Param::from_tensor(linear);
            }
        }
        if let Some(b) = get1("model.visual.patch_embed.proj.bias")
            .or_else(|| get1("visual.patch_embed.proj.bias"))
        {
            self.patch_embed.bias = Some(Param::from_tensor(b));
        }
        if let Some(w) = get_emb("model.visual.pos_embed.weight")
            .or_else(|| get_emb("visual.pos_embed.weight"))
        {
            self.pos_embed.weight = Param::from_tensor(w);
            self.clear_pos_cache();
        }

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let p = format!("model.visual.blocks.{i}");
            let p2 = format!("visual.blocks.{i}");
            let prefix = if tensors.keys().any(|k| k.starts_with(&p)) {
                p
            } else {
                p2
            };
            if let Some(w) = get1(&format!("{prefix}.norm1.weight")) {
                block.norm1.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.norm1.bias")) {
                block.norm1.bias = Param::from_tensor(b);
            }
            if let Some(w) = get2(&format!("{prefix}.attn.qkv.weight")) {
                block.attn.qkv.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.attn.qkv.bias")) {
                block.attn.qkv.bias = Some(Param::from_tensor(b));
            }
            if let Some(w) = get2(&format!("{prefix}.attn.proj.weight")) {
                block.attn.proj.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.attn.proj.bias")) {
                block.attn.proj.bias = Some(Param::from_tensor(b));
            }
            if let Some(w) = get1(&format!("{prefix}.norm2.weight")) {
                block.norm2.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.norm2.bias")) {
                block.norm2.bias = Param::from_tensor(b);
            }
            if let Some(w) = get2(&format!("{prefix}.mlp.linear_fc1.weight"))
                .or_else(|| get2(&format!("{prefix}.mlp.fc1.weight")))
            {
                block.mlp.fc1.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.mlp.linear_fc1.bias"))
                .or_else(|| get1(&format!("{prefix}.mlp.fc1.bias")))
            {
                block.mlp.fc1.bias = Some(Param::from_tensor(b));
            }
            if let Some(w) = get2(&format!("{prefix}.mlp.linear_fc2.weight"))
                .or_else(|| get2(&format!("{prefix}.mlp.fc2.weight")))
            {
                block.mlp.fc2.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.mlp.linear_fc2.bias"))
                .or_else(|| get1(&format!("{prefix}.mlp.fc2.bias")))
            {
                block.mlp.fc2.bias = Some(Param::from_tensor(b));
            }
        }

        // Merger + deepstack: key names vary across exports; load when present.
        let load_merger = |m: &mut PatchMerger, prefix: &str| {
            if let Some(w) = get1(&format!("{prefix}.norm.weight")) {
                m.norm.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.norm.bias")) {
                m.norm.bias = Param::from_tensor(b);
            }
            if let Some(w) = get2(&format!("{prefix}.linear_fc1.weight")) {
                m.linear_fc1.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.linear_fc1.bias")) {
                m.linear_fc1.bias = Some(Param::from_tensor(b));
            }
            if let Some(w) = get2(&format!("{prefix}.linear_fc2.weight")) {
                m.linear_fc2.weight = Param::from_tensor(w);
            }
            if let Some(b) = get1(&format!("{prefix}.linear_fc2.bias")) {
                m.linear_fc2.bias = Some(Param::from_tensor(b));
            }
        };
        load_merger(&mut self.merger, "model.visual.merger");
        for (i, m) in self.deepstack_mergers.iter_mut().enumerate() {
            load_merger(m, &format!("model.visual.deepstack_merger_list.{i}"));
        }
    }
}

/// Flatten HF Conv3d patch-embed weight `[out, in, t, h, w]` into Burn Linear `[in*t*h*w, out]`.
fn conv3d_weight_to_linear(data: &TensorData, device: &Device) -> Option<Tensor<2>> {
    if data.rank() != 5 {
        return None;
    }
    let [out_c, in_c, t, h, w] = data.shape().dims::<5>();
    let in_flat = in_c * t * h * w;
    let vals: Vec<f32> = data.try_to_vec_as::<f32>().ok()?;
    let flat = TensorData::new(vals, [out_c, in_flat]);
    Some(Tensor::<2>::from_data(flat, device).transpose())
}

async fn interpolate_pos_embed(
    pos_embed: &Embedding,
    num_pos: usize,
    gh: usize,
    gw: usize,
    gt: usize,
    spatial_merge: usize,
    device: &Device,
    cache: &Arc<Mutex<PosEmbedCpuCache>>,
) -> Tensor<2> {
    let side = (num_pos as f64).sqrt() as usize;
    let merge = spatial_merge.max(1);
    let grid_key = (gh, gw, gt, merge);

    // Fast path: reuse a previously interpolated grid (common in the demo).
    if let Ok(guard) = cache.lock() {
        if let Some(out) = guard.grids.get(&grid_key) {
            let c = guard.table_c;
            return Tensor::<1>::from_floats(out.as_slice(), device).reshape([gt * gh * gw, c]);
        }
    }

    // Ensure the square embedding table is on the CPU (one GPU readback per encoder).
    let need_table = cache
        .lock()
        .map(|g| g.table.is_none())
        .unwrap_or(true);
    if need_table {
        let ids: Vec<i32> = (0..num_pos as i32).collect();
        let square = pos_embed.forward(Tensor::<2, Int>::from_ints(
            TensorData::new(ids, [1, num_pos]),
            device,
        )); // [1, num_pos, C]
        let [_b, _n, c] = square.dims();
        let table: Vec<f32> = square
            .reshape([side * side, c])
            .into_data_async()
            .await
            .ok()
            .and_then(|d| d.try_to_vec().ok())
            .unwrap_or_else(|| vec![0.0; side * side * c]);
        if let Ok(mut guard) = cache.lock() {
            guard.table = Some(table);
            guard.table_c = c;
            guard.table_side = side;
        }
    }

    let out = {
        let mut guard = cache.lock().expect("pos embed cache");
        let c = guard.table_c.max(1);
        let side = guard.table_side.max(1);
        let table = guard.table.as_ref().map(Vec::as_slice).unwrap_or(&[]);
        let mut out = Vec::with_capacity(gt * gh * gw * c);
        for _t in 0..gt {
            for hb in 0..(gh / merge) {
                for wb in 0..(gw / merge) {
                    for hi in 0..merge {
                        for wi in 0..merge {
                            let y = hb * merge + hi;
                            let x = wb * merge + wi;
                            // HF align_corners=True bilinear into side×side table.
                            let (ys, yw) = bilinear_taps(y, gh, side);
                            let (xs, xw) = bilinear_taps(x, gw, side);
                            for ch in 0..c {
                                let mut acc = 0.0f32;
                                for (ty, wy) in ys.iter().zip(yw.iter()) {
                                    for (tx, wx) in xs.iter().zip(xw.iter()) {
                                        let idx = (*ty * side + *tx) * c + ch;
                                        acc += table.get(idx).copied().unwrap_or(0.0) * wy * wx;
                                    }
                                }
                                out.push(acc);
                            }
                        }
                    }
                }
            }
        }
        guard.grids.insert(grid_key, out.clone());
        (out, c)
    };
    let (out, c) = out;
    Tensor::<1>::from_floats(out.as_slice(), device).reshape([gt * gh * gw, c])
}

/// `align_corners=True` bilinear taps for one axis.
fn bilinear_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let src = if size <= 1 {
        0.0
    } else {
        index as f64 * (side - 1) as f64 / (size - 1) as f64
    };
    let floor = src.floor();
    let t0 = floor.clamp(0.0, (side - 1) as f64) as usize;
    let t1 = (floor + 1.0).clamp(0.0, (side - 1) as f64) as usize;
    let w1 = (src - floor).clamp(0.0, 1.0) as f32;
    let w0 = 1.0 - w1;
    ([t0, t1], [w0, w1])
}

fn vision_rope_2d(
    gh: usize,
    gw: usize,
    gt: usize,
    spatial_merge: usize,
    head_dim: usize,
    device: &Device,
) -> (Tensor<2>, Tensor<2>) {
    let theta = 10_000.0f64;
    let dim = head_dim / 2;
    let num_freqs = dim / 2;
    let inv_freq: Vec<f64> = (0..num_freqs)
        .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / dim as f64))
        .collect();

    let seq = gt * gh * gw;
    let mut cos_data = Vec::with_capacity(seq * head_dim);
    let mut sin_data = Vec::with_capacity(seq * head_dim);
    let merge = spatial_merge.max(1);

    // Emit (h, w) coords in spatial-merge block order to match patchify.
    for _t in 0..gt {
        for hb in 0..(gh / merge) {
            for wb in 0..(gw / merge) {
                for hi in 0..merge {
                    for wi in 0..merge {
                        let y = hb * merge + hi;
                        let x = wb * merge + wi;
                        for i in 0..num_freqs {
                            let angle_h = y as f64 * inv_freq[i];
                            cos_data.push(angle_h.cos() as f32);
                            sin_data.push(angle_h.sin() as f32);
                        }
                        for i in 0..num_freqs {
                            let angle_w = x as f64 * inv_freq[i];
                            cos_data.push(angle_w.cos() as f32);
                            sin_data.push(angle_w.sin() as f32);
                        }
                        let start = cos_data.len() - dim;
                        cos_data.extend_from_within(start..);
                        sin_data.extend_from_within(start..);
                    }
                }
            }
        }
    }

    let cos = Tensor::<1>::from_floats(cos_data.as_slice(), device).reshape([seq, head_dim]);
    let sin = Tensor::<1>::from_floats(sin_data.as_slice(), device).reshape([seq, head_dim]);
    (cos, sin)
}
