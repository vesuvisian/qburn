use burn::prelude::*;

/// Standard 1D rotary position embedding tables.
pub struct RotaryEmbedding {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_data = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_data = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(i as f64 * 2.0 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            let start = cos_data.len() - half_dim;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }

        let cos = Tensor::<1>::from_floats(cos_data.as_slice(), device).reshape([max_seq_len, head_dim]);
        let sin = Tensor::<1>::from_floats(sin_data.as_slice(), device).reshape([max_seq_len, head_dim]);
        Self { cos, sin }
    }

    pub fn apply(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        start_pos: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let [_batch, _heads, seq_len, _dim] = q.dims();
        let cos = self
            .cos
            .clone()
            .slice(s![start_pos..start_pos + seq_len, ..])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);
        let sin = self
            .sin
            .clone()
            .slice(s![start_pos..start_pos + seq_len, ..])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);
        let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
        let k_embed = k.clone() * cos + rotate_half(k) * sin;
        (q_embed, k_embed)
    }
}

/// Apply rotary with explicit cos/sin `[seq, head_dim]`.
pub fn apply_rotary(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: &Tensor<2>,
    sin: &Tensor<2>,
) -> (Tensor<4>, Tensor<4>) {
    let cos = cos.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let sin = sin.clone().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
    let q_embed = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
    let k_embed = k.clone() * cos + rotate_half(k) * sin;
    (q_embed, k_embed)
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let [batch, heads, seq_len, dim] = x.dims();
    let half = dim / 2;
    let x1 = x.clone().slice(s![0..batch, 0..heads, 0..seq_len, 0..half]);
    let x2 = x.slice(s![0..batch, 0..heads, 0..seq_len, half..dim]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}
