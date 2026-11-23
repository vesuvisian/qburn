use burn::prelude::*;

/// Multi-resolution RoPE for Qwen3-VL (T/H/W interleaved sections).
pub struct MRopeEmbedding {
    inv_freq: Vec<f64>,
    dim_assignment: Vec<usize>,
    pub mrope_section: [usize; 3],
    pub head_dim: usize,
}

impl MRopeEmbedding {
    pub fn new(
        mrope_section: [usize; 3],
        rope_theta: f64,
        head_dim: usize,
        _device: &Device,
    ) -> Self {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(i as f64 * 2.0 / head_dim as f64))
            .collect();
        // `mrope_section` already counts frequencies over the half-dim band
        // (e.g. [24,20,20] → 64 = head_dim/2). Do not halve again — that
        // scrambles the interleaved T/H/W layout HF expects.
        debug_assert_eq!(
            mrope_section[0] + mrope_section[1] + mrope_section[2],
            half_dim,
            "mrope_section must sum to head_dim/2"
        );
        let dim_assignment = build_dim_assignment(&mrope_section, half_dim);
        Self {
            inv_freq,
            dim_assignment,
            mrope_section,
            head_dim,
        }
    }

    /// `position_ids`: 3 rows (T, H, W), each length `seq_len`.
    pub fn compute_cos_sin(
        &self,
        position_ids: &[Vec<usize>],
        device: &Device,
    ) -> (Tensor<2>, Tensor<2>) {
        assert_eq!(position_ids.len(), 3);
        let seq_len = position_ids[0].len();
        let half_dim = self.head_dim / 2;
        let mut cos_data = Vec::with_capacity(seq_len * self.head_dim);
        let mut sin_data = Vec::with_capacity(seq_len * self.head_dim);

        for pos_idx in 0..seq_len {
            for freq_idx in 0..half_dim {
                let component = self.dim_assignment[freq_idx];
                let position = position_ids[component][pos_idx] as f64;
                let angle = position * self.inv_freq[freq_idx];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            let start = cos_data.len() - half_dim;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }

        let cos =
            Tensor::<1>::from_floats(cos_data.as_slice(), device).reshape([seq_len, self.head_dim]);
        let sin =
            Tensor::<1>::from_floats(sin_data.as_slice(), device).reshape([seq_len, self.head_dim]);
        (cos, sin)
    }

    pub fn precompute_text_tables(
        &self,
        max_positions: usize,
        device: &Device,
    ) -> (Tensor<2>, Tensor<2>) {
        let half_dim = self.head_dim / 2;
        let mut cos_data = Vec::with_capacity(max_positions * self.head_dim);
        let mut sin_data = Vec::with_capacity(max_positions * self.head_dim);
        for pos in 0..max_positions {
            for freq_idx in 0..half_dim {
                let angle = pos as f64 * self.inv_freq[freq_idx];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            let start = cos_data.len() - half_dim;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }
        let cos = Tensor::<1>::from_floats(cos_data.as_slice(), device)
            .reshape([max_positions, self.head_dim]);
        let sin = Tensor::<1>::from_floats(sin_data.as_slice(), device)
            .reshape([max_positions, self.head_dim]);
        (cos, sin)
    }
}

/// Build interleaved T/H/W frequency assignment matching Qwen3-VL.
fn build_dim_assignment(mrope_section: &[usize; 3], half_dim: usize) -> Vec<usize> {
    let mut assignment = vec![0usize; half_dim];
    let mut counts = [0usize; 3];
    for freq_idx in 0..half_dim {
        let mut component = freq_idx % 3;
        let mut tried = 0;
        while counts[component] >= mrope_section[component] && tried < 3 {
            component = (component + 1) % 3;
            tried += 1;
        }
        assignment[freq_idx] = component;
        counts[component] += 1;
    }
    assignment
}

/// Build 3D position ids for a multimodal sequence.
///
/// For text-only tokens, T=H=W=sequential index.
/// For vision tokens spanning `grid_thw` after merge, fill spatial grids then continue text.
pub fn build_position_ids(
    input_ids: &[u32],
    image_token_id: u32,
    grid_thw: Option<(usize, usize, usize)>,
    spatial_merge: usize,
) -> Vec<Vec<usize>> {
    let seq_len = input_ids.len();
    let mut t = vec![0usize; seq_len];
    let mut h = vec![0usize; seq_len];
    let mut w = vec![0usize; seq_len];

    let mut pos = 0usize;
    let mut i = 0usize;
    while i < seq_len {
        if input_ids[i] == image_token_id {
            if let Some((gt, gh, gw)) = grid_thw {
                let mh = gh / spatial_merge;
                let mw = gw / spatial_merge;
                let mut vision_pos = 0usize;
                let start = i;
                while i < seq_len && input_ids[i] == image_token_id {
                    let local = vision_pos;
                    let ti = local / (mh * mw);
                    let rem = local % (mh * mw);
                    let hi = rem / mw;
                    let wi = rem % mw;
                    t[i] = pos + ti;
                    h[i] = pos + hi;
                    w[i] = pos + wi;
                    vision_pos += 1;
                    i += 1;
                }
                let max_vision = [*t[start..i].iter().max().unwrap_or(&pos),
                    *h[start..i].iter().max().unwrap_or(&pos),
                    *w[start..i].iter().max().unwrap_or(&pos)]
                    .into_iter()
                    .max()
                    .unwrap_or(pos);
                pos = max_vision + 1;
                let _ = (gt, mh, mw); // grid_t used via ti
            } else {
                t[i] = pos;
                h[i] = pos;
                w[i] = pos;
                pos += 1;
                i += 1;
            }
        } else {
            t[i] = pos;
            h[i] = pos;
            w[i] = pos;
            pos += 1;
            i += 1;
        }
    }
    vec![t, h, w]
}
