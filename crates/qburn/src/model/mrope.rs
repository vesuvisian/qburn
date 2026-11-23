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
/// For each contiguous `<|image_pad|>` or `<|video_pad|>` run, consume the next
/// `(t,h,w)` grid (after spatial merge), fill T/H/W (temporal index rises when
/// `grid_t > 1`), then continue text from `max+1`.
/// Matches Qwen3-VL multi-image / video indexing (`get_rope_index_3`).
pub fn build_position_ids(
    input_ids: &[u32],
    image_token_id: u32,
    video_token_id: u32,
    grids_thw: &[(usize, usize, usize)],
    spatial_merge: usize,
) -> Vec<Vec<usize>> {
    let seq_len = input_ids.len();
    let mut t = vec![0usize; seq_len];
    let mut h = vec![0usize; seq_len];
    let mut w = vec![0usize; seq_len];

    let mut pos = 0usize;
    let mut i = 0usize;
    let mut grid_idx = 0usize;
    while i < seq_len {
        let id = input_ids[i];
        if id == image_token_id || id == video_token_id {
            let pad_id = id;
            assert!(
                grid_idx < grids_thw.len(),
                "vision pad run without a matching grid_thw (grid_idx={grid_idx})"
            );
            let (_gt, gh, gw) = grids_thw[grid_idx];
            grid_idx += 1;
            let mh = gh / spatial_merge;
            let mw = gw / spatial_merge;
            let mut vision_pos = 0usize;
            let start = i;
            while i < seq_len && input_ids[i] == pad_id {
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
            let max_vision = [
                *t[start..i].iter().max().unwrap_or(&pos),
                *h[start..i].iter().max().unwrap_or(&pos),
                *w[start..i].iter().max().unwrap_or(&pos),
            ]
            .into_iter()
            .max()
            .unwrap_or(pos);
            pos = max_vision + 1;
        } else {
            t[i] = pos;
            h[i] = pos;
            w[i] = pos;
            pos += 1;
            i += 1;
        }
    }
    assert_eq!(
        grid_idx,
        grids_thw.len(),
        "unused grid_thw entries (used {grid_idx} of {})",
        grids_thw.len()
    );
    vec![t, h, w]
}

#[cfg(test)]
mod tests {
    use super::build_position_ids;

    const IMG: u32 = 151_655;
    const VID: u32 = 151_656;

    #[test]
    fn two_grids_continue_after_each_vision_max() {
        // text(1) + img pads(4) + text(1) + img pads(4) + text(1)
        // grids: (1,4,4) merge=2 → mh=mw=2 → 4 tokens each
        let ids = vec![1, IMG, IMG, IMG, IMG, 2, IMG, IMG, IMG, IMG, 3];
        let grids = [(1usize, 4, 4), (1, 4, 4)];
        let pos = build_position_ids(&ids, IMG, VID, &grids, 2);
        let t = &pos[0];
        let h = &pos[1];
        let w = &pos[2];

        // First text token at 0.
        assert_eq!((t[0], h[0], w[0]), (0, 0, 0));
        // First vision block: H/W in {0,1} offset by pos=1 → {1,2}; max = 2.
        assert_eq!((h[1], w[1]), (1, 1)); // hi=0,wi=0
        assert_eq!((h[2], w[2]), (1, 2)); // hi=0,wi=1
        assert_eq!((h[3], w[3]), (2, 1)); // hi=1,wi=0
        assert_eq!((h[4], w[4]), (2, 2)); // hi=1,wi=1
        // Text between at max_vision+1 = 3.
        assert_eq!((t[5], h[5], w[5]), (3, 3, 3));
        // Second vision offset by 4; max = 5.
        assert_eq!((h[6], w[6]), (4, 4));
        assert_eq!((h[9], w[9]), (5, 5));
        // Final text at 6.
        assert_eq!((t[10], h[10], w[10]), (6, 6, 6));
    }

    #[test]
    fn adjacent_vision_blocks() {
        // Multi chat template puts vision_end + vision_start between pad runs
        // (non-image tokens), so the pad spans are separate.
        let sep = 151_653u32;
        let ids = vec![IMG, IMG, IMG, IMG, sep, sep, IMG, IMG, IMG, IMG];
        let grids = [(1usize, 4, 4), (1, 4, 4)];
        let pos = build_position_ids(&ids, IMG, VID, &grids, 2);
        // First block max = 1 (pos starts 0, H/W in 0..1).
        assert_eq!(pos[1][0], 0);
        assert_eq!(pos[1][3], 1);
        // Separators continue from 2, 3.
        assert_eq!((pos[0][4], pos[0][5]), (2, 3));
        // Second block starts at 4.
        assert_eq!(pos[1][6], 4);
        assert_eq!(pos[1][9], 5);
    }

    #[test]
    fn video_grid_t_raises_temporal_axis() {
        // Legacy contiguous grid_t>1 span (Qwen2-VL style). Qwen3-VL callers
        // expand to repeated (1,h,w) grids instead — see qwen3_video_split_spans.
        let mut ids = vec![1u32];
        ids.extend(std::iter::repeat(VID).take(12));
        ids.push(2u32);
        let grids = [(3usize, 4, 4)];
        let pos = build_position_ids(&ids, IMG, VID, &grids, 2);
        let t = &pos[0];
        assert_eq!(t[0], 0);
        assert_eq!(t[1], 1); // ti=0 + pos
        assert_eq!(t[5], 2); // ti=1
        assert_eq!(t[9], 3); // ti=2
        assert_eq!(t[13], 4);
    }

    #[test]
    fn qwen3_video_split_spans_use_t_equals_one() {
        // Qwen3-VL: timestamp text + vision_start/end between temporal groups;
        // each group is grid (1,4,4) → 4 video pads.
        let sep = 151_653u32; // stand-in for vision_end / timestamp / vision_start
        let mut ids = vec![1u32];
        ids.extend(std::iter::repeat(VID).take(4));
        ids.push(sep);
        ids.extend(std::iter::repeat(VID).take(4));
        ids.push(2u32);
        let grids = [(1usize, 4, 4), (1, 4, 4)];
        let pos = build_position_ids(&ids, IMG, VID, &grids, 2);
        let t = &pos[0];
        // Text at 0; first vision T stays flat (ti=0).
        assert_eq!(t[0], 0);
        assert_eq!(t[1], 1);
        assert_eq!(t[4], 1);
        // Separator continues after max H/W (=2) → 3.
        assert_eq!(t[5], 3);
        // Second vision offset by 4; T still flat within the span.
        assert_eq!(t[6], 4);
        assert_eq!(t[9], 4);
        assert_eq!(t[10], 6);
    }
}
