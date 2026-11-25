use burn::prelude::*;

/// Key-value cache for autoregressive generation.
///
/// Shape: `[batch, num_heads, seq_len, head_dim]`.
///
/// Prefill keeps the exact K/V tensor (no padded zero buffer). The first decode
/// append promotes it into a `max_seq`-capacity buffer once; later tokens
/// `slice_assign` in place via `take()` (no full-buffer clone before write).
pub struct KvCache {
    /// Exact-size prefix after prefill, before the capacity buffer exists.
    dense: Option<Tensor<4>>,
    /// Padded capacity buffer used for decode appends.
    buffer: Option<Tensor<4>>,
    max_seq_len: usize,
    cur_seq_len: usize,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
}

impl KvCache {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            dense: None,
            buffer: None,
            max_seq_len,
            cur_seq_len: 0,
            batch_size,
            num_heads,
            head_dim,
            device: device.clone(),
        }
    }

    pub fn reset(&mut self) {
        self.dense = None;
        self.buffer = None;
        self.cur_seq_len = 0;
    }

    fn prefix(&self, buf: &Tensor<4>) -> Tensor<4> {
        buf.clone().slice(s![
            0..self.batch_size,
            0..self.num_heads,
            0..self.cur_seq_len,
            0..self.head_dim
        ])
    }

    /// Ensure `buffer` holds a `max_seq`-capacity tensor with the current
    /// prefix in `[0..cur_seq_len]`. Consumes `dense` if present.
    fn promote_to_buffer(&mut self) {
        if self.buffer.is_some() {
            return;
        }
        let mut buf = Tensor::zeros(
            [
                self.batch_size,
                self.num_heads,
                self.max_seq_len,
                self.head_dim,
            ],
            &self.device,
        );
        if let Some(dense) = self.dense.take() {
            let dense_len = dense.dims()[2];
            let len = dense_len.min(self.max_seq_len);
            let dense = if dense_len > self.max_seq_len {
                let start = dense_len - self.max_seq_len;
                dense.slice(s![
                    0..self.batch_size,
                    0..self.num_heads,
                    start..dense_len,
                    0..self.head_dim
                ])
            } else {
                dense
            };
            buf = buf.slice_assign(
                s![
                    0..self.batch_size,
                    0..self.num_heads,
                    0..len,
                    0..self.head_dim
                ],
                dense,
            );
            self.cur_seq_len = len;
        }
        self.buffer = Some(buf);
    }

    pub fn forward(&mut self, tensor: Tensor<4>) -> Tensor<4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();
        debug_assert_eq!(batch_size, self.batch_size);
        debug_assert_eq!(num_heads, self.num_heads);
        debug_assert_eq!(head_dim, self.head_dim);

        // Prefill / first write: keep the exact tensor; defer capacity buffer.
        if self.cur_seq_len == 0 && self.buffer.is_none() {
            if seq_len >= self.max_seq_len {
                let start = seq_len - self.max_seq_len;
                let truncated = tensor.slice(s![
                    0..batch_size,
                    0..num_heads,
                    start..seq_len,
                    0..head_dim
                ]);
                let out = truncated.clone();
                self.dense = Some(truncated);
                self.cur_seq_len = self.max_seq_len;
                return out;
            }
            let out = tensor.clone();
            self.dense = Some(tensor);
            self.cur_seq_len = seq_len;
            return out;
        }

        self.promote_to_buffer();

        if seq_len >= self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            let truncated =
                tensor.slice(s![0..batch_size, 0..num_heads, start..seq_len, 0..head_dim]);
            let mut buf = Tensor::zeros(
                [
                    self.batch_size,
                    self.num_heads,
                    self.max_seq_len,
                    self.head_dim,
                ],
                &self.device,
            );
            buf = buf.slice_assign(
                s![0..batch_size, 0..num_heads, 0..self.max_seq_len, 0..head_dim],
                truncated,
            );
            self.cur_seq_len = self.max_seq_len;
            let out = self.prefix(&buf);
            self.buffer = Some(buf);
            return out;
        }

        let mut new_seq_len = self.cur_seq_len + seq_len;
        let mut buf = self.buffer.take().expect("promoted buffer");

        if new_seq_len > self.max_seq_len {
            // Slide window: drop oldest tokens, then append.
            let keep = self.max_seq_len - seq_len;
            let prev = buf.slice(s![
                0..batch_size,
                0..num_heads,
                (self.cur_seq_len - keep)..self.cur_seq_len,
                0..head_dim
            ]);
            buf = Tensor::zeros(
                [
                    self.batch_size,
                    self.num_heads,
                    self.max_seq_len,
                    self.head_dim,
                ],
                &self.device,
            )
            .slice_assign(
                s![0..batch_size, 0..num_heads, 0..keep, 0..head_dim],
                prev,
            );
            self.cur_seq_len = keep;
            new_seq_len = self.max_seq_len;
        }

        buf = buf.slice_assign(
            s![
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..head_dim
            ],
            tensor,
        );
        self.cur_seq_len = new_seq_len;
        let out = self.prefix(&buf);
        self.buffer = Some(buf);
        out
    }

    pub fn len(&self) -> usize {
        self.cur_seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_seq_len == 0
    }
}
