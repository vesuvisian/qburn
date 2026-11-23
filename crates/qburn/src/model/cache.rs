use burn::prelude::*;

/// Key-value cache for autoregressive generation.
///
/// Shape: `[batch, num_heads, seq_len, head_dim]`.
pub struct KvCache {
    cache: Tensor<4>,
    max_seq_len: usize,
    cur_seq_len: usize,
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
            cache: Tensor::zeros([batch_size, num_heads, max_seq_len, head_dim], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cache = Tensor::zeros(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    pub fn forward(&mut self, tensor: Tensor<4>) -> Tensor<4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();

        if seq_len >= self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            let truncated =
                tensor.slice(s![0..batch_size, 0..num_heads, start..seq_len, 0..head_dim]);
            self.cache = self.cache.clone().slice_assign(
                s![0..batch_size, 0..num_heads, 0..self.max_seq_len, 0..head_dim],
                truncated,
            );
            self.cur_seq_len = self.max_seq_len;
            return self
                .cache
                .clone()
                .slice(s![0..batch_size, 0..num_heads, 0..self.max_seq_len, 0..head_dim]);
        }

        let mut new_seq_len = self.cur_seq_len + seq_len;
        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev = self.cache.clone().slice(s![
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..head_dim
            ]);
            self.cache = self.cache.clone().slice_assign(
                s![0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..head_dim],
                prev,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            s![
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..head_dim
            ],
            tensor,
        );
        self.cur_seq_len = new_seq_len;
        self.cache
            .clone()
            .slice(s![0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..head_dim])
    }

    pub fn len(&self) -> usize {
        self.cur_seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_seq_len == 0
    }
}
