use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

/// SwiGLU feed-forward with fused gate+up projection.
#[derive(Module, Debug)]
pub struct FeedForward {
    gate_up_proj: Linear,
    down_proj: Linear,
    #[module(skip)]
    intermediate_size: usize,
}

impl FeedForward {
    pub fn new(d_model: usize, intermediate_size: usize, device: &Device) -> Self {
        Self {
            gate_up_proj: LinearConfig::new(d_model, 2 * intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, d_model)
                .with_bias(false)
                .init(device),
            intermediate_size,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate_up = self.gate_up_proj.forward(x);
        let [b, s, _] = gate_up.dims();
        let i = self.intermediate_size;
        let gate = activation::silu(gate_up.clone().slice(s![0..b, 0..s, 0..i]));
        let up = gate_up.slice(s![0..b, 0..s, i..2 * i]);
        self.down_proj.forward(gate * up)
    }

    pub fn load_weights(&mut self, gate_up: Tensor<2>, down: Tensor<2>) {
        self.gate_up_proj.weight = Param::from_tensor(gate_up);
        self.down_proj.weight = Param::from_tensor(down);
    }
}
