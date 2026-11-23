use burn::module::Param;
use burn::prelude::*;

/// RMSNorm with HF-compatible `weight` parameter name.
#[derive(Module, Debug)]
pub struct RmsNorm {
    pub weight: Param<Tensor<1>>,
    #[module(skip)]
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([size], device)),
            eps,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        // Pre-scale to reduce f16 overflow risk on large activations.
        let s = x.clone().abs().max_dim(2).clamp_min(1.0);
        let x_scaled = x / s;
        let variance = x_scaled.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x_scaled * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze::<3>()
    }

    pub fn forward_2d(&self, x: Tensor<2>) -> Tensor<2> {
        let s = x.clone().abs().max_dim(1).clamp_min(1.0);
        let x_scaled = x / s;
        let variance = x_scaled.clone().powf_scalar(2.0).mean_dim(1);
        let normed = x_scaled * (variance + self.eps).sqrt().recip();
        normed * self.weight.val().unsqueeze_dim::<2>(0)
    }

    pub fn set_weight(&mut self, weight: Tensor<1>) {
        self.weight = Param::from_tensor(weight);
    }
}
