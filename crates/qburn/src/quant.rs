//! Offline weight quantization helpers for burnpack export.

use burn::module::{Module, ParamGroup, Quantizer};
use burn::tensor::quantization::{Calibration, QuantScheme, QuantValue, ScaleDtype};

use crate::vl::Qwen3Vl;

/// Supported export quantization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuant {
    /// Full precision (as loaded into the module, typically F32).
    Fp32,
    /// Store floats as F16 in the burnpack (~2× smaller).
    Fp16,
    /// Burn Q4S per-block symmetric quantization on every float weight (~4–8× smaller).
    Q4,
    /// Asymmetric mix: Q4S on bulky Linear stacks only; embeddings, norms, and
    /// vision I/O (patch/pos/merger/DeepStack) stay float and are saved as F16.
    Q4Mix,
}

impl WeightQuant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Q4 => "q4",
            Self::Q4Mix => "q4-mix",
        }
    }

    pub fn pack_name(self) -> &'static str {
        match self {
            Self::Fp32 => "model-fp32.bpk",
            Self::Fp16 => "model-fp16.bpk",
            Self::Q4 => "model-q4.bpk",
            Self::Q4Mix => "model-q4-mix.bpk",
        }
    }
}

fn q4s_scheme() -> QuantScheme {
    QuantScheme::default()
        .per_block([32], ScaleDtype::F16)
        .with_value(QuantValue::Q4S)
}

/// Linear stacks to Q4S under [`WeightQuant::Q4Mix`].
///
/// Predicates match leaf path suffixes so RMSNorm / LayerNorm / biases stay float.
/// Merger and patch_embed linears are intentionally omitted (stay F16).
fn q4_mix_linear_group() -> ParamGroup {
    ParamGroup::from_any_predicates(vec![
        "qkv_proj.weight",
        "o_proj.weight",
        "gate_up_proj.weight",
        "down_proj.weight",
        "lm_head.weight",
        // Vision block linears (Burn field names under vision.blocks.*).
        "attn.qkv.weight",
        "attn.proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
    ])
}

/// Apply in-memory weight quantization before saving.
///
/// No-op for Fp16/Fp32 (those are handled by burnpack adapters at save time).
pub fn apply_weight_quant(model: Qwen3Vl, mode: WeightQuant) -> Qwen3Vl {
    match mode {
        WeightQuant::Fp32 | WeightQuant::Fp16 => model,
        WeightQuant::Q4 => {
            let mut quantizer = Quantizer::new(Calibration::MinMax, q4s_scheme());
            model.quantize_weights(&mut quantizer)
        }
        WeightQuant::Q4Mix => {
            let mut quantizer = Quantizer::new(Calibration::MinMax, q4s_scheme());
            model.quantize_weights_group(&mut quantizer, q4_mix_linear_group())
        }
    }
}
