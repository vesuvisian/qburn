//! Weight loading from SafeTensors / Burnpack.
//!
//! - [`load_from_hf_dir`] — desktop parity path (HF key layout → Burn modules).
//! - [`load_from_burnpack`] / [`load_from_burnpack_bytes`] — converted packs for CLI/wasm.
//! - [`save_burnpack`] / [`shard_file`] — used by `qburn-converter`.
//!
//! Vision `patch_embed.proj.weight` is HF Conv3d `[out,in,t,h,w]`; see
//! [`crate::model::VisionEncoder::load_hf_tensors`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use burn::tensor::{AllocationProperty, Bytes, TensorData};
use burn::tensor::DType;
use burn_store::{BurnpackStore, FloatCastAdapter, ModuleSnapshot};
use half::f16;

use crate::config::Qwen3VlConfig;
use crate::quant::WeightQuant;
use crate::vl::Qwen3Vl;

/// Errors while loading weights.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{0}")]
    Msg(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Load a Qwen3-VL model from a HuggingFace checkpoint directory
/// containing `config.json` and `*.safetensors` shards.
pub fn load_from_hf_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Qwen3Vl, LoadError> {
    let dir = dir.as_ref();
    let cfg = Qwen3VlConfig::from_file(dir.join("config.json"))
        .map_err(|e| LoadError::Msg(e.to_string()))?;

    let mut model = Qwen3Vl::new(cfg.clone(), device);
    let tensors = load_safetensor_map(dir)?;
    apply_hf_tensors(&mut model, &tensors, device)?;
    if cfg.text_config.tie_word_embeddings {
        model.decoder.tie_embeddings();
    }
    Ok(model)
}

/// Load from a converter out-dir (`manifest.json` + `.bpk.part` shards) or a single `.bpk`.
pub fn load_from_burnpack_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Qwen3Vl, LoadError> {
    let dir = dir.as_ref();
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.is_file() {
        // Single pack named by quant convention, or any `*.bpk`.
        for name in [
            "model-q4-mix.bpk",
            "model-q4.bpk",
            "model-fp16.bpk",
            "model-fp32.bpk",
        ] {
            let p = dir.join(name);
            if p.is_file() {
                let quant = match name {
                    "model-q4-mix.bpk" => WeightQuant::Q4Mix,
                    "model-q4.bpk" => WeightQuant::Q4,
                    "model-fp16.bpk" => WeightQuant::Fp16,
                    _ => WeightQuant::Fp32,
                };
                return load_from_burnpack(&p, device, quant);
            }
        }
        return Err(LoadError::Msg(format!(
            "no manifest.json or model-*.bpk under {}",
            dir.display()
        )));
    }

    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| LoadError::Msg(format!("manifest json: {e}")))?;
    let quant = match manifest.get("quant").and_then(|v| v.as_str()).unwrap_or("fp16") {
        "q4-mix" => WeightQuant::Q4Mix,
        "q4" => WeightQuant::Q4,
        "fp32" => WeightQuant::Fp32,
        _ => WeightQuant::Fp16,
    };
    let shard_paths: Vec<PathBuf> = manifest
        .get("shards")
        .and_then(|s| s.as_array())
        .ok_or_else(|| LoadError::Msg("manifest missing shards[]".into()))?
        .iter()
        .filter_map(|s| s.get("file").and_then(|f| f.as_str()).map(|f| dir.join(f)))
        .collect();
    if shard_paths.is_empty() {
        return Err(LoadError::Msg("manifest shards[] is empty".into()));
    }
    for p in &shard_paths {
        if !p.is_file() {
            return Err(LoadError::Msg(format!("missing shard {}", p.display())));
        }
    }
    let bytes = concat_shard_files(&shard_paths)?;
    let cfg = if dir.join("config.json").is_file() {
        Qwen3VlConfig::from_file(dir.join("config.json")).map_err(|e| LoadError::Msg(e.to_string()))?
    } else if let Some(c) = manifest.get("config") {
        serde_json::from_value(c.clone()).unwrap_or_else(|_| Qwen3VlConfig::qwen3_vl_2b())
    } else {
        Qwen3VlConfig::qwen3_vl_2b()
    };
    load_from_burnpack_bytes(bytes_from_vec(bytes), cfg, device, quant)
}

/// Auto-detect HF safetensors dir vs converter burnpack out-dir.
pub fn load_model_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Qwen3Vl, LoadError> {
    let dir = dir.as_ref();
    if dir.join("manifest.json").is_file()
        || dir.join("model-q4-mix.bpk").is_file()
        || dir.join("model-q4.bpk").is_file()
        || dir.join("model-fp16.bpk").is_file()
        || dir.join("model-fp32.bpk").is_file()
    {
        return load_from_burnpack_dir(dir, device);
    }
    load_from_hf_dir(dir, device)
}

/// Load from a burnpack file produced by the converter.
pub fn load_from_burnpack(
    path: impl AsRef<Path>,
    device: &Device,
    quant: WeightQuant,
) -> Result<Qwen3Vl, LoadError> {
    let path = path.as_ref();
    let cfg_path = path
        .parent()
        .map(|p| p.join("config.json"))
        .unwrap_or_else(|| path.with_extension("json"));
    let cfg = if cfg_path.exists() {
        Qwen3VlConfig::from_file(&cfg_path).map_err(|e| LoadError::Msg(e.to_string()))?
    } else {
        Qwen3VlConfig::qwen3_vl_2b()
    };
    let bytes = std::fs::read(path)?;
    load_from_burnpack_bytes(bytes_from_vec(bytes), cfg, device, quant)
}

/// Load burnpack from bytes (wasm / sharded-concat path).
///
/// For [`WeightQuant::Fp16`] / [`WeightQuant::Q4Mix`] packs, float tensors are cast
/// F16→F32 on load (QFloat weights pass through). Uniform [`WeightQuant::Q4`] packs
/// load quantized weights as-is.
///
/// For `Q4Mix`, when `tie_word_embeddings` is set, `lm_head` is rebuilt from the
/// float embedding so logits match the unquantized token table.
pub fn load_from_burnpack_bytes(
    bytes: Bytes,
    cfg: Qwen3VlConfig,
    device: &Device,
    quant: WeightQuant,
) -> Result<Qwen3Vl, LoadError> {
    let tie = cfg.text_config.tie_word_embeddings;
    let mut model = Qwen3Vl::new(cfg, device);
    let mut store = BurnpackStore::from_bytes(Some(bytes));
    match quant {
        WeightQuant::Fp16 | WeightQuant::Q4Mix => {
            // F16 on disk → F32 in every float param (covers custom RmsNorm / LayerNormSimple).
            // QFloat (Q4S) tensors are left alone by FloatCastAdapter.
            store = store.with_from_adapter(FloatCastAdapter::to(DType::F32));
        }
        WeightQuant::Fp32 | WeightQuant::Q4 => {}
    }
    model
        .load_from(&mut store)
        .map_err(|e| LoadError::Msg(format!("burnpack bytes load: {e:?}")))?;
    if matches!(quant, WeightQuant::Q4Mix) && tie {
        model.decoder.tie_embeddings();
    }
    Ok(model)
}

/// Wrap a `Vec<u8>` as Burn `Bytes` for wasm / in-memory loads.
pub fn bytes_from_vec(data: Vec<u8>) -> Bytes {
    Bytes::from_shared(::bytes::Bytes::from(data), AllocationProperty::Native)
}

fn load_safetensor_map(dir: &Path) -> Result<HashMap<String, TensorData>, LoadError> {
    let mut map = HashMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(LoadError::Msg(format!(
            "no .safetensors files in {}",
            dir.display()
        )));
    }

    for file in files {
        // Use SafetensorsStore to read; collect via temporary empty module is awkward,
        // so parse with the safetensors crate via burn_store bridge.
        let bytes = std::fs::read(&file)?;
        let st = safetensors_view(&bytes)?;
        for (name, data) in st {
            map.insert(name, data);
        }
    }
    Ok(map)
}

fn safetensors_view(bytes: &[u8]) -> Result<Vec<(String, TensorData)>, LoadError> {
    use safetensors::SafeTensors;
    let st = SafeTensors::deserialize(bytes).map_err(|e| LoadError::Msg(e.to_string()))?;
    let mut out = Vec::new();
    for name in st.names() {
        let t = st
            .tensor(name)
            .map_err(|e| LoadError::Msg(e.to_string()))?;
        let shape: Vec<usize> = t.shape().to_vec();
        let data = match t.dtype() {
            safetensors::Dtype::F32 => {
                let v: Vec<f32> = bytemuck_cast_slice(t.data());
                TensorData::new(v, shape)
            }
            safetensors::Dtype::F16 => {
                let v: Vec<f16> = bytemuck_cast_slice_f16(t.data());
                let f32s: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
                TensorData::new(f32s, shape)
            }
            safetensors::Dtype::BF16 => {
                // Interpret as u16 bit patterns → f32 via half::bf16 if available; fallback cast.
                let raw = t.data();
                let mut f32s = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // bfloat16 → f32: shift left 16
                    f32s.push(f32::from_bits((bits as u32) << 16));
                }
                TensorData::new(f32s, shape)
            }
            other => {
                return Err(LoadError::Msg(format!(
                    "unsupported safetensors dtype {other:?} for {name}"
                )));
            }
        };
        out.push((name.to_string(), data));
    }
    Ok(out)
}

fn bytemuck_cast_slice(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytemuck_cast_slice_f16(data: &[u8]) -> Vec<f16> {
    data.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn tensor2(data: &TensorData, device: &Device) -> Tensor<2> {
    assert_eq!(data.rank(), 2, "expected rank-2 tensor, got {:?}", data.shape());
    Tensor::<2>::from_data(data.clone(), device)
}

fn tensor1(data: &TensorData, device: &Device) -> Tensor<1> {
    assert_eq!(data.rank(), 1, "expected rank-1 tensor, got {:?}", data.shape());
    Tensor::<1>::from_data(data.clone(), device)
}

/// Apply HuggingFace key layout onto the Burn modules.
///
/// HF Linear weights are `[out, in]`; Burn row-major Linear is `[in, out]` — we transpose.
fn apply_hf_tensors(
    model: &mut Qwen3Vl,
    tensors: &HashMap<String, TensorData>,
    device: &Device,
) -> Result<(), LoadError> {
    let get = |key: &str| -> Result<&TensorData, LoadError> {
        tensors
            .get(key)
            .ok_or_else(|| LoadError::Msg(format!("missing tensor {key}")))
    };

    // Embeddings
    if let Ok(w) = get("model.language_model.embed_tokens.weight") {
        model.decoder.set_embed_weight(tensor2(w, device));
    } else if let Ok(w) = get("model.embed_tokens.weight") {
        model.decoder.set_embed_weight(tensor2(w, device));
    }

    let n_layers = model.decoder.config.num_hidden_layers;
    for i in 0..n_layers {
        let prefix = format!("model.language_model.layers.{i}");
        let prefix_alt = format!("model.layers.{i}");
        let p = if tensors.keys().any(|k| k.starts_with(&prefix)) {
            prefix
        } else {
            prefix_alt
        };

        let q = transpose2(get(&format!("{p}.self_attn.q_proj.weight"))?, device);
        let k = transpose2(get(&format!("{p}.self_attn.k_proj.weight"))?, device);
        let v = transpose2(get(&format!("{p}.self_attn.v_proj.weight"))?, device);
        let qkv = Tensor::cat(vec![q, k, v], 1);
        let o = transpose2(get(&format!("{p}.self_attn.o_proj.weight"))?, device);
        let q_norm = tensor1(get(&format!("{p}.self_attn.q_norm.weight"))?, device);
        let k_norm = tensor1(get(&format!("{p}.self_attn.k_norm.weight"))?, device);
        let gate = transpose2(get(&format!("{p}.mlp.gate_proj.weight"))?, device);
        let up = transpose2(get(&format!("{p}.mlp.up_proj.weight"))?, device);
        let gate_up = Tensor::cat(vec![gate, up], 1);
        let down = transpose2(get(&format!("{p}.mlp.down_proj.weight"))?, device);
        let input_ln = tensor1(get(&format!("{p}.input_layernorm.weight"))?, device);
        let post_ln = tensor1(get(&format!("{p}.post_attention_layernorm.weight"))?, device);

        model.decoder.load_layer_weights(
            i, qkv, o, q_norm, k_norm, gate_up, down, input_ln, post_ln,
        );
    }

    if let Ok(w) = get("model.language_model.norm.weight") {
        model.decoder.set_norm_weight(tensor1(w, device));
    } else if let Ok(w) = get("model.norm.weight") {
        model.decoder.set_norm_weight(tensor1(w, device));
    }

    if let Ok(w) = get("lm_head.weight") {
        model.decoder.set_lm_head_weight(transpose2(w, device));
    }

    model.vision.load_hf_tensors(tensors, device);
    Ok(())
}

fn transpose2(data: &TensorData, device: &Device) -> Tensor<2> {
    tensor2(data, device).transpose()
}

/// Save a model snapshot as burnpack (used by converter).
///
/// - [`WeightQuant::Fp16`] / [`WeightQuant::Q4Mix`]: cast residual float params to F16
///   (QFloat passes through for `Q4Mix`).
/// - [`WeightQuant::Q4`]: expects the model to already be fully Q4-quantized in memory.
/// - [`WeightQuant::Fp32`]: stores tensors as currently resident (typically F32).
pub fn save_burnpack(
    model: &mut Qwen3Vl,
    path: impl AsRef<Path>,
    quant: WeightQuant,
) -> Result<(), LoadError> {
    let mut store = BurnpackStore::from_file(path.as_ref()).overwrite(true);
    match quant {
        WeightQuant::Fp16 | WeightQuant::Q4Mix => {
            store = store.with_to_adapter(FloatCastAdapter::to(DType::F16));
        }
        WeightQuant::Fp32 | WeightQuant::Q4 => {}
    }
    model
        .save_into(&mut store)
        .map_err(|e| LoadError::Msg(format!("burnpack save: {e:?}")))?;
    Ok(())
}

/// Split a burnpack into fixed-size byte shards for Hub / CDN hosting.
///
/// Returns relative shard file names in order. The original `pack_path` is kept
/// as the concatenation source; callers may delete it after sharding if desired.
pub fn shard_file(
    pack_path: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    shard_mib: usize,
    stem: &str,
) -> Result<Vec<String>, LoadError> {
    let pack_path = pack_path.as_ref();
    let out_dir = out_dir.as_ref();
    let data = std::fs::read(pack_path)?;
    let chunk = shard_mib.max(1).saturating_mul(1024 * 1024);
    let n = data.len().div_ceil(chunk).max(1);
    let mut names = Vec::with_capacity(n);
    for (i, part) in data.chunks(chunk).enumerate() {
        let name = format!("{stem}-{i:05}-of-{n:05}.bpk.part");
        std::fs::write(out_dir.join(&name), part)?;
        names.push(name);
    }
    Ok(names)
}

/// Concatenate shard files (or a single burnpack) into one byte buffer.
pub fn concat_shard_files(paths: &[impl AsRef<Path>]) -> Result<Vec<u8>, LoadError> {
    let mut out = Vec::new();
    for p in paths {
        let mut part = std::fs::read(p.as_ref())?;
        out.append(&mut part);
    }
    Ok(out)
}
