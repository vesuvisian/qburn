//! Offline weight conversion: HF safetensors → quantized burnpacks + sharded manifest.
//!
//! # Examples
//!
//! ```bash
//! cargo run -p qburn-converter --release -- \
//!   --model-dir weights/Qwen3-VL-2B-Instruct \
//!   --out-dir weights/burnpack-q4-mix \
//!   --quant q4-mix --shard-mib 512
//! ```
//!
//! Upload the out-dir to Hugging Face Hub. Point the Pages demo at `manifest.json`
//! (`window.QBURN_MANIFEST`). Prefer `q4-mix` for browser quality. See `web/static/how-it-works.html`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use qburn::load::{load_from_hf_dir, save_burnpack, shard_file};
use qburn::quant::apply_weight_quant;
use qburn::{resolve_device, Qwen3VlConfig, WeightQuant};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "qburn-converter", about = "Convert Qwen3-VL weights to burnpack shards")]
struct Args {
    /// Local HuggingFace model directory (or download target).
    #[arg(long)]
    model_dir: PathBuf,

    /// Output directory for burnpack shards + manifest.json.
    #[arg(long)]
    out_dir: PathBuf,

    /// Optional HuggingFace repo id to download first (e.g. Qwen/Qwen3-VL-2B-Instruct).
    #[arg(long)]
    hf_repo: Option<String>,

    /// Weight storage mode: fp32, fp16 (F16 cast), q4 (uniform Q4S), or q4-mix
    /// (Q4S on bulky Linears; embeddings / norms / vision I/O stay F16).
    #[arg(long, default_value = "fp16")]
    quant: QuantMode,

    /// Max shard size in MiB for Hub-friendly parts (default 512).
    #[arg(long, default_value_t = 512)]
    shard_mib: usize,

    /// Keep the unsharded `.bpk` after writing `.bpk.part` shards.
    #[arg(long, default_value_t = false)]
    keep_full: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantMode {
    Fp32,
    Fp16,
    Q4,
    #[value(name = "q4-mix")]
    Q4Mix,
}

impl From<QuantMode> for WeightQuant {
    fn from(value: QuantMode) -> Self {
        match value {
            QuantMode::Fp32 => WeightQuant::Fp32,
            QuantMode::Fp16 => WeightQuant::Fp16,
            QuantMode::Q4 => WeightQuant::Q4,
            QuantMode::Q4Mix => WeightQuant::Q4Mix,
        }
    }
}

#[derive(Serialize)]
struct Manifest {
    model: String,
    quant: String,
    config: Qwen3VlConfig,
    /// Ordered byte parts that concatenate to the burnpack.
    shards: Vec<ShardEntry>,
    tokenizer: String,
    notes: String,
}

#[derive(Serialize)]
struct ShardEntry {
    file: String,
    bytes: u64,
    sha256: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(repo) = &args.hf_repo {
        download_hf_snapshot(repo, &args.model_dir)?;
    }
    if !args.model_dir.join("config.json").exists() {
        bail!(
            "missing config.json in {} — pass --hf-repo or point --model-dir at a local checkout",
            args.model_dir.display()
        );
    }

    fs::create_dir_all(&args.out_dir)?;
    let quant: WeightQuant = args.quant.into();
    let device = resolve_device();
    eprintln!("loading HF weights on {:?} (quant={})…", device, quant.as_str());

    let model = load_from_hf_dir(&args.model_dir, &device).context("load HF dir")?;
    let cfg = model.config.clone();

    eprintln!("applying {}…", quant.as_str());
    let mut model = apply_weight_quant(model, quant);

    let tok_src = args.model_dir.join("tokenizer.json");
    if tok_src.exists() {
        fs::copy(&tok_src, args.out_dir.join("tokenizer.json"))?;
    }

    let pack_name = quant.pack_name();
    let pack_path = args.out_dir.join(pack_name);
    eprintln!("writing {} …", pack_path.display());
    save_burnpack(&mut model, &pack_path, quant).context("save burnpack")?;

    fs::write(
        args.out_dir.join("config.json"),
        serde_json::to_string_pretty(&cfg)?,
    )?;

    let meta = fs::metadata(&pack_path)?;
    eprintln!(
        "full pack: {:.1} MiB",
        meta.len() as f64 / (1024.0 * 1024.0)
    );

    let stem = pack_name.trim_end_matches(".bpk");
    let shard_names = shard_file(&pack_path, &args.out_dir, args.shard_mib, stem)
        .context("shard burnpack")?;
    eprintln!(
        "split into {} shard(s) of ≤{} MiB",
        shard_names.len(),
        args.shard_mib
    );

    if !args.keep_full && shard_names.len() > 1 {
        fs::remove_file(&pack_path)?;
        eprintln!("removed unsharded {pack_name} (pass --keep-full to retain)");
    } else if shard_names.len() == 1 {
        // Single part — rename part back to the canonical pack name for simplicity.
        let part = args.out_dir.join(&shard_names[0]);
        if part != pack_path {
            let _ = fs::remove_file(&pack_path);
            fs::rename(&part, &pack_path)?;
        }
    }

    let shard_entries: Vec<ShardEntry> = if shard_names.len() == 1 {
        let bytes = fs::metadata(&pack_path)?.len();
        vec![ShardEntry {
            file: pack_name.into(),
            bytes,
            sha256: None,
        }]
    } else {
        shard_names
            .iter()
            .map(|name| {
                let bytes = fs::metadata(args.out_dir.join(name))
                    .map(|m| m.len())
                    .unwrap_or(0);
                ShardEntry {
                    file: name.clone(),
                    bytes,
                    sha256: None,
                }
            })
            .collect()
    };

    let total: u64 = shard_entries.iter().map(|s| s.bytes).sum();
    if total >= 100 * 1024 * 1024 {
        eprintln!(
            "note: total {:.1} MiB — host on Hugging Face Hub (not GitHub git). Parts are ≤{} MiB.",
            total as f64 / (1024.0 * 1024.0),
            args.shard_mib
        );
    }

    let manifest = Manifest {
        model: "Qwen3-VL-2B-Instruct".into(),
        quant: quant.as_str().into(),
        config: cfg,
        shards: shard_entries,
        tokenizer: "tokenizer.json".into(),
        notes: match quant {
            WeightQuant::Q4Mix => {
                "Concatenate shards in order to rebuild the burnpack. quant=q4-mix: \
                 Q4S (block 32, MinMax) on decoder/vision-block Linears only; \
                 embeddings, norms, and vision patch/pos/merger/DeepStack stay F16. \
                 Vision ViT blocks stay Q4 so the pack fits under the ~2 GiB wasm32 \
                 shard-concat ceiling. Burn PTQ is MinMax only (no AWQ/GPTQ). \
                 On load, lm_head is re-tied from float embed_tokens when configured."
                    .into()
            }
            _ => format!(
                "Concatenate shards in order to rebuild the burnpack. quant={} via Burn FloatCast/Q4S.",
                quant.as_str()
            ),
        },
    };
    fs::write(
        args.out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    eprintln!("wrote manifest.json");
    Ok(())
}

fn download_hf_snapshot(repo: &str, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    let files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    ];
    let client = reqwest::blocking::Client::builder()
        .user_agent("qburn-converter/0.1")
        .build()?;

    for f in files {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{f}");
        let path = dest.join(f);
        if path.exists() {
            continue;
        }
        eprintln!("fetch {f}…");
        let resp = client.get(&url).send();
        match resp {
            Ok(r) if r.status().is_success() => {
                fs::write(path, r.bytes()?)?;
            }
            _ => eprintln!("  skip {f} (not found)"),
        }
    }

    let index_path = dest.join("model.safetensors.index.json");
    let mut shard_names = Vec::new();
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_str(&fs::read_to_string(&index_path)?)?;
        if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
            for v in map.values() {
                if let Some(s) = v.as_str() {
                    if !shard_names.iter().any(|x| x == s) {
                        shard_names.push(s.to_string());
                    }
                }
            }
        }
    } else {
        shard_names.push("model.safetensors".into());
    }

    for name in shard_names {
        let path = dest.join(&name);
        if path.exists() {
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{name}");
        eprintln!("download {name}…");
        let mut resp = client.get(&url).send()?.error_for_status()?;
        let total = resp.content_length().unwrap_or(0);
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes}",
            )?
            .progress_chars("#>-"),
        );
        let mut file = fs::File::create(&path)?;
        let mut downloaded = 0u64;
        let mut buf = [0u8; 1024 * 256];
        use std::io::{Read, Write};
        loop {
            let n = resp.read(&mut buf)?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])?;
            downloaded += n as u64;
            pb.set_position(downloaded);
        }
        pb.finish_and_clear();
    }
    Ok(())
}
