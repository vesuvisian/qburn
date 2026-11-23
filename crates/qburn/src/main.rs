//! Native CLI for Qwen3-VL scene description / text generation.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use qburn::generate::{generate_text, generate_text_streaming};
use qburn::load::load_model_dir;
use qburn::model::RotaryEmbedding;
use qburn::tokenizer::QburnTokenizer;
use qburn::vision::ImageProcessor;
use qburn::{
    backend_name, resolve_device, with_length_budget, DescribeTimings, Device, PrefillProbe,
    Qwen3Vl, Qwen3VlConfig,
};

/// Image extensions discovered when scanning directories (PNG / JPEG / WebP).
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "jpe", "jfif"];

#[derive(Parser, Debug)]
#[command(name = "qburn", about = "Burn-native Qwen3-VL")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Text-only generation.
    Prompt {
        /// HF checkpoint dir or converter burnpack out-dir (`manifest.json` + shards).
        #[arg(long)]
        model: PathBuf,
        /// User prompt text.
        prompt: String,
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Softmax temperature; `0` = greedy argmax.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Stream tokens to stdout as they are produced.
        #[arg(long)]
        stream: bool,
        /// Print latency metrics to stderr (TTFT, total, tok/s).
        #[arg(long)]
        metrics: bool,
    },
    /// Describe one or more images (model loaded once).
    Describe {
        /// HF checkpoint dir or converter burnpack out-dir (`manifest.json` + shards).
        #[arg(long)]
        model: PathBuf,
        /// Image files and/or directories (dirs yield supported image files).
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,
        /// Recurse into subdirectories when a path is a folder.
        #[arg(short, long)]
        recursive: bool,
        #[arg(long, default_value = "Describe this scene.")]
        prompt: String,
        /// Generation budget. The prompt is steered to finish within this many tokens.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Softmax temperature; `0` = greedy argmax.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Print latency metrics to stderr (preprocess, TTFT, total, tok/s).
        #[arg(long)]
        metrics: bool,
        /// Dump prefill first-token top-10 logits to stderr (backend parity).
        #[arg(long)]
        probe: bool,
    },
    /// Print the built-in 2B config JSON.
    Config,
}

/// Wall-clock timings for one generation (printed with `--metrics`).
struct GenMetrics {
    /// Image decode / resize / patchify (describe only).
    preprocess_ms: Option<f64>,
    /// Vision encoder + multimodal splice (describe only).
    vision_ms: Option<f64>,
    /// First decoder forward + first sample (describe only).
    prefill_ms: Option<f64>,
    /// Autoregressive steps after the first token (describe), or post-TTFT (prompt).
    decode_ms: Option<f64>,
    /// Time from generation start until the first token.
    ttft_ms: Option<f64>,
    /// Full generation wall time.
    total_ms: f64,
    tokens: usize,
}

impl GenMetrics {
    fn decode_tok_s(&self) -> Option<f64> {
        let decode_ms = self.decode_ms.or_else(|| {
            let ttft = self.ttft_ms?;
            Some(self.total_ms - ttft)
        })?;
        if self.tokens <= 1 || decode_ms <= 0.0 {
            return None;
        }
        Some((self.tokens - 1) as f64 / (decode_ms / 1000.0))
    }

    fn print(&self) {
        let mut parts = Vec::new();
        if let Some(ms) = self.preprocess_ms {
            parts.push(format!("preprocess={ms:.1}ms"));
        }
        if let Some(ms) = self.vision_ms {
            parts.push(format!("vision={ms:.1}ms"));
        }
        if let Some(ms) = self.prefill_ms {
            parts.push(format!("prefill={ms:.1}ms"));
        }
        match self.ttft_ms {
            Some(ms) => parts.push(format!("ttft={ms:.1}ms")),
            None => parts.push("ttft=n/a".into()),
        }
        if let Some(ms) = self.decode_ms {
            parts.push(format!("decode={ms:.1}ms"));
        }
        parts.push(format!("total={:.2}s", self.total_ms / 1000.0));
        parts.push(format!("tokens={}", self.tokens));
        if let Some(tps) = self.decode_tok_s() {
            parts.push(format!("tok/s={tps:.2}"));
        }
        eprintln!("metrics: {}", parts.join("  "));
    }
}

/// Tracks TTFT via the first token callback; finish after generation returns.
struct MetricsProbe {
    start: Instant,
    first_token: Option<Instant>,
}

impl MetricsProbe {
    fn start() -> Self {
        Self {
            start: Instant::now(),
            first_token: None,
        }
    }

    fn on_token(&mut self) {
        if self.first_token.is_none() {
            self.first_token = Some(Instant::now());
        }
    }

    fn finish(self, tokens: usize) -> GenMetrics {
        let ttft_ms = self
            .first_token
            .map(|t| t.duration_since(self.start).as_secs_f64() * 1000.0);
        let total_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        GenMetrics {
            preprocess_ms: None,
            vision_ms: None,
            prefill_ms: None,
            decode_ms: ttft_ms.map(|ttft| (total_ms - ttft).max(0.0)),
            ttft_ms,
            total_ms,
            tokens,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Config => {
            let cfg = Qwen3VlConfig::qwen3_vl_2b();
            println!("{}", serde_json::to_string_pretty(&cfg)?);
        }
        Commands::Prompt {
            model,
            prompt,
            system,
            max_tokens,
            temperature,
            stream,
            metrics,
        } => {
            run_prompt(
                &model,
                &system,
                &prompt,
                max_tokens,
                temperature,
                stream,
                metrics,
            )?;
        }
        Commands::Describe {
            model,
            paths,
            recursive,
            prompt,
            max_tokens,
            temperature,
            metrics,
            probe,
        } => {
            let images = collect_images(&paths, recursive)?;
            run_describe(
                &model,
                &images,
                &prompt,
                max_tokens,
                temperature,
                metrics,
                probe,
            )?;
        }
    }
    Ok(())
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|ext| e.eq_ignore_ascii_case(ext))
        })
}

/// Expand file and directory paths into a sorted, deduped list of image files.
fn collect_images(paths: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        if !path.exists() {
            bail!("path not found: {}", path.display());
        }
        if path.is_file() {
            // Explicit files are always queued; preprocess will reject non-images.
            out.push(path.clone());
        } else if path.is_dir() {
            collect_from_dir(path, recursive, &mut out)
                .with_context(|| format!("scan {}", path.display()))?;
        } else {
            bail!("not a file or directory: {}", path.display());
        }
    }
    out.sort();
    out.dedup();
    if out.is_empty() {
        bail!(
            "no images found (looking for {}{})",
            IMAGE_EXTENSIONS.join(", "),
            if recursive { "; recursive" } else { "" }
        );
    }
    Ok(out)
}

fn collect_from_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let ft = entry
            .file_type()
            .with_context(|| format!("file_type {}", path.display()))?;
        if ft.is_file() {
            if is_image_path(&path) {
                out.push(path);
            }
        } else if recursive && ft.is_dir() {
            collect_from_dir(&path, true, out)?;
        }
    }
    Ok(())
}

fn run_prompt(
    model_dir: &PathBuf,
    system: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    stream: bool,
    metrics: bool,
) -> Result<()> {
    let device = resolve_device();
    eprintln!(
        "loading model from {} on {} ({:?})…",
        model_dir.display(),
        backend_name(),
        device
    );
    let vl = load_model_dir(model_dir, &device).context("load model")?;
    let tok = QburnTokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e))?;

    let chat = tok.apply_chat_template(system, prompt);
    let ids = tok.encode(&chat).map_err(|e| anyhow::anyhow!(e))?;
    let rope = RotaryEmbedding::new(
        vl.decoder.config.head_dim,
        (ids.len() + max_tokens + 64).next_power_of_two().max(2048),
        vl.decoder.config.rope_theta,
        &device,
    );
    let eos = vl.decoder.config.eos_token_id;

    // Metrics need per-token callbacks, so use the streaming path when enabled.
    if stream || metrics {
        let mut probe = metrics.then(MetricsProbe::start);
        let tokens = generate_text_streaming(
            &vl.decoder,
            &rope,
            &ids,
            max_tokens,
            eos,
            temperature,
            &device,
            |tid| {
                if let Some(p) = probe.as_mut() {
                    p.on_token();
                }
                if stream {
                    if let Ok(piece) = tok.decode(&[tid]) {
                        print!("{piece}");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                }
            },
        );
        if stream {
            println!();
        } else {
            let text = tok.decode(&tokens).map_err(|e| anyhow::anyhow!(e))?;
            println!("{text}");
        }
        eprintln!("generated {} tokens", tokens.len());
        if let Some(p) = probe {
            p.finish(tokens.len()).print();
        }
    } else {
        let tokens = generate_text(
            &vl.decoder,
            &rope,
            &ids,
            max_tokens,
            eos,
            temperature,
            &device,
        );
        let text = tok.decode(&tokens).map_err(|e| anyhow::anyhow!(e))?;
        println!("{text}");
    }
    Ok(())
}

fn run_describe(
    model_dir: &PathBuf,
    images: &[PathBuf],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    metrics: bool,
    probe: bool,
) -> Result<()> {
    let device = resolve_device();
    eprintln!(
        "loading model from {} on {} ({:?})…",
        model_dir.display(),
        backend_name(),
        device
    );
    let vl = load_model_dir(model_dir, &device).context("load model")?;
    let tok = QburnTokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e))?;
    let processor = ImageProcessor::from_config(&vl.config.vision_config);
    let prompt = with_length_budget(prompt, max_tokens);
    let batch = images.len() > 1;
    let mut failures = 0usize;
    let mut ttfts = Vec::new();

    for (i, image) in images.iter().enumerate() {
        if batch {
            if i > 0 {
                println!();
            }
            eprintln!("=== [{}/{}] {} ===", i + 1, images.len(), image.display());
        }

        match describe_one(
            &vl,
            &tok,
            &processor,
            image,
            &prompt,
            max_tokens,
            temperature,
            &device,
            metrics,
            probe,
        ) {
            Ok(m) => {
                eprintln!("generated {} tokens", m.tokens);
                if metrics {
                    if let Some(ttft) = m.ttft_ms {
                        ttfts.push(ttft);
                    }
                    m.print();
                }
            }
            Err(e) => {
                failures += 1;
                eprintln!("error: {e:#}");
            }
        }
    }

    if metrics && ttfts.len() > 1 {
        let avg = ttfts.iter().sum::<f64>() / ttfts.len() as f64;
        eprintln!(
            "metrics summary: images={}  avg_ttft={:.1}ms  min_ttft={:.1}ms  max_ttft={:.1}ms",
            ttfts.len(),
            avg,
            ttfts.iter().cloned().fold(f64::INFINITY, f64::min),
            ttfts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
    }

    if failures > 0 {
        bail!("{failures}/{} image(s) failed", images.len());
    }
    Ok(())
}

fn describe_one(
    vl: &Qwen3Vl,
    tok: &QburnTokenizer,
    processor: &ImageProcessor,
    image: &PathBuf,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    device: &Device,
    metrics: bool,
    probe: bool,
) -> Result<GenMetrics> {
    let preprocess_start = Instant::now();
    let img = processor
        .preprocess_path(image)
        .map_err(|e| anyhow::anyhow!(e))?;
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "image grid {:?} → {} merge tokens",
        img.grid_thw, img.num_merge_tokens
    );

    let chat = tok.apply_vl_chat_template(
        prompt,
        img.num_merge_tokens,
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
    );
    let ids = tok.encode(&chat).map_err(|e| anyhow::anyhow!(e))?;

    let mut timings = DescribeTimings::default();
    let mut prefill_probe = PrefillProbe::default();
    let mut cb = |tid: u32| {
        if let Ok(piece) = tok.decode(&[tid]) {
            print!("{piece}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    };
    let tokens = pollster::block_on(vl.describe_greedy(
        &ids,
        &img,
        max_tokens,
        temperature,
        device,
        Some(&mut cb),
        Some(&mut timings),
        probe.then_some(&mut prefill_probe),
    ));
    println!();
    if probe {
        print_prefill_probe(tok, &prefill_probe);
    }
    if tokens.is_empty() {
        bail!("model produced no tokens — check vision weight loading");
    }
    Ok(GenMetrics {
        preprocess_ms: metrics.then_some(preprocess_ms),
        vision_ms: metrics.then_some(timings.vision_ms),
        prefill_ms: metrics.then_some(timings.prefill_ms),
        decode_ms: metrics.then_some(timings.decode_ms),
        ttft_ms: Some(timings.ttft_ms()),
        total_ms: timings.total_ms(),
        tokens: tokens.len(),
    })
}

fn print_prefill_probe(tok: &QburnTokenizer, probe: &PrefillProbe) {
    let greedy_piece = tok
        .decode(&[probe.greedy_id])
        .unwrap_or_else(|_| "?".into());
    eprintln!(
        "prefill probe: backend={} greedy_id={} greedy={:?}",
        backend_name(),
        probe.greedy_id,
        greedy_piece
    );
    for (rank, (id, logit)) in probe.topk.iter().enumerate() {
        let piece = tok.decode(&[*id]).unwrap_or_else(|_| "?".into());
        eprintln!(
            "  top{:<2} id={:<6} logit={:>9.4} piece={:?}",
            rank + 1,
            id,
            logit,
            piece
        );
    }
}
