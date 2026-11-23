//! Native CLI for Qwen3-VL scene description / text generation.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use qburn::generate::{generate_text, generate_text_streaming};
use qburn::load::load_model_dir;
use qburn::model::RotaryEmbedding;
use qburn::tokenizer::QburnTokenizer;
use qburn::vision::ImageProcessor;
use qburn::{
    backend_name, resolve_device, sample_frame_indices, temporal_group_timestamps,
    with_length_budget, DescribeTimings, Device, PrefillProbe, Qwen3Vl, Qwen3VlConfig,
    VIDEO_DEFAULT_FPS, VIDEO_MAX_FRAMES_CLI, VIDEO_MIN_FRAMES,
};

/// Image extensions discovered when scanning directories (PNG / JPEG / WebP).
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "jpe", "jfif"];
/// Container formats decoded via `ffmpeg` for single-video describe.
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "webm", "mov", "mkv", "avi"];

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
    /// Describe one or more images, or a single video (model loaded once).
    Describe {
        /// HF checkpoint dir or converter burnpack out-dir (`manifest.json` + shards).
        #[arg(long)]
        model: PathBuf,
        /// Image files and/or directories, or one video file / `--video` frame folder.
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,
        /// Recurse into subdirectories when a path is a folder (images only).
        #[arg(short, long)]
        recursive: bool,
        /// Treat all images as one multi-image prompt (default: caption each file).
        /// Incompatible with video.
        #[arg(long)]
        together: bool,
        /// Treat the single path as one video: a container file or an ordered
        /// frame directory (skips per-image batch expansion).
        #[arg(long)]
        video: bool,
        /// Sample this many frames per second from the video (default 2).
        /// Mutually exclusive with `--num-frames`.
        #[arg(long)]
        fps: Option<f64>,
        /// Sample exactly this many frames (uniform). Mutually exclusive with `--fps`.
        #[arg(long)]
        num_frames: Option<usize>,
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
            together,
            video,
            fps,
            num_frames,
            prompt,
            max_tokens,
            temperature,
            metrics,
            probe,
        } => {
            if fps.is_some() && num_frames.is_some() {
                bail!("--fps and --num-frames are mutually exclusive");
            }
            let media = classify_describe_paths(&paths, recursive, video)?;
            match media {
                DescribeMedia::Images(images) => {
                    if fps.is_some() || num_frames.is_some() {
                        bail!("--fps / --num-frames apply only to video describe");
                    }
                    run_describe(
                        &model,
                        &images,
                        together,
                        &prompt,
                        max_tokens,
                        temperature,
                        metrics,
                        probe,
                    )?;
                }
                DescribeMedia::Video(source) => {
                    if together {
                        bail!("--together is image-only; omit it for video");
                    }
                    run_describe_video(
                        &model,
                        &source,
                        fps,
                        num_frames,
                        &prompt,
                        max_tokens,
                        temperature,
                        metrics,
                        probe,
                    )?;
                }
            }
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

fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| {
            VIDEO_EXTENSIONS
                .iter()
                .any(|ext| e.eq_ignore_ascii_case(ext))
        })
}

enum VideoSource {
    /// Container file decoded with ffmpeg.
    File(PathBuf),
    /// Ordered image frames (lexicographic) — no ffmpeg.
    FrameDir(PathBuf),
}

enum DescribeMedia {
    Images(Vec<PathBuf>),
    Video(VideoSource),
}

/// Classify describe paths: either a batch of images, or exactly one video.
fn classify_describe_paths(
    paths: &[PathBuf],
    recursive: bool,
    force_video: bool,
) -> Result<DescribeMedia> {
    if paths.is_empty() {
        bail!("no paths given");
    }

    let video_files: Vec<_> = paths
        .iter()
        .filter(|p| p.is_file() && is_video_path(p))
        .cloned()
        .collect();

    if force_video {
        if paths.len() != 1 {
            bail!("--video expects exactly one path (file or frame directory)");
        }
        let path = &paths[0];
        if !path.exists() {
            bail!("path not found: {}", path.display());
        }
        if path.is_file() {
            if !is_video_path(path) && !is_image_path(path) {
                // Allow a single image as a degenerate video; prefer video ext.
                bail!(
                    "--video file must be a video ({})",
                    VIDEO_EXTENSIONS.join(", ")
                );
            }
            if is_video_path(path) {
                return Ok(DescribeMedia::Video(VideoSource::File(path.clone())));
            }
            bail!("--video with an image file is unsupported; pass a container or frame directory");
        }
        if path.is_dir() {
            return Ok(DescribeMedia::Video(VideoSource::FrameDir(path.clone())));
        }
        bail!("not a file or directory: {}", path.display());
    }

    if !video_files.is_empty() {
        if video_files.len() != 1 || paths.len() != 1 {
            bail!(
                "single-video describe takes exactly one video file (found {}); \
                 do not mix with images",
                video_files.len()
            );
        }
        return Ok(DescribeMedia::Video(VideoSource::File(video_files[0].clone())));
    }

    let images = collect_images(paths, recursive)?;
    Ok(DescribeMedia::Images(images))
}

/// Expand file and directory paths into a sorted, deduped list of image files.
fn collect_images(paths: &[PathBuf], recursive: bool) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for path in paths {
        if !path.exists() {
            bail!("path not found: {}", path.display());
        }
        if path.is_file() {
            if is_video_path(path) {
                bail!(
                    "video file {} mixed into image describe — pass it alone",
                    path.display()
                );
            }
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

fn collect_frame_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_from_dir(dir, false, &mut out)
        .with_context(|| format!("scan frame dir {}", dir.display()))?;
    out.sort();
    out.dedup();
    if out.is_empty() {
        bail!(
            "no frames found in {} (looking for {})",
            dir.display(),
            IMAGE_EXTENSIONS.join(", ")
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

fn run_describe_video(
    model_dir: &PathBuf,
    source: &VideoSource,
    fps: Option<f64>,
    num_frames: Option<usize>,
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

    let label = match source {
        VideoSource::File(p) => p.display().to_string(),
        VideoSource::FrameDir(p) => format!("{} (frames)", p.display()),
    };
    eprintln!("video: {label}");

    let preprocess_start = Instant::now();
    let (frame_paths, frame_times) = load_video_frame_paths(source, fps, num_frames)?;
    eprintln!("sampled {} frame(s)", frame_paths.len());

    let mut frames = Vec::with_capacity(frame_paths.len());
    for (i, path) in frame_paths.iter().enumerate() {
        let img = image::open(path)
            .with_context(|| format!("open frame {}: {}", i, path.display()))?;
        frames.push(img);
    }
    let input = processor
        .preprocess_video_frames(&frames)
        .map_err(|e| anyhow::anyhow!(e))?;
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
    let (gt, gh, gw) = input.grid_thw;
    let merge = processor.spatial_merge_size.max(1);
    let tokens_per_group = (gh / merge) * (gw / merge);
    let group_times =
        temporal_group_timestamps(&frame_times, processor.temporal_patch_size.max(1));
    if group_times.len() != gt {
        bail!(
            "timestamp groups {} != video grid_t {gt} (frames={}, times={:?})",
            group_times.len(),
            frame_paths.len(),
            frame_times
        );
    }
    eprintln!(
        "video grid {:?} → {} merge tokens ({} per temporal group), timestamps {:?}",
        input.grid_thw, input.num_merge_tokens, tokens_per_group, group_times
    );

    let chat = tok.apply_vl_video_chat_template(
        &prompt,
        tokens_per_group,
        &group_times,
        "<|vision_start|>",
        "<|video_pad|>",
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
        std::slice::from_ref(&input),
        max_tokens,
        temperature,
        &device,
        Some(&mut cb),
        Some(&mut timings),
        probe.then_some(&mut prefill_probe),
    ));
    println!();
    if probe {
        print_prefill_probe(&tok, &prefill_probe);
    }
    if tokens.is_empty() {
        bail!("model produced no tokens — check vision weight loading");
    }
    eprintln!("generated {} tokens", tokens.len());
    if metrics {
        GenMetrics {
            preprocess_ms: Some(preprocess_ms),
            vision_ms: Some(timings.vision_ms),
            prefill_ms: Some(timings.prefill_ms),
            decode_ms: Some(timings.decode_ms),
            ttft_ms: Some(timings.ttft_ms()),
            total_ms: timings.total_ms(),
            tokens: tokens.len(),
        }
        .print();
    }
    Ok(())
}

/// Resolve sampled frame image paths + timeline seconds for a video source.
///
/// Container files require `ffmpeg` on PATH. Frame directories are subsampled
/// in-process with [`sample_frame_indices`] (times assume 24 fps source).
fn load_video_frame_paths(
    source: &VideoSource,
    fps: Option<f64>,
    num_frames: Option<usize>,
) -> Result<(Vec<PathBuf>, Vec<f64>)> {
    match source {
        VideoSource::FrameDir(dir) => {
            let all = collect_frame_dir(dir)?;
            let src_fps = 24.0;
            let indices = sample_frame_indices(
                all.len(),
                src_fps,
                fps.or(if num_frames.is_none() {
                    Some(VIDEO_DEFAULT_FPS)
                } else {
                    None
                }),
                num_frames,
                VIDEO_MIN_FRAMES,
                VIDEO_MAX_FRAMES_CLI,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
            let times = indices
                .iter()
                .map(|&i| i as f64 / src_fps)
                .collect::<Vec<_>>();
            let paths = indices.into_iter().map(|i| all[i].clone()).collect();
            Ok((paths, times))
        }
        VideoSource::File(path) => extract_frames_ffmpeg(path, fps, num_frames),
    }
}

/// Probe duration, sample timestamps in `[0, duration)`, extract PNGs via ffmpeg.
///
/// Prefer format duration over packet counts — `nb_read_packets` can overshoot
/// the real timeline and send `-ss` past EOF (ffmpeg exits 0 with no file).
fn extract_frames_ffmpeg(
    path: &Path,
    fps: Option<f64>,
    num_frames: Option<usize>,
) -> Result<(Vec<PathBuf>, Vec<f64>)> {
    which_ffmpeg()?;

    let (duration, _src_fps) = probe_video_meta(path)?;
    let times = sample_timestamps(
        duration,
        fps.or(if num_frames.is_none() {
            Some(VIDEO_DEFAULT_FPS)
        } else {
            None
        }),
        num_frames,
        VIDEO_MIN_FRAMES,
        VIDEO_MAX_FRAMES_CLI,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    let tmp = tempfile_dir("qburn-video-")?;
    let mut out_paths = Vec::with_capacity(times.len());
    for (i, &t) in times.iter().enumerate() {
        let out = tmp.join(format!("frame-{i:05}.png"));
        extract_frame_at(path, t, &out)?;
        out_paths.push(out);
    }
    Ok((out_paths, times))
}

fn extract_frame_at(video: &Path, t: f64, dest: &Path) -> Result<()> {
    // Fast path: `-ss` before `-i`. Near EOF this can exit 0 with no file.
    let ok = run_ffmpeg_extract(video, t, dest, true)?;
    if ok {
        return Ok(());
    }
    let _ = std::fs::remove_file(dest);
    // Accurate fallback: decode then seek.
    let ok = run_ffmpeg_extract(video, t, dest, false)?;
    if !ok {
        bail!(
            "ffmpeg wrote no pixels for frame at t={t:.3}s from {}",
            video.display()
        );
    }
    Ok(())
}

fn run_ffmpeg_extract(video: &Path, t: f64, dest: &Path, ss_before_input: bool) -> Result<bool> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if ss_before_input {
        cmd.args(["-ss", &format!("{t:.6}"), "-i"]).arg(video);
    } else {
        cmd.arg("-i").arg(video).args(["-ss", &format!("{t:.6}")]);
    }
    let status = cmd
        .args(["-frames:v", "1", "-update", "1"])
        .arg(dest)
        .status()
        .with_context(|| format!("ffmpeg extract frame at t={t:.3}"))?;
    if !status.success() {
        return Ok(false);
    }
    Ok(std::fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false))
}

/// Uniform sample timestamps in `[0, duration)`, clamped off the EOF edge.
fn sample_timestamps(
    duration: f64,
    fps: Option<f64>,
    num_frames: Option<usize>,
    min_frames: usize,
    max_frames: usize,
) -> Result<Vec<f64>, String> {
    if !(duration > 0.0) {
        return Err("video has invalid duration".into());
    }
    if fps.is_some() && num_frames.is_some() {
        return Err("`fps` and `num_frames` are mutually exclusive".into());
    }
    let mut n = if let Some(n) = num_frames {
        n.max(1)
    } else {
        let fps = fps.unwrap_or(VIDEO_DEFAULT_FPS);
        ((duration * fps).round() as usize).max(1)
    };
    n = n.max(min_frames).min(max_frames).max(1);
    // Stay clear of EOF — late `-ss` often yields "No filtered frames" with exit 0.
    let end = (duration - 0.15).max(0.0);
    if n == 1 {
        return Ok(vec![(end / 2.0).min(end)]);
    }
    let mut times = Vec::with_capacity(n);
    for i in 0..n {
        times.push((i as f64 * end / (n - 1) as f64).min(end));
    }
    Ok(times)
}

fn which_ffmpeg() -> Result<()> {
    let status = Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        _ => bail!("ffmpeg not found on PATH (required for video container files)"),
    }
}

/// Returns `(duration_seconds, src_fps)`.
fn probe_video_meta(path: &Path) -> Result<(f64, f64)> {
    // Prefer ffprobe; fall back to ffmpeg -i stderr parse.
    if let Ok(meta) = probe_with_ffprobe(path) {
        return Ok(meta);
    }
    probe_with_ffmpeg_i(path)
}

fn probe_with_ffprobe(path: &Path) -> Result<(f64, f64)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("ffprobe")?;
    if !out.status.success() {
        bail!("ffprobe failed");
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).context("ffprobe json")?;
    let stream = v
        .get("streams")
        .and_then(|s| s.as_array())
        .and_then(|a| a.first());

    let fps = stream
        .and_then(|s| parse_rate(s.get("avg_frame_rate")).or_else(|| parse_rate(s.get("r_frame_rate"))))
        .unwrap_or(24.0);

    let duration = v
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&d| d > 0.0)
        .context("ffprobe: could not determine duration")?;

    Ok((duration, fps))
}

fn parse_rate(v: Option<&serde_json::Value>) -> Option<f64> {
    let s = v?.as_str()?;
    if s == "0/0" || s.is_empty() {
        return None;
    }
    if let Some((a, b)) = s.split_once('/') {
        let a: f64 = a.parse().ok()?;
        let b: f64 = b.parse().ok()?;
        if b == 0.0 {
            return None;
        }
        return Some(a / b);
    }
    s.parse().ok()
}

fn probe_with_ffmpeg_i(path: &Path) -> Result<(f64, f64)> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(path)
        .output()
        .context("ffmpeg -i")?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let mut fps = 24.0_f64;
    let mut duration_s: Option<f64> = None;
    for line in stderr.lines() {
        if let Some(idx) = line.find("Duration:") {
            // Duration: 00:00:10.50,
            let rest = &line[idx + "Duration:".len()..];
            let tok = rest.split(',').next().unwrap_or("").trim();
            if let Some(secs) = parse_hms(tok) {
                duration_s = Some(secs);
            }
        }
        if line.contains("Video:") {
            // ..., 30 fps, ... or 29.97 fps
            for part in line.split(',') {
                let part = part.trim();
                if let Some(num) = part.strip_suffix(" fps") {
                    if let Ok(v) = num.trim().parse::<f64>() {
                        fps = v;
                    }
                } else if let Some(num) = part.strip_suffix(" tbr") {
                    if let Ok(v) = num.trim().parse::<f64>() {
                        fps = v;
                    }
                }
            }
        }
    }
    let duration = duration_s.context("could not parse video duration (install ffprobe)")?;
    if !(duration > 0.0) {
        bail!("invalid video duration");
    }
    Ok((duration, fps))
}

fn parse_hms(s: &str) -> Option<f64> {
    let parts: Vec<_> = s.trim().split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sec)
}

fn tempfile_dir(prefix: &str) -> Result<PathBuf> {
    let mut dir = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    dir.push(format!("{prefix}{stamp}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    Ok(dir)
}

fn run_describe(
    model_dir: &PathBuf,
    images: &[PathBuf],
    together: bool,
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

    if together {
        eprintln!("multi-image prompt: {} file(s)", images.len());
        for (i, image) in images.iter().enumerate() {
            eprintln!("  [{}] {}", i + 1, image.display());
        }
        match describe_images(
            &vl,
            &tok,
            &processor,
            images,
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
                    m.print();
                }
            }
            Err(e) => bail!("{e:#}"),
        }
        return Ok(());
    }

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

        match describe_images(
            &vl,
            &tok,
            &processor,
            std::slice::from_ref(image),
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

fn describe_images(
    vl: &Qwen3Vl,
    tok: &QburnTokenizer,
    processor: &ImageProcessor,
    images: &[PathBuf],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    device: &Device,
    metrics: bool,
    probe: bool,
) -> Result<GenMetrics> {
    let preprocess_start = Instant::now();
    let mut inputs = Vec::with_capacity(images.len());
    for image in images {
        let img = processor
            .preprocess_path(image)
            .map_err(|e| anyhow::anyhow!("{}: {e}", image.display()))?;
        eprintln!(
            "image {} grid {:?} → {} merge tokens",
            image.display(),
            img.grid_thw,
            img.num_merge_tokens
        );
        inputs.push(img);
    }
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    let merge_counts: Vec<usize> = inputs.iter().map(|img| img.num_merge_tokens).collect();
    let chat = tok.apply_vl_chat_template_multi(
        prompt,
        &merge_counts,
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
        &inputs,
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
