//! Wasm bindings for Qwen3-VL scene description (WebGPU).
//!
//! Runs inside a **Web Worker** (`web/static/worker.js`):
//! - Fetch/cache via [`WorkerGlobalScope`] (not `window`).
//! - Device via [`qburn::resolve_device_async`] before loading shards.
//! - [`QburnApp::describe`] is async (`into_data_async` for argmax / pos-embed).
//!
//! Matmul on wasm uses Burn 0.23 Hardware-aware CubeCL selection (see
//! `patches/README.md`). Prefer q4-mix / q4 burnpacks — fp16/fp32 exceed
//! typical wasm32 memory after shard concat.

use std::cell::RefCell;

use burn::prelude::Device;
use futures::future::{try_join, try_join_all};
use qburn::load::{bytes_from_vec, load_from_burnpack_bytes};
use qburn::tokenizer::QburnTokenizer;
use qburn::vision::ImageProcessor;
use qburn::{resolve_device_async, PrefillProbe, Qwen3Vl, Qwen3VlConfig, WeightQuant};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Browser demo cap: dense ViT attention is O(patches²); ~256k pixels keeps
/// TTFT practical on typical WebGPU adapters while remaining merge-friendly.
const WEB_MAX_PIXELS: usize = 262_144;

thread_local! {
    static APP: RefCell<Option<AppState>> = const { RefCell::new(None) };
}

struct AppState {
    model: Qwen3Vl,
    tokenizer: QburnTokenizer,
    processor: ImageProcessor,
    device: Device,
    quant_label: String,
}

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

/// Window or dedicated worker — the demo loads wasm inside a Web Worker, where
/// `web_sys::window()` is `None`.
enum FetchScope {
    Window(web_sys::Window),
    Worker(web_sys::WorkerGlobalScope),
}

fn fetch_scope() -> Result<FetchScope, JsValue> {
    if let Some(window) = web_sys::window() {
        return Ok(FetchScope::Window(window));
    }
    js_sys::global()
        .dyn_into::<web_sys::WorkerGlobalScope>()
        .map(FetchScope::Worker)
        .map_err(|_| JsValue::from_str("no window or worker scope"))
}

impl FetchScope {
    fn caches(&self) -> Result<web_sys::CacheStorage, JsValue> {
        match self {
            Self::Window(w) => w.caches(),
            Self::Worker(w) => w.caches(),
        }
    }

    fn fetch_with_request(&self, request: &Request) -> js_sys::Promise {
        match self {
            Self::Window(w) => w.fetch_with_request(request),
            Self::Worker(w) => w.fetch_with_request(request),
        }
    }
}

/// Fetch URL bytes, preferring Cache API when available.
async fn fetch_bytes(url: &str, cache_name: &str) -> Result<Vec<u8>, JsValue> {
    let scope = fetch_scope()?;

    if let Ok(cache_storage) = scope.caches() {
        let cache_val = JsFuture::from(cache_storage.open(cache_name)).await?;
        let cache: web_sys::Cache = cache_val.into();
        let matched = JsFuture::from(cache.match_with_str(url)).await?;
        if !matched.is_undefined() && !matched.is_null() {
            let resp: Response = matched.into();
            let buf = JsFuture::from(resp.array_buffer()?).await?;
            return Ok(js_sys::Uint8Array::new(&buf).to_vec());
        }
    }

    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(url, &opts)?;
    let resp_val = JsFuture::from(scope.fetch_with_request(&request)).await?;
    let resp: Response = resp_val.into();
    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "fetch failed: {} {}",
            resp.status(),
            url
        )));
    }

    // Cache a clone before consuming the body — avoids a second full download
    // via `cache.add` (important for multi‑hundred‑MB shards on a local server).
    if let Ok(cache_storage) = scope.caches() {
        if let Ok(cache_val) = JsFuture::from(cache_storage.open(cache_name)).await {
            let cache: web_sys::Cache = cache_val.into();
            if let Ok(cloned) = resp.clone() {
                let _ = JsFuture::from(cache.put_with_str(url, &cloned)).await;
            }
        }
    }

    let buf = JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// Append `part` into `dst` without amortised doubling past wasm32 `isize::MAX`.
fn append_shard(dst: &mut Vec<u8>, part: &[u8]) -> Result<(), JsValue> {
    let need = dst
        .len()
        .checked_add(part.len())
        .ok_or_else(|| JsValue::from_str("weight concat overflow"))?;
    if need > (isize::MAX as usize) {
        return Err(JsValue::from_str(&format!(
            "weights too large for wasm32 ({} bytes) — use smaller quant / fewer shards",
            need
        )));
    }
    // Exact reserve: Vec's default doubling of ~1 GiB would request 2 GiB and
    // panic with `capacity overflow` on wasm32.
    if dst.capacity() < need {
        dst.reserve_exact(need - dst.len());
    }
    dst.extend_from_slice(part);
    Ok(())
}

#[wasm_bindgen]
pub struct QburnApp;

#[wasm_bindgen]
impl QburnApp {
    /// Load model from a manifest URL (JSON with `shards[0].file` relative to manifest).
    #[wasm_bindgen]
    pub async fn init(manifest_url: String) -> Result<QburnApp, JsValue> {
        let manifest_bytes = fetch_bytes(&manifest_url, "qburn-weights").await?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| JsValue::from_str(&format!("manifest json: {e}")))?;

        let base = manifest_url
            .rsplit_once('/')
            .map(|(b, _)| format!("{b}/"))
            .unwrap_or_default();

        let shard_entries = manifest["shards"]
            .as_array()
            .ok_or_else(|| JsValue::from_str("manifest missing shards[]"))?;
        let shard_files: Vec<(String, Option<usize>)> = shard_entries
            .iter()
            .filter_map(|s| {
                let file = s.get("file")?.as_str()?.to_string();
                let bytes = s
                    .get("bytes")
                    .and_then(|b| b.as_u64())
                    .and_then(|n| usize::try_from(n).ok());
                Some((file, bytes))
            })
            .collect();
        if shard_files.is_empty() {
            return Err(JsValue::from_str("manifest shards[] is empty"));
        }
        let tok_name = manifest
            .get("tokenizer")
            .and_then(|v| v.as_str())
            .unwrap_or("tokenizer.json");

        let quant_label = manifest
            .get("quant")
            .and_then(|v| v.as_str())
            .unwrap_or("fp16")
            .to_string();
        let quant = match quant_label.as_str() {
            "q4-mix" => WeightQuant::Q4Mix,
            "q4" => WeightQuant::Q4,
            "fp32" => WeightQuant::Fp32,
            _ => WeightQuant::Fp16,
        };

        let cfg: Qwen3VlConfig = if let Some(c) = manifest.get("config") {
            serde_json::from_value(c.clone()).unwrap_or_else(|_| Qwen3VlConfig::qwen3_vl_2b())
        } else {
            Qwen3VlConfig::qwen3_vl_2b()
        };

        // Acquire GPU before pulling ~1 GiB of shards so we fail fast.
        web_sys::console::log_1(&"initializing WebGPU…".into());
        let device = resolve_device_async()
            .await
            .map_err(|e| JsValue::from_str(&e))?;

        let known_total: Option<usize> = shard_files.iter().try_fold(0usize, |acc, (_, n)| {
            Some(acc.checked_add((*n)?)?)
        });
        let mut weight_bytes = match known_total {
            Some(n) if n > (isize::MAX as usize) => {
                return Err(JsValue::from_str(&format!(
                    "weights too large for wasm32 ({n} bytes)"
                )));
            }
            Some(n) => Vec::with_capacity(n),
            None => Vec::new(),
        };

        let n_shards = shard_files.len();
        web_sys::console::log_1(
            &format!("fetching {n_shards} weight shard(s) + tokenizer in parallel…").into(),
        );
        let tok_url = format!("{base}{tok_name}");
        let shard_futs = shard_files.iter().enumerate().map(|(i, (shard, _))| {
            let weight_url = format!("{base}{shard}");
            async move {
                let part = fetch_bytes(&weight_url, "qburn-weights").await?;
                web_sys::console::log_1(
                    &format!("shard {}/{n_shards} ready ({})", i + 1, weight_url).into(),
                );
                Ok::<_, JsValue>((i, part))
            }
        });
        let (tok_bytes, shard_parts) = try_join(
            fetch_bytes(&tok_url, "qburn-weights"),
            try_join_all(shard_futs),
        )
        .await?;
        let mut ordered = shard_parts;
        ordered.sort_by_key(|(i, _)| *i);
        for (_, part) in ordered {
            append_shard(&mut weight_bytes, &part)?;
        }

        web_sys::console::log_1(
            &format!(
                "loading burnpack ({} bytes, quant={:?}, fusion=off)…",
                weight_bytes.len(),
                quant
            )
            .into(),
        );
        let model = load_from_burnpack_bytes(
            bytes_from_vec(weight_bytes),
            cfg.clone(),
            &device,
            quant,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let tokenizer =
            QburnTokenizer::from_bytes(&tok_bytes).map_err(|e| JsValue::from_str(&e))?;
        let mut processor = ImageProcessor::from_config(&cfg.vision_config);
        processor.max_pixels = WEB_MAX_PIXELS;
        web_sys::console::log_1(
            &format!("image max_pixels={WEB_MAX_PIXELS} (web demo cap)").into(),
        );

        APP.with(|slot| {
            *slot.borrow_mut() = Some(AppState {
                model,
                tokenizer,
                processor,
                device,
                quant_label,
            });
        });
        Ok(QburnApp)
    }

    /// Describe an uploaded image; streams pieces via `on_token`.
    ///
    /// `temperature <= 0` is greedy argmax; `> 0` samples with softmax.
    /// Set `probe` to dump prefill top-k (extra GPU readback — off by default).
    #[wasm_bindgen]
    pub async fn describe(
        &self,
        image_bytes: Vec<u8>,
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        probe: bool,
        on_token: js_sys::Function,
    ) -> Result<String, JsValue> {
        // Pull owned state out of the thread-local briefly so we can `.await`
        // without holding a RefCell borrow across the yield.
        let (model, tokenizer, processor, device, quant_label) = APP.with(|slot| {
            let mut slot = slot.borrow_mut();
            let app = slot
                .take()
                .ok_or_else(|| JsValue::from_str("call init() first"))?;
            Ok::<_, JsValue>((
                app.model,
                app.tokenizer,
                app.processor,
                app.device,
                app.quant_label,
            ))
        })?;

        let result = async {
            let img = processor
                .preprocess_bytes(&image_bytes)
                .map_err(|e| JsValue::from_str(&e))?;

            let prompt = qburn::with_length_budget(&prompt, max_tokens);
            let chat = tokenizer.apply_vl_chat_template(
                &prompt,
                img.num_merge_tokens,
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
            );
            let ids = tokenizer
                .encode(&chat)
                .map_err(|e| JsValue::from_str(&e))?;

            web_sys::console::log_1(
                &format!(
                    "describe: grid={:?} merge_tokens={} prompt_tokens={}",
                    img.grid_thw,
                    img.num_merge_tokens,
                    ids.len()
                )
                .into(),
            );

            let mut out = String::new();
            let mut cb = |tid: u32| {
                if let Ok(piece) = tokenizer.decode(&[tid]) {
                    out.push_str(&piece);
                    let _ = on_token.call1(&JsValue::NULL, &JsValue::from_str(&piece));
                }
            };
            let mut probe_buf = PrefillProbe::default();
            let probe_slot = probe.then_some(&mut probe_buf);
            let _tokens = model
                .describe_greedy(
                    &ids,
                    &img,
                    max_tokens,
                    temperature,
                    &device,
                    Some(&mut cb),
                    None,
                    probe_slot,
                )
                .await;

            if probe {
                let greedy_piece = tokenizer
                    .decode(&[probe_buf.greedy_id])
                    .unwrap_or_else(|_| "?".into());
                let topk_fmt: Vec<String> = probe_buf
                    .topk
                    .iter()
                    .map(|(id, logit)| {
                        let piece = tokenizer.decode(&[*id]).unwrap_or_else(|_| "?".into());
                        format!("{id}:{logit:.4}:{piece:?}")
                    })
                    .collect();
                web_sys::console::log_1(
                    &format!(
                        "prefill probe: backend=webgpu greedy_id={} greedy={:?} topk=[{}]",
                        probe_buf.greedy_id,
                        greedy_piece,
                        topk_fmt.join(", ")
                    )
                    .into(),
                );
            }

            Ok::<_, JsValue>(out)
        }
        .await;

        APP.with(|slot| {
            *slot.borrow_mut() = Some(AppState {
                model,
                tokenizer,
                processor,
                device,
                quant_label,
            });
        });
        result
    }

    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        "webgpu".into()
    }

    /// Weight quant label from the loaded manifest (`q4-mix`, `q4`, …).
    #[wasm_bindgen(js_name = quantName)]
    pub fn quant_name(&self) -> String {
        APP.with(|slot| {
            slot.borrow()
                .as_ref()
                .map(|a| a.quant_label.clone())
                .unwrap_or_else(|| "unknown".into())
        })
    }
}
