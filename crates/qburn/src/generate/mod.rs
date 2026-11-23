//! Autoregressive generation helpers.

use std::collections::HashSet;

use burn::prelude::*;
use burn::tensor::activation;

use crate::model::{
    build_causal_mask, new_caches, AttentionKvCache, RotaryEmbedding, TextDecoder,
};

/// Soft repetition penalty for greedy multimodal decode (matches tjs demo).
pub const GREEDY_REPETITION_PENALTY: f32 = 1.1;
/// Ban completing an already-seen n-gram (0 = off).
pub const GREEDY_NO_REPEAT_NGRAM: usize = 5;

/// Qwen chat stop ids: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>`.
pub const QWEN_STOP_TOKEN_IDS: [u32; 3] = [151_645, 151_643, 151_644];

/// True when `id` should end generation (EOS / turn boundary).
pub fn is_stop_token(id: u32) -> bool {
    QWEN_STOP_TOKEN_IDS.contains(&id)
}

/// Greedy / sampled generation over text-only prompts.
///
/// `temperature <= 0` selects argmax; `temperature > 0` draws from softmax(logits / T).
pub fn generate_text(
    decoder: &TextDecoder,
    rope: &RotaryEmbedding,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    temperature: f32,
    device: &Device,
) -> Vec<u32> {
    let mut caches = new_caches(decoder, 1, prompt_ids.len() + max_new_tokens + 8, device);
    let mut all_ids = prompt_ids.to_vec();

    let tokens = ids_to_tensor(&all_ids, device);
    let mask = build_causal_mask(all_ids.len(), all_ids.len(), device);
    let logits = decoder.forward(tokens, rope, Some(mask), &mut caches, 0);
    let mut next = sample_next(&logits, temperature);

    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        if next == eos_token_id {
            break;
        }
        generated.push(next);
        all_ids.push(next);
        let start = all_ids.len() - 1;
        let tokens = ids_to_tensor(&[next], device);
        // Decode steps are q_len=1 over the KV cache — already causal.
        let logits = decoder.forward(tokens, rope, None, &mut caches, start);
        next = sample_next(&logits, temperature);
    }
    generated
}

/// Stream callback: receives decoded token id each step.
pub type TokenCallback = Box<dyn FnMut(u32)>;

pub fn generate_text_streaming(
    decoder: &TextDecoder,
    rope: &RotaryEmbedding,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    temperature: f32,
    device: &Device,
    mut on_token: impl FnMut(u32),
) -> Vec<u32> {
    let mut caches = new_caches(decoder, 1, prompt_ids.len() + max_new_tokens + 8, device);
    let mut all_ids = prompt_ids.to_vec();

    let tokens = ids_to_tensor(&all_ids, device);
    let mask = build_causal_mask(all_ids.len(), all_ids.len(), device);
    let logits = decoder.forward(tokens, rope, Some(mask), &mut caches, 0);
    let mut next = sample_next(&logits, temperature);

    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        if next == eos_token_id {
            break;
        }
        on_token(next);
        generated.push(next);
        all_ids.push(next);
        let start = all_ids.len() - 1;
        let tokens = ids_to_tensor(&[next], device);
        // Decode steps are q_len=1 over the KV cache — already causal.
        let logits = decoder.forward(tokens, rope, None, &mut caches, start);
        next = sample_next(&logits, temperature);
    }
    generated
}

/// Backward-compatible alias — greedy (`temperature = 0`).
pub fn generate_greedy_text(
    decoder: &TextDecoder,
    rope: &RotaryEmbedding,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    device: &Device,
) -> Vec<u32> {
    generate_text(
        decoder,
        rope,
        prompt_ids,
        max_new_tokens,
        eos_token_id,
        0.0,
        device,
    )
}

/// Backward-compatible alias — greedy streaming.
pub fn generate_greedy_text_streaming(
    decoder: &TextDecoder,
    rope: &RotaryEmbedding,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_token_id: u32,
    device: &Device,
    on_token: impl FnMut(u32),
) -> Vec<u32> {
    generate_text_streaming(
        decoder,
        rope,
        prompt_ids,
        max_new_tokens,
        eos_token_id,
        0.0,
        device,
        on_token,
    )
}

pub fn ids_to_tensor(ids: &[u32], device: &Device) -> Tensor<2, Int> {
    let data: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
    Tensor::<1, Int>::from_ints(data.as_slice(), device).reshape([1, ids.len()])
}

fn last_logits_row(logits: &Tensor<3>) -> Tensor<2> {
    let [_, seq, vocab] = logits.dims();
    // Slice keeps batch=1 and seq=1 → [1, 1, vocab]; squeeze only the seq axis → [1, vocab].
    logits
        .clone()
        .slice(s![0..1, seq - 1..seq, 0..vocab])
        .squeeze_dim::<2>(1)
}

/// Top-k entries from the last sequence position (id, logit), highest first.
pub async fn topk_last_async(logits: &Tensor<3>, k: usize) -> Vec<(u32, f32)> {
    let last = last_logits_row(logits);
    let data = last
        .into_data_async()
        .await
        .unwrap_or_else(|_| burn::tensor::TensorData::new(vec![0.0f32], [1]));
    let row: Vec<f32> = data.try_to_vec().unwrap_or_else(|_| vec![0.0]);
    topk_cpu(&row, k)
}

/// Sync top-k wrapper for native backends.
pub fn topk_last(logits: &Tensor<3>, k: usize) -> Vec<(u32, f32)> {
    #[cfg(target_family = "wasm")]
    {
        let _ = (logits, k);
        panic!("topk_last is sync-only; call topk_last_async on wasm");
    }
    #[cfg(not(target_family = "wasm"))]
    {
        pollster::block_on(topk_last_async(logits, k))
    }
}

fn topk_cpu(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .into_iter()
        .map(|(i, v)| (i as u32, v))
        .collect()
}

/// Async next-token selection — required on wasm (no sync GPU readback).
pub async fn sample_next_async(logits: &Tensor<3>, temperature: f32) -> u32 {
    sample_next_guarded_async(logits, temperature, &[], 1.0, 0).await
}

/// Like [`sample_next_async`], with optional greedy anti-loop guards.
///
/// When `temperature <= 0`, applies HF-style repetition penalty and
/// `no_repeat_ngram_size` bans on the CPU logits row before argmax. Sampling
/// (`temperature > 0`) leaves logits untouched (penalty/ngram ignored).
pub async fn sample_next_guarded_async(
    logits: &Tensor<3>,
    temperature: f32,
    generated: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
) -> u32 {
    let last = last_logits_row(logits);
    if temperature <= 0.0 {
        let need_guard = (repetition_penalty - 1.0).abs() > f32::EPSILON
            || no_repeat_ngram_size > 0;
        if !need_guard {
            let idx = last.argmax(1);
            let data = idx
                .into_data_async()
                .await
                .unwrap_or_else(|_| burn::tensor::TensorData::new(vec![0i32], [1]));
            let v: Vec<i32> = data.try_to_vec().unwrap_or_else(|_| vec![0]);
            return v[0] as u32;
        }
        let data = last
            .into_data_async()
            .await
            .unwrap_or_else(|_| burn::tensor::TensorData::new(vec![0.0f32], [1]));
        let mut row: Vec<f32> = data.try_to_vec().unwrap_or_else(|_| vec![0.0]);
        apply_repetition_penalty(&mut row, generated, repetition_penalty);
        ban_no_repeat_ngram(&mut row, generated, no_repeat_ngram_size);
        return argmax_cpu(&row);
    }

    let scaled = last / temperature;
    let probs = activation::softmax(scaled, 1);
    let data = probs
        .into_data_async()
        .await
        .unwrap_or_else(|_| burn::tensor::TensorData::new(vec![1.0f32], [1]));
    let probs: Vec<f32> = data.try_to_vec().unwrap_or_else(|_| vec![1.0]);
    multinomial(&probs)
}

/// HF repetition penalty on a logits row (unique prior tokens).
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() <= f32::EPSILON || generated.is_empty() {
        return;
    }
    let mut seen = HashSet::with_capacity(generated.len());
    for &tok in generated {
        if !seen.insert(tok) {
            continue;
        }
        let i = tok as usize;
        if i >= logits.len() {
            continue;
        }
        if logits[i] < 0.0 {
            logits[i] *= penalty;
        } else {
            logits[i] /= penalty;
        }
    }
}

/// Set logits of tokens that would complete a repeated `n`-gram to `-inf`.
pub fn ban_no_repeat_ngram(logits: &mut [f32], generated: &[u32], n: usize) {
    for tok in no_repeat_ngram_banned(generated, n) {
        let i = tok as usize;
        if i < logits.len() {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// Tokens that would complete an already-seen `n`-gram in `generated`.
pub fn no_repeat_ngram_banned(generated: &[u32], n: usize) -> Vec<u32> {
    if n <= 1 || generated.len() < n - 1 {
        return Vec::new();
    }
    let prefix_len = n - 1;
    let prefix = &generated[generated.len() - prefix_len..];
    let mut banned = Vec::new();
    let last_start = generated.len().saturating_sub(prefix_len);
    for i in 0..last_start {
        if &generated[i..i + prefix_len] == prefix {
            banned.push(generated[i + prefix_len]);
        }
    }
    banned
}

fn argmax_cpu(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Sync wrapper for native backends (`pollster`). Panics on wasm — use [`sample_next_async`].
pub fn sample_next(logits: &Tensor<3>, temperature: f32) -> u32 {
    #[cfg(target_family = "wasm")]
    {
        let _ = (logits, temperature);
        panic!("sample_next is sync-only; call sample_next_async on wasm");
    }
    #[cfg(not(target_family = "wasm"))]
    {
        pollster::block_on(sample_next_async(logits, temperature))
    }
}

/// Async argmax of the last sequence position.
pub async fn argmax_last_async(logits: &Tensor<3>) -> u32 {
    sample_next_async(logits, 0.0).await
}

/// Sync argmax wrapper.
pub fn argmax_last(logits: &Tensor<3>) -> u32 {
    sample_next(logits, 0.0)
}

fn multinomial(probs: &[f32]) -> u32 {
    let mut u = [0u8; 8];
    if getrandom::fill(&mut u).is_err() {
        return probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    // Interpret bytes as an integer in [0, 2^64), not as IEEE-754 bits.
    // `f64::from_le_bytes` yields negatives/NaNs ~half the time and collapses
    // sampling onto vocab id 0 ("!" in Qwen BPE).
    let r = u64::from_le_bytes(u) as f64 / (u64::MAX as f64 + 1.0);
    let mut cum = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cum += p as f64;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

/// Reset caches helper.
pub fn reset_caches(caches: &mut [AttentionKvCache]) {
    for c in caches.iter_mut() {
        c.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multinomial_degenerate() {
        assert_eq!(multinomial(&[1.0]), 0);
        let p = multinomial(&[0.0, 0.0, 1.0]);
        assert_eq!(p, 2);
    }

    #[test]
    fn multinomial_unit_interval_avoids_token_zero_bias() {
        // With mass only on the last bin, samples must never land on index 0.
        // The old `f64::from_le_bytes` RNG produced negatives ~50% of the time,
        // which always selected token 0 ("!" in Qwen BPE).
        let probs = [0.0f32, 0.0, 0.0, 1.0];
        for _ in 0..64 {
            assert_eq!(multinomial(&probs), 3);
        }
    }

    #[test]
    fn topk_cpu_orders_highest_first() {
        let logits = [0.1f32, 5.0, 2.0, 4.0];
        let top = topk_cpu(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], (1, 5.0));
        assert_eq!(top[1], (3, 4.0));
        assert_eq!(top[2], (2, 2.0));
    }

    #[test]
    fn no_repeat_ngram_bans_continuations() {
        // After [a,b,c,d,e,a,b,c,d], n=5 bans e (would repeat a,b,c,d,e).
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3, 4];
        let banned = no_repeat_ngram_banned(&tokens, 5);
        assert!(banned.contains(&5));
    }

    #[test]
    fn repetition_penalty_downweights_seen() {
        let mut logits = vec![2.0f32, 3.0, 1.0];
        apply_repetition_penalty(&mut logits, &[1], 1.1);
        assert!((logits[1] - 3.0 / 1.1).abs() < 1e-5);
        assert_eq!(logits[0], 2.0);
    }

    #[test]
    fn stop_token_set_covers_qwen_chat() {
        assert!(is_stop_token(151_645));
        assert!(is_stop_token(151_643));
        assert!(is_stop_token(151_644));
        assert!(!is_stop_token(42));
    }
}
