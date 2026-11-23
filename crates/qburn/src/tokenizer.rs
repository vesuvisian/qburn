//! HuggingFace `tokenizers` wrapper (fancy-regex for wasm compatibility).

use std::path::Path;

/// Thin wrapper around `tokenizers::Tokenizer` with Qwen chat helpers.
pub struct QburnTokenizer {
    inner: tokenizers::Tokenizer,
}

impl QburnTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("load tokenizer bytes: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    /// Text-only chat template.
    pub fn apply_chat_template(&self, system: &str, user: &str) -> String {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    /// Multimodal chat template with a single image block.
    ///
    /// Inserts `vision_start` + `num_image_tokens` × `image_pad` + `vision_end`.
    pub fn apply_vl_chat_template(
        &self,
        user_text: &str,
        num_image_tokens: usize,
        vision_start: &str,
        image_pad: &str,
        vision_end: &str,
    ) -> String {
        self.apply_vl_chat_template_multi(
            user_text,
            &[num_image_tokens],
            vision_start,
            image_pad,
            vision_end,
        )
    }

    /// Multimodal chat template with one vision block per image.
    ///
    /// For each `n` in `num_image_tokens`, inserts
    /// `vision_start` + `n` × `image_pad` + `vision_end`, then the user text.
    pub fn apply_vl_chat_template_multi(
        &self,
        user_text: &str,
        num_image_tokens: &[usize],
        vision_start: &str,
        image_pad: &str,
        vision_end: &str,
    ) -> String {
        let _ = self;
        format_vl_chat_template_multi(
            user_text,
            num_image_tokens,
            vision_start,
            image_pad,
            vision_end,
        )
    }

    /// Qwen3-VL video chat template: one timestamped vision block per temporal group.
    ///
    /// HF encodes time as text (`<X.X seconds>`) and splits `video_grid_thw` into
    /// `grid_t` spans with `t=1` for M-RoPE — not one contiguous video-pad run.
    pub fn apply_vl_video_chat_template(
        &self,
        user_text: &str,
        tokens_per_temporal_group: usize,
        timestamps_sec: &[f64],
        vision_start: &str,
        video_pad: &str,
        vision_end: &str,
    ) -> String {
        let _ = self;
        format_vl_video_chat_template(
            user_text,
            tokens_per_temporal_group,
            timestamps_sec,
            vision_start,
            video_pad,
            vision_end,
        )
    }
}

fn format_vl_chat_template_multi(
    user_text: &str,
    num_image_tokens: &[usize],
    vision_start: &str,
    image_pad: &str,
    vision_end: &str,
) -> String {
    let mut vision = String::new();
    for &n in num_image_tokens {
        vision.push_str(vision_start);
        vision.push_str(&image_pad.repeat(n));
        vision.push_str(vision_end);
    }
    format!(
        "<|im_start|>user\n{vision}{user_text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn format_vl_video_chat_template(
    user_text: &str,
    tokens_per_temporal_group: usize,
    timestamps_sec: &[f64],
    vision_start: &str,
    video_pad: &str,
    vision_end: &str,
) -> String {
    let mut vision = String::new();
    for &ts in timestamps_sec {
        vision.push_str(&format!("<{ts:.1} seconds>"));
        vision.push_str(vision_start);
        vision.push_str(&video_pad.repeat(tokens_per_temporal_group));
        vision.push_str(vision_end);
    }
    format!(
        "<|im_start|>user\n{vision}{user_text}<|im_end|>\n<|im_start|>assistant\n"
    )
}

#[cfg(test)]
mod tests {
    use super::{format_vl_chat_template_multi, format_vl_video_chat_template};

    #[test]
    fn multi_template_emits_n_blocks() {
        let s = format_vl_chat_template_multi("What changed?", &[2, 3], "<vs>", "<pad>", "<ve>");
        assert_eq!(
            s,
            "<|im_start|>user\n<vs><pad><pad><ve><vs><pad><pad><pad><ve>What changed?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn single_block_counts() {
        let one = format_vl_chat_template_multi("hi", &[4], "S", "P", "E");
        assert!(one.contains("SPPPPE"));
        assert_eq!(one.matches('S').count(), 1);
        assert_eq!(one.matches('E').count(), 1);
    }

    #[test]
    fn video_template_emits_timestamped_blocks() {
        let s = format_vl_video_chat_template(
            "Describe this video.",
            2,
            &[0.0, 0.5],
            "<vs>",
            "<vp>",
            "<ve>",
        );
        assert_eq!(
            s,
            "<|im_start|>user\n<0.0 seconds><vs><vp><vp><ve><0.5 seconds><vs><vp><vp><ve>Describe this video.<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}