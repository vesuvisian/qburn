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

    /// Multimodal chat template with image placeholder tokens.
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
        let pads = image_pad.repeat(num_image_tokens);
        format!(
            "<|im_start|>user\n{vision_start}{pads}{vision_end}{user_text}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}
