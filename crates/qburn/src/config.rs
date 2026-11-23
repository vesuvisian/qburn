//! Qwen3-VL configuration matching HuggingFace `config.json`.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Text decoder configuration (`text_config`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}

/// Vision encoder configuration (`vision_config`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_depth")]
    pub depth: usize,
    #[serde(default = "default_vision_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_heads")]
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_out_hidden")]
    pub out_hidden_size: usize,
    #[serde(default = "default_num_pos_embed")]
    pub num_position_embeddings: usize,
    #[serde(default = "default_deepstack")]
    pub deepstack_visual_indexes: Vec<usize>,
}

/// M-RoPE scaling block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Top-level Qwen3-VL config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3VlConfig {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    #[serde(default = "default_image_token")]
    pub image_token_id: u32,
    #[serde(default = "default_video_token")]
    pub video_token_id: u32,
    #[serde(default = "default_vision_start")]
    pub vision_start_token_id: u32,
    #[serde(default = "default_vision_end")]
    pub vision_end_token_id: u32,
}

impl Qwen3VlConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Built-in Qwen3-VL-2B-Instruct preset.
    pub fn qwen3_vl_2b() -> Self {
        Self {
            text_config: TextConfig {
                hidden_size: 2048,
                num_hidden_layers: 28,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                intermediate_size: 6144,
                vocab_size: 151_936,
                max_position_embeddings: 262_144,
                rms_norm_eps: 1e-6,
                rope_theta: 5_000_000.0,
                head_dim: 128,
                tie_word_embeddings: true,
                rope_scaling: Some(RopeScaling {
                    mrope_section: Some(vec![24, 20, 20]),
                    mrope_interleaved: Some(true),
                    rope_type: Some("default".into()),
                }),
                bos_token_id: 151_643,
                eos_token_id: 151_645,
            },
            vision_config: VisionConfig {
                depth: 24,
                hidden_size: 1024,
                intermediate_size: 4096,
                num_heads: 16,
                in_channels: 3,
                patch_size: 16,
                spatial_merge_size: 2,
                temporal_patch_size: 2,
                out_hidden_size: 2048,
                num_position_embeddings: 2304,
                deepstack_visual_indexes: vec![5, 11, 17],
            },
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
        }
    }

    pub fn mrope_section(&self) -> [usize; 3] {
        self.text_config
            .rope_scaling
            .as_ref()
            .and_then(|r| r.mrope_section.as_ref())
            .filter(|s| s.len() == 3)
            .map(|s| [s[0], s[1], s[2]])
            .unwrap_or([24, 20, 20])
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn patch_embed_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn merged_hidden_size(&self) -> usize {
        self.hidden_size * self.spatial_merge_size * self.spatial_merge_size
    }
}

fn default_max_pos() -> usize {
    262_144
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    5_000_000.0
}
fn default_head_dim() -> usize {
    128
}
fn default_bos() -> u32 {
    151_643
}
fn default_eos() -> u32 {
    151_645
}
fn default_vision_depth() -> usize {
    24
}
fn default_vision_hidden() -> usize {
    1024
}
fn default_vision_intermediate() -> usize {
    4096
}
fn default_vision_heads() -> usize {
    16
}
fn default_in_channels() -> usize {
    3
}
fn default_patch_size() -> usize {
    16
}
fn default_spatial_merge() -> usize {
    2
}
fn default_temporal_patch() -> usize {
    2
}
fn default_out_hidden() -> usize {
    2048
}
fn default_num_pos_embed() -> usize {
    2304
}
fn default_deepstack() -> Vec<usize> {
    vec![5, 11, 17]
}
fn default_image_token() -> u32 {
    151_655
}
fn default_video_token() -> u32 {
    151_656
}
fn default_vision_start() -> u32 {
    151_652
}
fn default_vision_end() -> u32 {
    151_653
}
