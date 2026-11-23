use qburn::Qwen3VlConfig;

#[test]
fn config_preset_matches_2b() {
    let cfg = Qwen3VlConfig::qwen3_vl_2b();
    assert_eq!(cfg.text_config.hidden_size, 2048);
    assert_eq!(cfg.text_config.num_hidden_layers, 28);
    assert_eq!(cfg.text_config.num_attention_heads, 16);
    assert_eq!(cfg.text_config.num_key_value_heads, 8);
    assert_eq!(cfg.text_config.head_dim, 128);
    assert_eq!(cfg.text_config.vocab_size, 151_936);
    assert_eq!(cfg.vision_config.depth, 24);
    assert_eq!(cfg.vision_config.hidden_size, 1024);
    assert_eq!(cfg.vision_config.out_hidden_size, 2048);
    assert_eq!(cfg.vision_config.patch_size, 16);
    assert_eq!(cfg.vision_config.spatial_merge_size, 2);
    assert_eq!(cfg.mrope_section(), [24, 20, 20]);
    assert_eq!(cfg.image_token_id, 151_655);
}

#[test]
fn rotary_embedding_constructs() {
    use qburn::model::RotaryEmbedding;
    use qburn::resolve_device;

    let device = resolve_device();
    let cfg = Qwen3VlConfig::qwen3_vl_2b().text_config;
    let _rope = RotaryEmbedding::new(cfg.head_dim, 64, cfg.rope_theta, &device);
}
