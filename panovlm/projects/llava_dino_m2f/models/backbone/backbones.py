from xtuner.model.utils import guess_load_checkpoint


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: float | None = None,
    pos_embed_rope_jitter_coords: float | None = None,
    pos_embed_rope_rescale_coords: float | None = None,
    pos_embed_rope_dtype: str = "fp32",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    check_hash: bool = False,
    weights: str = None,
    **kwargs,
):
    from projects.llava_dino_m2f.models.backbone.vision_transformer import DinoVisionTransformer

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed_rope_base=pos_embed_rope_base,
        pos_embed_rope_min_period=pos_embed_rope_min_period,
        pos_embed_rope_max_period=pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=pos_embed_rope_dtype,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale_init,
        norm_layer=norm_layer,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        state_dict = guess_load_checkpoint(weights)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()
    return model


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: str = None,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if 'eadcf0ff' in weights:
        untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: str = None,
    check_hash: bool = False,
    **kwargs,
):
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.4,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )