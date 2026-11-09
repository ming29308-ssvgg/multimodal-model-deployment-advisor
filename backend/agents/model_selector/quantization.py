from .state import ExecutionVariant, ModelSpec


def _variant_from_baseline(
    base: ExecutionVariant,
    *,
    variant_id: str,
    label: str,
    mode: str,
    quantization: str,
    vram_factor: float,
    ram_factor: float,
    quality_rank: int,
    backend: str,
    confidence: str = "heuristic",
    original_vram_gb: float | None = None,
    original_ram_gb: float | None = None,
) -> ExecutionVariant:
    estimated_vram = round(base.estimated_vram_gb * vram_factor, 2)
    estimated_ram = round(max(base.min_ram_gb * ram_factor, 1), 2)
    reference_vram = (
        original_vram_gb
        if original_vram_gb is not None
        else base.estimated_vram_gb
    )
    reference_ram = (
        original_ram_gb
        if original_ram_gb is not None
        else base.min_ram_gb
    )
    vram_reduction = (
        round((1 - estimated_vram / reference_vram) * 100)
        if reference_vram > 0 and estimated_vram < reference_vram
        else 0
    )
    return ExecutionVariant(
        id=variant_id,
        label=label,
        mode=mode,
        quantization=quantization,
        estimated_vram_gb=estimated_vram,
        min_ram_gb=estimated_ram,
        quality_rank=quality_rank,
        confidence=confidence,
        settings={
            **base.settings,
            "original_vram_gb": round(reference_vram, 2),
            "original_ram_gb": round(reference_ram, 2),
            "vram_reduction_percent": vram_reduction,
            "quantization_backend": backend,
            "reference_quantization": base.quantization,
            "estimate_method": "scaled_from_catalog_baseline",
        },
    )


def _gguf_variants(base: ExecutionVariant) -> list[ExecutionVariant]:
    fp16_vram = round(base.estimated_vram_gb * 2.8, 2)
    fp16_ram = round(base.min_ram_gb * 2.4, 2)
    specifications = [
        ("inferred-gguf-fp16", "GGUF FP16", "direct", "FP16", 2.8, 2.4, 98, "heuristic"),
        ("inferred-gguf-q8", "GGUF Q8", "quantized", "Q8_0", 1.65, 1.5, 92, "heuristic"),
        ("inferred-gguf-q6", "GGUF Q6", "quantized", "Q6_K", 1.30, 1.22, 86, "heuristic"),
        ("inferred-gguf-q5", "GGUF Q5", "quantized", "Q5_K_M", 1.15, 1.10, 80, "heuristic"),
        ("catalog", "目录 Q4 基准", "quantized", "Q4_K_M", 1.0, 1.0, 74, "catalog"),
        ("inferred-gguf-q3", "GGUF Q3", "quantized", "Q3_K_M", 0.85, 0.75, 62, "heuristic"),
    ]
    return [
        _variant_from_baseline(
            base,
            variant_id=variant_id,
            label=label,
            mode=mode,
            quantization=quantization,
            vram_factor=vram_factor,
            ram_factor=ram_factor,
            quality_rank=quality_rank,
            backend="GGUF/llama.cpp",
            confidence=confidence,
            original_vram_gb=fp16_vram,
            original_ram_gb=fp16_ram,
        )
        for (
            variant_id,
            label,
            mode,
            quantization,
            vram_factor,
            ram_factor,
            quality_rank,
            confidence,
        ) in specifications
    ]


def _transformers_int4_variants(base: ExecutionVariant) -> list[ExecutionVariant]:
    fp16_vram = round(base.estimated_vram_gb * 2.6, 2)
    fp16_ram = round(base.min_ram_gb * 2.2, 2)
    specifications = [
        (
            "inferred-transformers-fp16",
            "Transformers FP16",
            "direct",
            "FP16",
            2.6,
            2.2,
            96,
            "heuristic",
        ),
        (
            "inferred-transformers-int8",
            "Transformers INT8",
            "quantized",
            "INT8 (bitsandbytes)",
            1.55,
            1.4,
            82,
            "heuristic",
        ),
        (
            "catalog",
            "目录 INT4 基准",
            "quantized",
            "INT4 (bitsandbytes)",
            1.0,
            1.0,
            66,
            "catalog",
        ),
    ]
    return [
        _variant_from_baseline(
            base,
            variant_id=variant_id,
            label=label,
            mode=mode,
            quantization=quantization,
            vram_factor=vram_factor,
            ram_factor=ram_factor,
            quality_rank=quality_rank,
            backend="bitsandbytes",
            confidence=confidence,
            original_vram_gb=fp16_vram,
            original_ram_gb=fp16_ram,
        )
        for (
            variant_id,
            label,
            mode,
            quantization,
            vram_factor,
            ram_factor,
            quality_rank,
            confidence,
        ) in specifications
    ]


def expand_catalog_quantization(
    model: ModelSpec,
    base: ExecutionVariant,
) -> list[ExecutionVariant]:
    """Expand a single catalog baseline into comparable precision levels."""
    normalized = (base.quantization or "").strip().upper()
    runtimes = set(model.runtimes)
    if normalized.startswith("Q4") and runtimes & {"ollama", "llama_cpp"}:
        return _gguf_variants(base)
    if normalized in {"INT4", "4BIT", "NF4"} and "transformers" in runtimes:
        return _transformers_int4_variants(base)
    return [base]


def _scaled_variant(
    base: ExecutionVariant,
    *,
    variant_id: str,
    label: str,
    quantization: str,
    vram_factor: float,
    ram_factor: float,
    quality_rank: int,
    backend: str,
    settings: dict[str, object] | None = None,
) -> ExecutionVariant:
    return ExecutionVariant(
        id=variant_id,
        label=label,
        mode="quantized",
        quantization=quantization,
        estimated_vram_gb=round(base.estimated_vram_gb * vram_factor, 2),
        min_ram_gb=round(max(base.min_ram_gb * ram_factor, 1), 2),
        quality_rank=quality_rank,
        confidence="heuristic",
        settings={
            **base.settings,
            "original_vram_gb": base.estimated_vram_gb,
            "original_ram_gb": base.min_ram_gb,
            "vram_reduction_percent": round((1 - vram_factor) * 100),
            "ram_reduction_percent": round((1 - ram_factor) * 100),
            "quantization_backend": backend,
            **(settings or {}),
        },
    )


def _base_with_strategy(
    base: ExecutionVariant,
    *,
    quantization: str,
    settings: dict[str, object] | None = None,
) -> ExecutionVariant:
    return base.model_copy(
        update={
            "label": f"{base.label} · {quantization}",
            "quantization": quantization,
            "settings": {
                **base.settings,
                "original_vram_gb": base.estimated_vram_gb,
                "original_ram_gb": base.min_ram_gb,
                **(settings or {}),
            },
        }
    )


def infer_execution_variants(
    model: ModelSpec,
    base: ExecutionVariant,
) -> list[ExecutionVariant]:
    """Generate conservative family-specific strategies when the catalog is silent."""
    if model.kind == "tool":
        return [
            _base_with_strategy(
                base,
                quantization="无需量化",
                settings={"strategy": "tool"},
            )
        ]

    runtimes = set(model.runtimes)
    capabilities = set(model.capabilities)

    if "faster_whisper" in runtimes:
        fp16 = _base_with_strategy(
            base,
            quantization="FP16",
            settings={"compute_type": "float16"},
        )
        return [
            fp16,
            _scaled_variant(
                base,
                variant_id="inferred-ctranslate2-int8-fp16",
                label="INT8 权重 + FP16 计算",
                quantization="INT8-FP16 (CTranslate2)",
                vram_factor=0.65,
                ram_factor=0.82,
                quality_rank=76,
                backend="CTranslate2",
                settings={"compute_type": "int8_float16"},
            ),
            _scaled_variant(
                base,
                variant_id="inferred-ctranslate2-int8",
                label="INT8 计算",
                quantization="INT8 (CTranslate2)",
                vram_factor=0.52,
                ram_factor=0.70,
                quality_rank=66,
                backend="CTranslate2",
                settings={"compute_type": "int8"},
            ),
        ]

    if "diffusers" in runtimes:
        fp16 = _base_with_strategy(
            base,
            quantization="FP16",
            settings={
                "torch_dtype": "float16",
                "verify_component_support": False,
            },
        )
        return [
            fp16,
            _scaled_variant(
                base,
                variant_id="inferred-diffusers-int8",
                label="Diffusers INT8",
                quantization="INT8 (bitsandbytes)",
                vram_factor=0.70,
                ram_factor=0.85,
                quality_rank=74,
                backend="bitsandbytes",
                settings={"verify_component_support": True},
            ),
            _scaled_variant(
                base,
                variant_id="inferred-diffusers-int4",
                label="Diffusers INT4",
                quantization="INT4 (bitsandbytes)",
                vram_factor=0.52,
                ram_factor=0.70,
                quality_rank=62,
                backend="bitsandbytes",
                settings={"verify_component_support": True},
            ),
        ]

    transformer_capabilities = {
        "text_generation",
        "summarization",
        "translation",
        "code_generation",
        "image_understanding",
        "video_understanding",
        "audio_understanding",
        "ocr",
    }
    if "transformers" in runtimes and capabilities & transformer_capabilities:
        fp16 = _base_with_strategy(
            base,
            quantization="FP16",
            settings={"torch_dtype": "float16"},
        )
        return [
            fp16,
            _scaled_variant(
                base,
                variant_id="inferred-transformers-int8",
                label="Transformers INT8",
                quantization="INT8 (bitsandbytes)",
                vram_factor=0.72,
                ram_factor=0.85,
                quality_rank=76,
                backend="bitsandbytes",
            ),
            _scaled_variant(
                base,
                variant_id="inferred-transformers-int4",
                label="Transformers INT4",
                quantization="INT4 (bitsandbytes)",
                vram_factor=0.52,
                ram_factor=0.70,
                quality_rank=64,
                backend="bitsandbytes",
            ),
        ]

    if base.estimated_vram_gb > 0:
        return [
            _base_with_strategy(
                base,
                quantization="FP16（未启用额外量化）",
                settings={"strategy": "native_precision"},
            )
        ]
    return [
        _base_with_strategy(
            base,
            quantization="无需量化（CPU）",
            settings={"strategy": "cpu_native"},
        )
    ]
