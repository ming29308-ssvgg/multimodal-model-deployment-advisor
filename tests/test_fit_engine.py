from backend.agents.model_selector.fit_engine import (
    build_execution_variants,
    evaluate_variant,
    select_execution_variant,
)
from backend.agents.model_selector.state import (
    ExecutionVariant,
    ExecutionVariantSpec,
    GPUDevice,
    HardwareInventory,
    ModelSpec,
)


def _inventory(
    *,
    free_vram: float,
    total_vram: float = 24,
    available_ram: float = 32,
    total_ram: float = 64,
) -> HardwareInventory:
    gpus = []
    if total_vram > 0:
        gpus = [
            GPUDevice(
                name="Test GPU",
                total_vram_gb=total_vram,
                free_vram_gb=free_vram,
            )
        ]
    return HardwareInventory(
        os_name="Linux",
        os_version="test",
        architecture="x86_64",
        cpu_model="Test CPU",
        physical_cores=8,
        logical_cores=16,
        ram_total_gb=total_ram,
        ram_available_gb=available_ram,
        disk_free_gb=500,
        gpus=gpus,
        runtimes={"transformers": True},
        environment_type="single_server",
    )


def _model(**updates) -> ModelSpec:
    values = {
        "model_id": "test/model",
        "display_name": "Test Model",
        "description": "test",
        "capabilities": ["text_generation"],
        "modalities": ["text"],
        "runtimes": ["transformers"],
        "min_vram_gb": 1,
        "recommended_vram_gb": 6,
        "estimated_vram_gb": 6,
        "min_ram_gb": 8,
        "parameter_size": "7B",
        "requires_gpu": True,
        "supports_cpu": True,
    }
    values.update(updates)
    return ModelSpec(**values)


def test_fit_boundaries_use_available_resource_utilization():
    model = _model()

    ideal = evaluate_variant(
        model,
        _inventory(free_vram=10, available_ram=20),
        ExecutionVariant(
            id="ideal",
            label="Ideal",
            mode="direct",
            estimated_vram_gb=6,
            min_ram_gb=8,
        ),
    )
    runnable = evaluate_variant(
        model,
        _inventory(free_vram=10, available_ram=20),
        ExecutionVariant(
            id="runnable",
            label="Runnable",
            mode="direct",
            estimated_vram_gb=8.5,
            min_ram_gb=8,
        ),
    )
    constrained = evaluate_variant(
        model,
        _inventory(free_vram=10, available_ram=20),
        ExecutionVariant(
            id="constrained",
            label="Constrained",
            mode="direct",
            estimated_vram_gb=9.8,
            min_ram_gb=8,
        ),
    )
    unsupported = evaluate_variant(
        model,
        _inventory(free_vram=10, available_ram=20),
        ExecutionVariant(
            id="unsupported",
            label="Unsupported",
            mode="direct",
            estimated_vram_gb=9.9,
            min_ram_gb=8,
        ),
    )

    assert ideal.fit_tier == "ideal"
    assert runnable.fit_tier == "runnable"
    assert constrained.fit_tier == "constrained"
    assert unsupported.fit_tier == "unsupported"


def test_zero_available_gpu_resource_is_unsupported_without_division_error():
    estimate = evaluate_variant(
        _model(),
        _inventory(free_vram=0),
        ExecutionVariant(
            id="direct",
            label="Direct",
            mode="direct",
            estimated_vram_gb=4,
            min_ram_gb=8,
        ),
    )

    assert estimate.fit_tier == "unsupported"
    assert estimate.utilization.vram_ratio is None
    assert estimate.utilization.peak_ratio is None


def test_selects_highest_quality_quantized_variant_that_fits():
    model = _model(
        execution_variants=[
            ExecutionVariantSpec(
                id="q8",
                label="Q8",
                mode="quantized",
                quantization="Q8_0",
                estimated_vram_gb=8,
                min_ram_gb=12,
                quality_rank=90,
            ),
            ExecutionVariantSpec(
                id="q4",
                label="Q4",
                mode="quantized",
                quantization="Q4_K_M",
                estimated_vram_gb=5,
                min_ram_gb=10,
                quality_rank=70,
            ),
        ]
    )

    estimate = select_execution_variant(
        model,
        _inventory(free_vram=6.5, available_ram=16),
        allow_offload=False,
    )

    assert estimate.variant.id == "q4"
    assert estimate.variant.quantization == "Q4_K_M"
    assert estimate.fit_tier == "runnable"
    assert estimate.confidence == "catalog"


def test_cpu_offload_variant_is_selected_when_direct_variant_does_not_fit():
    estimate = select_execution_variant(
        _model(estimated_vram_gb=10, recommended_vram_gb=10),
        _inventory(free_vram=6, available_ram=24),
        allow_offload=True,
    )

    assert estimate.variant.mode == "cpu_offload"
    assert estimate.fit_tier == "constrained"
    assert estimate.variant.estimated_vram_gb < 10
    assert estimate.variant.min_ram_gb > 8
    assert estimate.confidence == "heuristic"


def test_diffusers_model_selects_inferred_int4_when_fp16_does_not_fit():
    model = _model(
        runtimes=["diffusers"],
        capabilities=["video_generation"],
        modalities=["video"],
        quantization=None,
        estimated_vram_gb=12,
        recommended_vram_gb=12,
        min_ram_gb=16,
        supports_cpu=False,
    )

    estimate = select_execution_variant(
        model,
        _inventory(
            free_vram=7.5,
            total_vram=8,
            available_ram=16,
            total_ram=24,
        ),
        allow_offload=False,
    )

    assert estimate.variant.quantization == "INT4 (bitsandbytes)"
    assert estimate.variant.mode == "quantized"
    assert estimate.variant.estimated_vram_gb == 6.24
    assert estimate.fit_tier == "runnable"
    assert estimate.confidence == "heuristic"
    assert estimate.variant.settings["original_vram_gb"] == 12
    assert estimate.variant.settings["quantization_backend"] == "bitsandbytes"


def test_diffusers_model_keeps_fp16_when_it_already_fits():
    model = _model(
        runtimes=["diffusers"],
        capabilities=["image_generation"],
        modalities=["image"],
        quantization=None,
        estimated_vram_gb=8,
        recommended_vram_gb=8,
        min_ram_gb=12,
        supports_cpu=False,
    )

    estimate = select_execution_variant(
        model,
        _inventory(
            free_vram=10.5,
            total_vram=12,
            available_ram=18,
            total_ram=24,
        ),
        allow_offload=False,
    )

    assert estimate.variant.quantization == "FP16"
    assert estimate.variant.mode == "direct"
    assert estimate.fit_tier == "runnable"


def test_catalog_execution_variants_prevent_inferred_quantization_variants():
    model = _model(
        quantization=None,
        execution_variants=[
            ExecutionVariantSpec(
                id="maintainer-int8",
                label="Maintainer INT8",
                mode="quantized",
                quantization="INT8 (vendor)",
                estimated_vram_gb=5,
                min_ram_gb=10,
                quality_rank=75,
            ),
        ],
    )

    variants = build_execution_variants(model, allow_offload=False)

    assert [variant.id for variant in variants] == ["maintainer-int8"]


def test_faster_whisper_uses_compute_type_specific_quantization():
    model = _model(
        runtimes=["faster_whisper"],
        capabilities=["speech_to_text"],
        modalities=["audio"],
        quantization=None,
        estimated_vram_gb=6,
        recommended_vram_gb=6,
        min_ram_gb=8,
        supports_cpu=True,
    )

    variants = build_execution_variants(model, allow_offload=False)

    assert [variant.quantization for variant in variants[:3]] == [
        "FP16",
        "INT8-FP16 (CTranslate2)",
        "INT8 (CTranslate2)",
    ]
    assert variants[1].settings["compute_type"] == "int8_float16"
    assert variants[2].settings["compute_type"] == "int8"


def test_tool_without_quantizer_reports_quantization_is_not_needed():
    model = _model(
        kind="tool",
        runtimes=["ffmpeg"],
        capabilities=["extract_audio"],
        modalities=["audio"],
        quantization=None,
        estimated_vram_gb=0,
        recommended_vram_gb=0,
        min_vram_gb=0,
        min_ram_gb=1,
        requires_gpu=False,
        supports_cpu=True,
    )

    variants = build_execution_variants(model, allow_offload=False)

    assert variants[0].quantization == "无需量化"
    assert variants[0].mode == "direct"


def test_gguf_q4_catalog_baseline_expands_to_multiple_quantization_levels():
    model = _model(
        runtimes=["ollama", "llama_cpp", "transformers"],
        quantization="Q4_K_M",
        estimated_vram_gb=4,
        recommended_vram_gb=4,
        min_ram_gb=8,
    )

    variants = build_execution_variants(model, allow_offload=False)

    assert [variant.quantization for variant in variants] == [
        "FP16",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
    ]
    assert next(
        variant for variant in variants if variant.quantization == "Q4_K_M"
    ).confidence == "catalog"
    assert next(
        variant for variant in variants if variant.quantization == "Q6_K"
    ).confidence == "heuristic"


def test_gguf_strategy_selects_highest_quality_level_that_fits_hardware():
    model = _model(
        runtimes=["ollama", "llama_cpp"],
        quantization="Q4_K_M",
        estimated_vram_gb=4,
        recommended_vram_gb=4,
        min_ram_gb=8,
    )

    estimate = select_execution_variant(
        model,
        _inventory(
            free_vram=6.5,
            total_vram=8,
            available_ram=16,
            total_ram=24,
        ),
        allow_offload=False,
    )

    assert estimate.variant.quantization == "Q6_K"
    assert estimate.variant.estimated_vram_gb == 5.2
    assert estimate.fit_tier == "runnable"


def test_transformers_int4_catalog_baseline_expands_to_int8_and_fp16():
    model = _model(
        runtimes=["transformers"],
        capabilities=["image_understanding"],
        modalities=["image"],
        quantization="int4",
        estimated_vram_gb=6,
        recommended_vram_gb=6,
        min_ram_gb=10,
    )

    variants = build_execution_variants(model, allow_offload=False)

    assert [variant.quantization for variant in variants] == [
        "FP16",
        "INT8 (bitsandbytes)",
        "INT4 (bitsandbytes)",
    ]
    assert variants[-1].confidence == "catalog"


def test_offload_uses_catalog_baseline_and_cpu_only_uses_smallest_variant():
    model = _model(
        runtimes=["ollama", "llama_cpp"],
        quantization="Q4_K_M",
        estimated_vram_gb=4,
        recommended_vram_gb=4,
        min_ram_gb=8,
        requires_gpu=False,
        supports_cpu=True,
    )

    variants = build_execution_variants(model, allow_offload=True)
    offload = next(variant for variant in variants if variant.mode == "cpu_offload")
    cpu_only = next(variant for variant in variants if variant.mode == "cpu_only")

    assert offload.quantization == "Q4_K_M"
    assert offload.settings["source_variant_id"] == "catalog"
    assert cpu_only.quantization == "Q3_K_M"
    assert cpu_only.settings["source_variant_id"] == "inferred-gguf-q3"
