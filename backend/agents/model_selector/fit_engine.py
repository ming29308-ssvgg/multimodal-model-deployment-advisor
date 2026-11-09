from .state import (
    ExecutionVariant,
    FitEstimate,
    HardwareInventory,
    ModelSpec,
    ResourceUtilization,
)
from .quantization import expand_catalog_quantization, infer_execution_variants


FIT_ORDER = {
    "ideal": 0,
    "runnable": 1,
    "constrained": 2,
    "unsupported": 3,
}


def _normalized_os(inventory: HardwareInventory) -> str:
    value = inventory.os_name.lower()
    if value.startswith("win"):
        return "windows"
    if value in {"darwin", "macos", "mac"}:
        return "darwin"
    return "linux" if value.startswith("linux") else value


def available_vram(
    inventory: HardwareInventory,
    model: ModelSpec,
) -> tuple[float, float]:
    if not inventory.gpus:
        return 0.0, 0.0
    if model.supports_multi_gpu:
        return (
            sum(gpu.total_vram_gb for gpu in inventory.gpus),
            sum(gpu.free_vram_gb for gpu in inventory.gpus),
        )
    return (
        max(gpu.total_vram_gb for gpu in inventory.gpus),
        max(gpu.free_vram_gb for gpu in inventory.gpus),
    )


def _ratio(required: float, available: float) -> float | None:
    if required <= 0:
        return 0.0
    if available <= 0:
        return None
    return round(required / available, 4)


def _base_variant(model: ModelSpec) -> ExecutionVariant:
    estimated_vram = (
        model.estimated_vram_gb
        if model.estimated_vram_gb is not None
        else model.recommended_vram_gb or model.min_vram_gb
    )
    return ExecutionVariant(
        id="catalog",
        label="目录推荐配置",
        mode="quantized" if model.quantization else "direct",
        quantization=model.quantization,
        estimated_vram_gb=estimated_vram,
        min_ram_gb=model.min_ram_gb,
        quality_rank=80,
        confidence="catalog",
        settings={},
    )


def build_execution_variants(
    model: ModelSpec,
    *,
    allow_offload: bool,
) -> list[ExecutionVariant]:
    if model.execution_variants:
        variants = [
            ExecutionVariant(
                **variant.model_dump(),
                confidence="catalog",
            )
            for variant in model.execution_variants
        ]
    else:
        catalog_variant = _base_variant(model)
        variants = (
            expand_catalog_quantization(model, catalog_variant)
            if model.quantization
            else infer_execution_variants(model, catalog_variant)
        )

    highest_quality = max(
        variants,
        key=lambda item: (item.quality_rank, item.estimated_vram_gb),
    )
    catalog_baseline = next(
        (
            variant
            for variant in variants
            if variant.id == "catalog" and variant.estimated_vram_gb > 0
        ),
        highest_quality,
    )
    if (
        allow_offload
        and model.supports_cpu
        and catalog_baseline.estimated_vram_gb > 0
    ):
        gpu_memory = round(catalog_baseline.estimated_vram_gb * 0.45, 2)
        offloaded_memory = max(
            catalog_baseline.estimated_vram_gb - gpu_memory,
            0,
        )
        variants.append(
            ExecutionVariant(
                id=f"{catalog_baseline.id}-cpu-offload",
                label=f"{catalog_baseline.label} + CPU Offload",
                mode="cpu_offload",
                quantization=catalog_baseline.quantization,
                estimated_vram_gb=gpu_memory,
                min_ram_gb=round(
                    catalog_baseline.min_ram_gb + offloaded_memory * 1.2,
                    2,
                ),
                quality_rank=max(catalog_baseline.quality_rank - 10, 0),
                confidence="heuristic",
                settings={
                    **catalog_baseline.settings,
                    "cpu_offload": True,
                    "source_variant_id": catalog_baseline.id,
                },
            )
        )
    if model.supports_cpu and not model.requires_gpu:
        smallest = min(
            (
                variant
                for variant in variants
                if variant.mode in {"direct", "quantized"}
            ),
            key=lambda item: (
                item.estimated_vram_gb,
                item.min_ram_gb,
                -item.quality_rank,
            ),
        )
        variants.append(
            ExecutionVariant(
                id=f"{smallest.id}-cpu-only",
                label=f"{smallest.label} + CPU",
                mode="cpu_only",
                quantization=smallest.quantization,
                estimated_vram_gb=0,
                min_ram_gb=round(
                    max(
                        smallest.min_ram_gb,
                        smallest.estimated_vram_gb * 1.35,
                    ),
                    2,
                ),
                quality_rank=max(smallest.quality_rank - 20, 0),
                confidence="heuristic",
                settings={
                    **smallest.settings,
                    "device": "cpu",
                    "source_variant_id": smallest.id,
                },
            )
        )
    return variants


def evaluate_variant(
    model: ModelSpec,
    inventory: HardwareInventory,
    variant: ExecutionVariant,
) -> FitEstimate:
    total_vram, free_vram = available_vram(inventory, model)
    utilization = ResourceUtilization(
        required_vram_gb=variant.estimated_vram_gb,
        available_vram_gb=free_vram,
        vram_ratio=_ratio(variant.estimated_vram_gb, free_vram),
        required_ram_gb=variant.min_ram_gb,
        available_ram_gb=inventory.ram_available_gb,
        ram_ratio=_ratio(variant.min_ram_gb, inventory.ram_available_gb),
    )
    ratios = [
        ratio
        for ratio in [utilization.vram_ratio, utilization.ram_ratio]
        if ratio is not None
    ]
    missing_required_resource = (
        variant.estimated_vram_gb > 0 and utilization.vram_ratio is None
    ) or (variant.min_ram_gb > 0 and utilization.ram_ratio is None)
    utilization.peak_ratio = (
        None if missing_required_resource else max(ratios, default=0.0)
    )

    os_supported = _normalized_os(inventory) in model.supported_os
    if model.kind == "tool":
        fit_tier = "ideal"
    elif not os_supported or missing_required_resource:
        fit_tier = "unsupported"
    elif variant.estimated_vram_gb > total_vram and variant.mode != "cpu_only":
        fit_tier = "unsupported"
    elif utilization.peak_ratio is None:
        fit_tier = "unsupported"
    elif utilization.peak_ratio <= 0.60:
        fit_tier = "ideal"
    elif utilization.peak_ratio <= 0.85:
        fit_tier = "runnable"
    elif utilization.peak_ratio <= 0.98:
        fit_tier = "constrained"
    else:
        fit_tier = "unsupported"

    if variant.mode == "cpu_offload" and fit_tier != "unsupported":
        fit_tier = "constrained"

    return FitEstimate(
        variant=variant,
        fit_tier=fit_tier,
        utilization=utilization,
        confidence=variant.confidence,
        basis={
            "source": variant.confidence,
            "os_supported": os_supported,
            "total_vram_gb": total_vram,
            "available_vram_gb": free_vram,
            "available_ram_gb": inventory.ram_available_gb,
            "thresholds": {
                "ideal_max": 0.60,
                "runnable_max": 0.85,
                "constrained_max": 0.98,
            },
            "settings": variant.settings,
        },
    )


def select_execution_variant(
    model: ModelSpec,
    inventory: HardwareInventory,
    *,
    allow_offload: bool,
) -> FitEstimate:
    estimates = [
        evaluate_variant(model, inventory, variant)
        for variant in build_execution_variants(
            model,
            allow_offload=allow_offload,
        )
    ]
    viable = [
        estimate
        for estimate in estimates
        if estimate.fit_tier != "unsupported"
    ]
    if viable:
        return max(
            viable,
            key=lambda item: (
                item.variant.quality_rank,
                -FIT_ORDER[item.fit_tier],
            ),
        )
    return min(
        estimates,
        key=lambda item: (
            (
                item.utilization.peak_ratio
                if item.utilization.peak_ratio is not None
                else float("inf")
            ),
            -item.variant.quality_rank,
        ),
    )
