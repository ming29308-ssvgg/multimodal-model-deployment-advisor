from .catalog import load_catalog
from .fit_engine import available_vram, select_execution_variant
from .scoring import PREFERENCE_WEIGHTS, rank_candidates
from .state import (
    CandidateRecommendation,
    FitEstimate,
    HardwareInventory,
    ModelSpec,
    StageRecommendation,
    TaskAnalysis,
)


def _runtime_for(model: ModelSpec, inventory: HardwareInventory) -> str:
    for runtime in model.runtimes:
        if inventory.runtimes.get(runtime, False):
            return runtime
    return model.runtimes[0]


def _parameter_billions(model: ModelSpec) -> float:
    if not model.parameter_size:
        return 0
    normalized = model.parameter_size.strip().upper()
    try:
        if normalized.endswith("B"):
            return float(normalized[:-1])
        if normalized.endswith("M"):
            return float(normalized[:-1]) / 1000
    except ValueError:
        return 0
    return 0


def _hardware_reasons(
    model: ModelSpec,
    inventory: HardwareInventory,
    estimate: FitEstimate,
    runtime: str,
) -> list[str]:
    reasons: list[str] = []
    total_vram, free_vram = available_vram(inventory, model)
    variant = estimate.variant
    fit = estimate.fit_tier
    original_vram = variant.settings.get("original_vram_gb")
    reduction_percent = variant.settings.get("vram_reduction_percent")

    if (
        variant.mode == "quantized"
        and isinstance(original_vram, (int, float))
        and original_vram > variant.estimated_vram_gb
    ):
        reduction_text = (
            f"，预计降低 {reduction_percent}%"
            if isinstance(reduction_percent, (int, float))
            else ""
        )
        reasons.append(
            f"原始配置约 {original_vram:.1f}GB 显存，采用 "
            f"{variant.quantization} 量化后约 "
            f"{variant.estimated_vram_gb:.1f}GB{reduction_text}。"
        )

    if variant.mode == "cpu_only":
        reasons.append(
            f"选择 CPU-only 方案，需要约 {variant.min_ram_gb:.1f}GB 可用内存。"
        )
    elif variant.mode == "cpu_offload":
        reasons.append(
            f"选择 CPU Offload，将 GPU 需求降至约 "
            f"{variant.estimated_vram_gb:.1f}GB，并使用约 "
            f"{variant.min_ram_gb:.1f}GB 内存。"
        )
    elif total_vram > 0:
        if free_vram >= variant.estimated_vram_gb:
            reasons.append(
                f"当前可用显存 {free_vram:.1f}GB，可覆盖所选方案约 "
                f"{variant.estimated_vram_gb:.1f}GB 的估算需求。"
            )
        else:
            reasons.append(
                f"当前可用显存 {free_vram:.1f}GB，低于所选方案约 "
                f"{variant.estimated_vram_gb:.1f}GB 的估算需求。"
            )
    elif variant.estimated_vram_gb == 0 and model.supports_cpu:
        reasons.append("未检测到可用 GPU；所选方案支持 CPU 推理。")

    if inventory.ram_available_gb >= variant.min_ram_gb:
        reasons.append(
            f"当前可用内存 {inventory.ram_available_gb:.1f}GB，可覆盖约 "
            f"{variant.min_ram_gb:.1f}GB 的估算需求。"
        )
    else:
        reasons.append(
            f"当前可用内存 {inventory.ram_available_gb:.1f}GB，低于约 "
            f"{variant.min_ram_gb:.1f}GB 的估算需求。"
        )

    if inventory.runtimes.get(runtime, False):
        reasons.append(f"已检测到推荐运行时 {runtime}。")
    else:
        reasons.append(f"尚未检测到推荐运行时 {runtime}，部署前需要安装。")

    if fit == "unsupported":
        reasons.append("该候选不适合在当前节点直接部署。")
    return reasons


def _tradeoffs(model: ModelSpec, estimate: FitEstimate) -> list[str]:
    size = _parameter_billions(model)
    fit = estimate.fit_tier
    variant = estimate.variant
    tradeoffs: list[str] = []
    if model.kind == "tool":
        return ["这是预处理工具，不占用模型推理显存。"]
    if size and size <= 3:
        tradeoffs.extend(["推理速度较快、资源余量较大。", "复杂任务质量通常低于更大参数模型。"])
    elif size and size <= 8:
        tradeoffs.extend(["质量与资源消耗较均衡。", "速度通常慢于 3B 级轻量模型。"])
    elif size:
        tradeoffs.extend(["更适合质量优先的复杂任务。", "显存、内存和启动时间开销较高。"])
    else:
        tradeoffs.append("参数规模未标准化，建议部署前用真实样例做基准测试。")
    if variant.mode == "quantized" and variant.quantization:
        reduction_percent = variant.settings.get("vram_reduction_percent")
        if isinstance(reduction_percent, (int, float)):
            tradeoffs.append(
                f"{variant.quantization} 预计降低约 {reduction_percent}% 显存占用，"
                "但可能带来精度、速度或算子兼容性变化。"
            )
        else:
            tradeoffs.append(
                f"{variant.quantization} 可降低资源占用，但可能带来少量精度损失。"
            )
    if variant.mode == "cpu_offload":
        tradeoffs.append("CPU Offload 会降低显存需求，但通常会明显降低生成或推理速度。")
    if fit == "constrained":
        tradeoffs.append("当前资源余量较小，其他 GPU 进程会影响稳定性。")
    elif fit == "unsupported":
        tradeoffs.append("应改用更高配置服务器或远端推理服务。")
    return tradeoffs


def _candidate(
    model: ModelSpec,
    inventory: HardwareInventory,
    allow_offload: bool,
) -> CandidateRecommendation:
    estimate = select_execution_variant(
        model,
        inventory,
        allow_offload=allow_offload,
    )
    fit = estimate.fit_tier
    runtime = _runtime_for(model, inventory)
    reasons = [f"支持任务能力：{', '.join(model.capabilities)}"]
    warnings: list[str] = []

    if inventory.runtimes.get(runtime, False):
        reasons.append(f"已检测到运行时：{runtime}")
    else:
        warnings.append(f"需要安装或配置运行时：{runtime}")

    if estimate.variant.quantization:
        reasons.append(f"建议精度/量化策略：{estimate.variant.quantization}")
    if estimate.variant.settings.get("verify_component_support"):
        warnings.append(
            "该量化方案依赖模型组件兼容性；部署前需确认当前 Diffusers "
            "版本及对应 Transformer 组件支持该量化后端"
        )
    if fit == "constrained":
        warnings.append("当前可用资源偏紧，建议释放显存、使用量化或启用 CPU Offload")
    elif fit == "unsupported":
        warnings.append("当前节点的系统、内存或显存不满足本地部署要求")
    if model.requires_token:
        warnings.append("模型访问可能需要单独的许可或访问令牌")

    command = model.command_templates.get(runtime)
    return CandidateRecommendation(
        model_id=model.model_id,
        display_name=model.display_name,
        score=0,
        fit_tier=fit,
        recommended_runtime=runtime,
        deployment_command=command,
        parameter_size=model.parameter_size,
        quantization=estimate.variant.quantization,
        min_vram_gb=model.min_vram_gb,
        recommended_vram_gb=model.recommended_vram_gb,
        estimated_vram_gb=estimate.variant.estimated_vram_gb,
        min_ram_gb=estimate.variant.min_ram_gb,
        hardware_reasons=_hardware_reasons(model, inventory, estimate, runtime),
        tradeoffs=_tradeoffs(model, estimate),
        reasons=reasons,
        warnings=warnings,
        source="local_catalog",
        execution_mode=estimate.variant.mode,
        utilization=estimate.utilization.model_dump(),
        estimate_confidence=estimate.confidence,
        estimate_basis=estimate.basis,
        selected_variant=estimate.variant.model_dump(),
    )


def recommend_models(
    analysis: TaskAnalysis,
    inventory: HardwareInventory,
    *,
    catalog: list[ModelSpec] | None = None,
    max_candidates_per_stage: int = 3,
    quality: str = "balanced",
    allow_offload: bool = True,
    external_candidates: dict[str, list[CandidateRecommendation]] | None = None,
) -> list[StageRecommendation]:
    models = catalog if catalog is not None else load_catalog()
    results: list[StageRecommendation] = []
    preference = quality if quality in PREFERENCE_WEIGHTS else "balanced"

    for stage in analysis.stages:
        matching = [model for model in models if stage.capability in model.capabilities]
        model_lookup = {model.model_id: model for model in matching}
        local_candidates = list(
            _candidate(model, inventory, allow_offload)
            for model in matching
        )
        combined = [
            *local_candidates,
            *((external_candidates or {}).get(stage.capability) or []),
        ]
        deduplicated: dict[str, CandidateRecommendation] = {}
        for candidate in combined:
            current = deduplicated.get(candidate.model_id)
            if current is None or candidate.score > current.score:
                deduplicated[candidate.model_id] = candidate
        candidates = rank_candidates(
            list(deduplicated.values()),
            inventory,
            preference=preference,
            model_lookup=model_lookup,
        )[:max_candidates_per_stage]
        local_supported = any(
            candidate.fit_tier in {"ideal", "runnable", "constrained"}
            for candidate in candidates
        )
        if not candidates:
            fallback = "模型目录中暂无该能力，请扩充 models.json 或接入外部模型服务。"
        elif local_supported:
            fallback = "优先使用首选本地方案；资源繁忙时可切换备选模型。"
        else:
            fallback = "当前环境不适合本地运行，建议使用更大显存的服务器或远端模型服务。"
        results.append(
            StageRecommendation(
                stage=stage,
                candidates=candidates,
                local_supported=local_supported,
                fallback=fallback,
            )
        )
    return results

