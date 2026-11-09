from collections.abc import Callable
from typing import Any, Literal

from .hardware import apply_hardware_override, detect_environment
from .llmfit_client import LlmfitClient
from .matcher import recommend_models
from .state import HardwareInventory, StageRecommendation, TaskAnalysis
from .task_classifier import classify_task


GraphState = dict[str, Any]


def classify_task_node(state: GraphState) -> GraphState:
    analysis = classify_task(state["task"])
    return {
        "task_analysis": analysis.model_dump(),
        "trace": ["classify_task"],
    }


def make_detect_environment_node(
    detector: Callable[[], HardwareInventory] = detect_environment,
) -> Callable[[GraphState], GraphState]:
    def detect_environment_node(state: GraphState) -> GraphState:
        inventory = apply_hardware_override(
            detector(),
            state.get("hardware_override"),
        )
        return {
            "hardware": inventory.model_dump(),
            "trace": ["detect_environment"],
        }

    return detect_environment_node


def route_environment(
    state: GraphState,
) -> Literal[
    "analyze_personal_computer",
    "analyze_single_server",
    "analyze_cluster",
]:
    environment_type = state["hardware"]["environment_type"]
    return f"analyze_{environment_type}"


def analyze_personal_computer_node(_: GraphState) -> GraphState:
    return {
        "environment_notes": [
            "个人电脑优先选择量化、小参数模型。",
            "复杂流水线按阶段串行加载，避免多个模型同时占用显存。",
        ],
        "trace": ["analyze_personal_computer"],
    }


def analyze_single_server_node(_: GraphState) -> GraphState:
    return {
        "environment_notes": [
            "单机服务器可优先使用容器化推理服务。",
            "多 GPU 环境可进一步评估张量并行、流水线并行或多副本。",
        ],
        "trace": ["analyze_single_server"],
    }


def analyze_cluster_node(_: GraphState) -> GraphState:
    return {
        "environment_notes": [
            "集群环境应按节点 GPU 类型、显存和副本数进行调度。",
            "第一版仅给出推荐，不会自动创建 Kubernetes 工作负载。",
        ],
        "trace": ["analyze_cluster"],
    }


def make_select_models_node(
    llmfit_client: LlmfitClient | None = None,
) -> Callable[[GraphState], GraphState]:
    def select_models_node(state: GraphState) -> GraphState:
        analysis = TaskAnalysis.model_validate(state["task_analysis"])
        inventory = HardwareInventory.model_validate(state["hardware"])
        preferences = state.get("preferences") or {}
        max_candidates = int(preferences.get("max_candidates_per_stage", 3))
        max_candidates = max(1, min(max_candidates, 10))
        external_candidates = {}
        if (
            llmfit_client is not None
            and bool(preferences.get("use_llmfit", True))
        ):
            for stage in analysis.stages:
                candidates = llmfit_client.recommend(
                    stage,
                    inventory,
                    limit=max_candidates,
                )
                if candidates:
                    external_candidates[stage.capability] = candidates

        recommendations = recommend_models(
            analysis,
            inventory,
            max_candidates_per_stage=max_candidates,
            quality=str(preferences.get("quality", "balanced")),
            allow_offload=bool(preferences.get("allow_offload", True)),
            external_candidates=external_candidates,
        )
        trace = ["select_models"]
        if external_candidates:
            trace.insert(0, "llmfit_candidates")
        return {
            "recommendations": [item.model_dump() for item in recommendations],
            "trace": trace,
        }

    return select_models_node


select_models_node = make_select_models_node()


def _environment_summary(inventory: HardwareInventory) -> str:
    gpu_summary = (
        f"{len(inventory.gpus)} 张 GPU，"
        f"最大单卡显存 {max(gpu.total_vram_gb for gpu in inventory.gpus):.1f}GB"
        if inventory.gpus
        else "无可识别独立 GPU"
    )
    if inventory.environment_type == "single_server":
        prefix = "单机服务器"
    elif inventory.environment_type == "cluster":
        prefix = "集群环境"
    else:
        prefix = "个人电脑"
    return (
        f"{prefix}：{inventory.os_name}，{inventory.logical_cores} 线程 CPU，"
        f"{inventory.ram_total_gb:.1f}GB 内存，{gpu_summary}"
    )


def build_plan_node(state: GraphState) -> GraphState:
    inventory = HardwareInventory.model_validate(state["hardware"])
    recommendations = [
        StageRecommendation.model_validate(item)
        for item in state["recommendations"]
    ]
    supported_count = sum(item.local_supported for item in recommendations)
    if supported_count == len(recommendations):
        execution_mode = (
            "local_pipeline" if len(recommendations) > 1 else "local_single_model"
        )
    elif supported_count == 0:
        execution_mode = "remote_or_larger_server"
    else:
        execution_mode = "hybrid_pipeline"

    steps: list[str] = []
    warnings = list(inventory.detection_warnings)
    for index, recommendation in enumerate(recommendations, start=1):
        if not recommendation.candidates:
            steps.append(f"{index}. {recommendation.stage.name}：暂无候选")
            warnings.append(recommendation.fallback)
            continue
        candidate = recommendation.candidates[0]
        steps.append(
            f"{index}. {recommendation.stage.name}：使用 "
            f"{candidate.display_name}（{candidate.recommended_runtime}，"
            f"{candidate.fit_tier}）"
        )
        warnings.extend(candidate.warnings)
        if not recommendation.local_supported:
            warnings.append(recommendation.fallback)

    next_actions = list(state.get("environment_notes") or [])
    if execution_mode == "remote_or_larger_server":
        next_actions.append("选择显存和内存更大的服务器，或配置远端模型服务。")
    elif execution_mode == "hybrid_pipeline":
        next_actions.append("本地运行可适配阶段，其余阶段调用服务器或远端服务。")

    plan = {
        "environment_summary": _environment_summary(inventory),
        "execution_mode": execution_mode,
        "steps": steps,
        "warnings": list(dict.fromkeys(warnings)),
        "next_actions": next_actions,
    }
    return {
        "deployment_plan": plan,
        "trace": ["build_plan"],
    }

