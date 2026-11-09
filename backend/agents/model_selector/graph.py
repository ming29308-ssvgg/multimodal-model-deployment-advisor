import operator
from collections.abc import Callable
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .hardware import detect_environment
from .nodes import (
    analyze_cluster_node,
    analyze_personal_computer_node,
    analyze_single_server_node,
    build_plan_node,
    classify_task_node,
    make_detect_environment_node,
    make_select_models_node,
    route_environment,
)
from .llmfit_client import LlmfitClient
from .state import HardwareInventory


class ModelSelectorState(TypedDict, total=False):
    task: str
    preferences: dict[str, Any]
    hardware_override: dict[str, Any]
    task_analysis: dict[str, Any]
    hardware: dict[str, Any]
    environment_notes: list[str]
    recommendations: list[dict[str, Any]]
    deployment_plan: dict[str, Any]
    trace: Annotated[list[str], operator.add]


def build_model_selector_graph(
    environment_detector: Callable[[], HardwareInventory] = detect_environment,
    llmfit_client: LlmfitClient | None = None,
):
    effective_llmfit_client = (
        llmfit_client
        if llmfit_client is not None
        else LlmfitClient.from_env()
    )
    builder = StateGraph(ModelSelectorState)
    builder.add_node("classify_task", classify_task_node)
    builder.add_node(
        "detect_environment",
        make_detect_environment_node(environment_detector),
    )
    builder.add_node("analyze_personal_computer", analyze_personal_computer_node)
    builder.add_node("analyze_single_server", analyze_single_server_node)
    builder.add_node("analyze_cluster", analyze_cluster_node)
    builder.add_node(
        "select_models",
        make_select_models_node(effective_llmfit_client),
    )
    builder.add_node("build_plan", build_plan_node)

    builder.add_edge(START, "classify_task")
    builder.add_edge("classify_task", "detect_environment")
    builder.add_conditional_edges(
        "detect_environment",
        route_environment,
        [
            "analyze_personal_computer",
            "analyze_single_server",
            "analyze_cluster",
        ],
    )
    builder.add_edge("analyze_personal_computer", "select_models")
    builder.add_edge("analyze_single_server", "select_models")
    builder.add_edge("analyze_cluster", "select_models")
    builder.add_edge("select_models", "build_plan")
    builder.add_edge("build_plan", END)
    return builder.compile()


model_selector_graph = build_model_selector_graph()


def run_recommendation(
    task: str,
    preferences: dict[str, Any] | None = None,
    hardware_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return model_selector_graph.invoke(
        {
            "task": task,
            "preferences": preferences or {},
            "hardware_override": hardware_override or {},
            "trace": [],
        }
    )

