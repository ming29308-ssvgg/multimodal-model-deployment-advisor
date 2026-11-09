import math
from collections.abc import Mapping, Sequence
from typing import Literal

from .state import CandidateRecommendation, HardwareInventory, ModelSpec


Preference = Literal["speed", "balanced", "quality"]

PREFERENCE_WEIGHTS: dict[Preference, dict[str, int]] = {
    "speed": {
        "hardware": 35,
        "speed": 35,
        "quality": 15,
        "readiness": 10,
        "confidence": 5,
    },
    "balanced": {
        "hardware": 35,
        "speed": 20,
        "quality": 25,
        "readiness": 10,
        "confidence": 10,
    },
    "quality": {
        "hardware": 30,
        "speed": 10,
        "quality": 45,
        "readiness": 10,
        "confidence": 5,
    },
}

CONFIDENCE_SCORES = {
    "benchmarked": 100,
    "calibrated": 90,
    "catalog": 75,
    "heuristic": 55,
    "unknown": 35,
}


def _bounded(value: float) -> int:
    return max(0, min(round(value), 100))


def _parameter_billions(parameter_size: str | None) -> float:
    if not parameter_size:
        return 0.0
    normalized = parameter_size.strip().upper()
    try:
        if normalized.endswith("B"):
            return float(normalized[:-1])
        if normalized.endswith("M"):
            return float(normalized[:-1]) / 1000
    except ValueError:
        return 0.0
    return 0.0


def _hardware_score(candidate: CandidateRecommendation) -> int:
    ratio = candidate.utilization.get("peak_ratio")
    if candidate.fit_tier == "unsupported":
        if isinstance(ratio, (int, float)):
            return _bounded(45 - max(float(ratio) - 1, 0) * 30)
        return 0
    if not isinstance(ratio, (int, float)):
        return {
            "ideal": 95,
            "runnable": 82,
            "constrained": 52,
        }[candidate.fit_tier]
    if candidate.fit_tier == "ideal":
        score = 100 - float(ratio) * 12
    elif candidate.fit_tier == "runnable":
        score = 96 - float(ratio) * 18
    else:
        score = 72 - float(ratio) * 22
    if candidate.execution_mode == "cpu_offload":
        score = min(score, 55)
    return _bounded(score)


def _speed_cost(candidate: CandidateRecommendation) -> float:
    cost = (
        candidate.estimated_vram_gb * 1.6
        + candidate.min_ram_gb * 0.35
        + _parameter_billions(candidate.parameter_size) * 0.8
    )
    if candidate.execution_mode == "cpu_offload":
        cost += 15
    elif candidate.execution_mode == "cpu_only":
        cost += 25
    return cost


def _quality_signal(
    candidate: CandidateRecommendation,
    model: ModelSpec | None,
) -> float:
    quality_rank = candidate.selected_variant.get("quality_rank", 50)
    if not isinstance(quality_rank, (int, float)):
        quality_rank = 50
    parameter_size = _parameter_billions(candidate.parameter_size)
    resource_class = (
        candidate.estimated_vram_gb * 1.5
        + candidate.min_ram_gb * 0.15
    )
    catalog_priority = model.catalog_priority if model is not None else 0
    external_quality_prior = candidate.score if candidate.source == "llmfit" else 0
    signal = (
        float(quality_rank) * 0.35
        + math.log2(1 + parameter_size) * 12
        + math.log2(1 + resource_class) * 8
        + catalog_priority * 0.5
        + external_quality_prior * 0.5
    )
    if candidate.execution_mode == "quantized":
        signal -= 2
    return signal


def _relative_scores(
    values: Sequence[float],
    *,
    inverse: bool,
) -> list[int]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [60 for _ in values]
    scores: list[int] = []
    for value in values:
        position = (value - minimum) / (maximum - minimum)
        if inverse:
            position = 1 - position
        scores.append(_bounded(20 + position * 80))
    return scores


def _group_relative_scores(
    candidates: Sequence[CandidateRecommendation],
    values: Sequence[float],
    *,
    inverse: bool,
) -> list[int]:
    scores = [60 for _ in candidates]
    for is_viable in (True, False):
        indexes = [
            index
            for index, candidate in enumerate(candidates)
            if (candidate.fit_tier != "unsupported") is is_viable
        ]
        group_scores = _relative_scores(
            [values[index] for index in indexes],
            inverse=inverse,
        )
        for index, score in zip(indexes, group_scores, strict=True):
            scores[index] = score
    return scores


def _readiness_score(
    candidate: CandidateRecommendation,
    inventory: HardwareInventory,
    model: ModelSpec | None,
) -> int:
    score = 100 if inventory.runtimes.get(candidate.recommended_runtime, False) else 45
    if model is not None and model.requires_token:
        score -= 20
    return _bounded(score)


def _ranking_reasons(
    candidate: CandidateRecommendation,
    preference: Preference,
    components: dict[str, int],
) -> list[str]:
    preference_labels = {
        "speed": "速度优先",
        "balanced": "均衡",
        "quality": "质量优先",
    }
    focus = "speed" if preference == "speed" else "quality" if preference == "quality" else "hardware"
    component_labels = {
        "hardware": "硬件适配",
        "speed": "速度效率",
        "quality": "预期质量",
    }
    reasons = [
        f"按“{preference_labels[preference]}”权重排序，"
        f"{component_labels[focus]}得分 {components[focus]}/100。",
        f"硬件适配 {components['hardware']}/100，"
        f"部署就绪度 {components['readiness']}/100。",
    ]
    if candidate.fit_tier == "unsupported":
        reasons.append("当前硬件不可直接运行，因此始终排在可运行候选之后。")
    elif candidate.execution_mode == "cpu_offload":
        reasons.append("需要 CPU Offload，速度和硬件余量评分受到惩罚。")
    return reasons


def rank_candidates(
    candidates: Sequence[CandidateRecommendation],
    inventory: HardwareInventory,
    *,
    preference: Preference,
    model_lookup: Mapping[str, ModelSpec] | None = None,
) -> list[CandidateRecommendation]:
    if not candidates:
        return []
    lookup = model_lookup or {}
    speed_scores = _group_relative_scores(
        candidates,
        [_speed_cost(candidate) for candidate in candidates],
        inverse=True,
    )
    quality_scores = _group_relative_scores(
        candidates,
        [
            _quality_signal(candidate, lookup.get(candidate.model_id))
            for candidate in candidates
        ],
        inverse=False,
    )
    weights = PREFERENCE_WEIGHTS[preference]
    ranked: list[CandidateRecommendation] = []
    for candidate, speed_score, quality_score in zip(
        candidates,
        speed_scores,
        quality_scores,
        strict=True,
    ):
        model = lookup.get(candidate.model_id)
        components = {
            "hardware": _hardware_score(candidate),
            "speed": speed_score,
            "quality": quality_score,
            "readiness": _readiness_score(candidate, inventory, model),
            "confidence": CONFIDENCE_SCORES.get(
                candidate.estimate_confidence,
                CONFIDENCE_SCORES["unknown"],
            ),
        }
        final_score = _bounded(
            sum(
                components[name] * weight
                for name, weight in weights.items()
            )
            / 100
        )
        score_breakdown = {
            **components,
            "weights": dict(weights),
            "final": final_score,
        }
        ranked.append(
            candidate.model_copy(
                update={
                    "score": final_score,
                    "preference": preference,
                    "score_breakdown": score_breakdown,
                    "ranking_reasons": _ranking_reasons(
                        candidate,
                        preference,
                        components,
                    ),
                }
            )
        )
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate.fit_tier != "unsupported",
            candidate.score,
            candidate.score_breakdown["hardware"],
        ),
        reverse=True,
    )
