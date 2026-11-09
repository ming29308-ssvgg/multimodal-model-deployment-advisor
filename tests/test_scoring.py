from backend.agents.model_selector.scoring import (
    PREFERENCE_WEIGHTS,
    rank_candidates,
)
from backend.agents.model_selector.state import (
    CandidateRecommendation,
    GPUDevice,
    HardwareInventory,
)


def _inventory() -> HardwareInventory:
    return HardwareInventory(
        os_name="Linux",
        os_version="test",
        architecture="x86_64",
        cpu_model="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_total_gb=64,
        ram_available_gb=56,
        disk_free_gb=500,
        gpus=[
            GPUDevice(
                name="NVIDIA Test GPU",
                total_vram_gb=24,
                free_vram_gb=22,
            )
        ],
        runtimes={"transformers": True, "diffusers": True},
        environment_type="single_server",
    )


def _candidate(
    model_id: str,
    *,
    fit_tier: str = "runnable",
    parameter_size: str | None = None,
    vram: float = 4,
    ram: float = 8,
    peak_ratio: float | None = 0.5,
    runtime: str = "transformers",
    execution_mode: str = "direct",
    confidence: str = "catalog",
) -> CandidateRecommendation:
    return CandidateRecommendation(
        model_id=model_id,
        display_name=model_id,
        score=0,
        fit_tier=fit_tier,
        recommended_runtime=runtime,
        parameter_size=parameter_size,
        estimated_vram_gb=vram,
        min_ram_gb=ram,
        execution_mode=execution_mode,
        utilization={"peak_ratio": peak_ratio},
        estimate_confidence=confidence,
        selected_variant={"quality_rank": 80},
    )


def test_preference_weights_match_the_approved_profiles():
    assert PREFERENCE_WEIGHTS == {
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
    assert all(sum(weights.values()) == 100 for weights in PREFERENCE_WEIGHTS.values())


def test_speed_and_quality_preferences_reverse_small_and_large_model_ranking():
    candidates = [
        _candidate("small", parameter_size="1B", vram=3, ram=6, peak_ratio=0.3),
        _candidate("large", parameter_size="14B", vram=12, ram=24, peak_ratio=0.55),
    ]

    speed = rank_candidates(candidates, _inventory(), preference="speed")
    quality = rank_candidates(candidates, _inventory(), preference="quality")

    assert [candidate.model_id for candidate in speed] == ["small", "large"]
    assert [candidate.model_id for candidate in quality] == ["large", "small"]


def test_unsupported_candidate_never_outranks_viable_candidate_for_quality():
    candidates = [
        _candidate("viable", parameter_size="3B", fit_tier="runnable"),
        _candidate(
            "unsupported-large",
            parameter_size="70B",
            fit_tier="unsupported",
            vram=48,
            ram=96,
            peak_ratio=1.4,
        ),
    ]

    ranked = rank_candidates(candidates, _inventory(), preference="quality")

    assert ranked[0].model_id == "viable"
    assert ranked[1].model_id == "unsupported-large"


def test_single_candidate_with_missing_metadata_gets_complete_bounded_breakdown():
    ranked = rank_candidates(
        [
            _candidate(
                "unknown",
                parameter_size=None,
                vram=0,
                ram=1,
                peak_ratio=None,
                runtime="missing_runtime",
                confidence="unknown",
            )
        ],
        _inventory(),
        preference="balanced",
    )

    candidate = ranked[0]
    assert candidate.preference == "balanced"
    assert candidate.score_breakdown["weights"] == PREFERENCE_WEIGHTS["balanced"]
    assert set(candidate.score_breakdown) == {
        "hardware",
        "speed",
        "quality",
        "readiness",
        "confidence",
        "weights",
        "final",
    }
    for key in ["hardware", "speed", "quality", "readiness", "confidence", "final"]:
        assert 0 <= candidate.score_breakdown[key] <= 100
    assert candidate.score == candidate.score_breakdown["final"]
    assert candidate.ranking_reasons
