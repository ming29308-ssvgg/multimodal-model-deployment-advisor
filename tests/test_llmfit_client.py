from backend.agents.model_selector.llmfit_client import LlmfitClient
from backend.agents.model_selector.state import GPUDevice, HardwareInventory
from backend.agents.model_selector.task_classifier import classify_task


def _inventory(*, source: str = "detected") -> HardwareInventory:
    return HardwareInventory(
        os_name="Linux",
        os_version="test",
        architecture="x86_64",
        cpu_model="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_total_gb=64,
        ram_available_gb=48,
        disk_free_gb=500,
        gpus=[
            GPUDevice(
                name="NVIDIA RTX 4090",
                total_vram_gb=24,
                free_vram_gb=22,
            )
        ],
        runtimes={"ollama": True, "vllm": True},
        environment_type="single_server",
        metadata={"hardware_source": source},
    )


def test_maps_llmfit_top_model_response_to_candidate():
    requested_urls: list[str] = []

    def fetch_json(url: str, timeout: float) -> dict:
        requested_urls.append(url)
        assert timeout == 1.5
        return {
            "models": [
                {
                    "id": "qwen3-8b",
                    "name": "Qwen3 8B",
                    "parameter_count": "8B",
                    "best_quant": "Q4_K_M",
                    "memory_required_gb": 6.2,
                    "memory_available_gb": 22,
                    "utilization_pct": 28.2,
                    "fit": "P",
                    "run_mode": "GPU",
                    "runtimes": ["ollama", "vllm"],
                    "score": 91,
                    "estimate_confidence": "high",
                    "estimate_basis": "community benchmark",
                }
            ]
        }

    client = LlmfitClient(
        "http://127.0.0.1:8787",
        timeout=1.5,
        fetch_json=fetch_json,
    )
    stage = classify_task("部署一个中文代码生成模型").stages[0]

    candidates = client.recommend(stage, _inventory(), limit=3)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.model_id == "qwen3-8b"
    assert candidate.source == "llmfit"
    assert candidate.fit_tier == "ideal"
    assert candidate.execution_mode == "direct"
    assert candidate.quantization == "Q4_K_M"
    assert candidate.estimate_confidence == "calibrated"
    assert candidate.utilization["peak_ratio"] == 0.282
    assert "use_case=coding" in requested_urls[0]


def test_llmfit_failure_returns_empty_candidates():
    def failing_fetcher(url: str, timeout: float) -> dict:
        raise TimeoutError("offline")

    client = LlmfitClient(
        "http://127.0.0.1:8787",
        fetch_json=failing_fetcher,
    )
    stage = classify_task("总结一篇中文文章").stages[0]

    assert client.recommend(stage, _inventory(), limit=3) == []


def test_llmfit_is_bypassed_for_video_and_simulated_hardware():
    calls = 0

    def fetch_json(url: str, timeout: float) -> dict:
        nonlocal calls
        calls += 1
        return {"models": []}

    client = LlmfitClient(
        "http://127.0.0.1:8787",
        fetch_json=fetch_json,
    )

    video_stage = classify_task("我想在本地部署一个文生视频模型").stages[0]
    text_stage = classify_task("总结一篇中文文章").stages[0]

    assert client.recommend(video_stage, _inventory(), limit=3) == []
    assert (
        client.recommend(
            text_stage,
            _inventory(source="simulated"),
            limit=3,
        )
        == []
    )
    assert calls == 0
