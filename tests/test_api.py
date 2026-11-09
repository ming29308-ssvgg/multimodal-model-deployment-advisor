import asyncio

import httpx
from backend.main import app


async def _post(path: str, payload: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.post(path, json=payload)


def test_recommend_endpoint_detects_hardware_without_hardware_input():
    response = asyncio.run(
        _post(
            "/api/v1/model-selector/recommend",
            {"task": "总结这篇中文文章"},
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_analysis"]["primary_modality"] == "text"
    assert body["hardware"]["ram_total_gb"] > 0
    assert "gpus" in body["hardware"]
    candidates = body["recommendations"][0]["candidates"]
    assert len(candidates) == 3
    assert len({candidate["model_id"] for candidate in candidates}) == 3
    for field in [
        "parameter_size",
        "quantization",
        "min_vram_gb",
        "recommended_vram_gb",
        "estimated_vram_gb",
        "min_ram_gb",
        "hardware_reasons",
        "tradeoffs",
    ]:
        assert field in candidates[0]
    assert candidates[0]["hardware_reasons"]
    assert candidates[0]["tradeoffs"]
    assert candidates[0]["preference"] == "balanced"
    assert candidates[0]["score"] == candidates[0]["score_breakdown"]["final"]
    assert set(candidates[0]["score_breakdown"]) == {
        "hardware",
        "speed",
        "quality",
        "readiness",
        "confidence",
        "weights",
        "final",
    }
    assert candidates[0]["ranking_reasons"]


def test_recommend_endpoint_rejects_empty_task():
    response = asyncio.run(
        _post(
            "/api/v1/model-selector/recommend",
            {"task": "   "},
        )
    )

    assert response.status_code == 422


def test_recommend_endpoint_accepts_hardware_override():
    response = asyncio.run(
        _post(
            "/api/v1/model-selector/recommend",
            {
                "task": "部署一个中文代码生成模型",
                "hardware_override": {
                    "os_name": "Linux",
                    "ram_total_gb": 64,
                    "ram_available_gb": 56,
                    "gpu_count": 1,
                    "gpu_name": "NVIDIA RTX 4090",
                    "gpu_total_vram_gb": 24,
                    "gpu_free_vram_gb": 22,
                    "environment_type": "single_server",
                },
            },
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hardware"]["metadata"]["hardware_source"] == "simulated"
    assert body["hardware"]["gpus"][0]["total_vram_gb"] == 24


def test_recommend_endpoint_can_return_ten_distinct_candidates():
    response = asyncio.run(
        _post(
            "/api/v1/model-selector/recommend",
            {
                "task": "总结一篇中文技术文章",
                "preferences": {
                    "max_candidates_per_stage": 10,
                    "use_llmfit": False,
                },
                "hardware_override": {
                    "os_name": "Linux",
                    "ram_total_gb": 256,
                    "ram_available_gb": 240,
                    "gpu_count": 2,
                    "gpu_name": "NVIDIA L40S",
                    "gpu_total_vram_gb": 48,
                    "gpu_free_vram_gb": 46,
                    "environment_type": "single_server",
                },
            },
        )
    )

    assert response.status_code == 200
    candidates = response.json()["recommendations"][0]["candidates"]
    assert len(candidates) == 10
    assert len({candidate["model_id"] for candidate in candidates}) == 10

