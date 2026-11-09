from backend.agents.model_selector.graph import build_model_selector_graph
from backend.agents.model_selector.state import (
    CandidateRecommendation,
    GPUDevice,
    HardwareInventory,
)


def _inventory(environment_type: str = "personal_computer") -> HardwareInventory:
    gpu_count = 1 if environment_type == "personal_computer" else 4
    ram = 16 if environment_type == "personal_computer" else 128
    return HardwareInventory(
        os_name="Windows" if environment_type == "personal_computer" else "Linux",
        os_version="11",
        architecture="AMD64",
        cpu_model="Test CPU",
        physical_cores=14,
        logical_cores=20 if environment_type == "personal_computer" else 64,
        ram_total_gb=ram,
        ram_available_gb=ram * 0.75,
        disk_free_gb=500,
        gpus=[
            GPUDevice(
                name=f"NVIDIA Test GPU {index}",
                total_vram_gb=8 if environment_type == "personal_computer" else 24,
                free_vram_gb=7 if environment_type == "personal_computer" else 22,
                driver_version="595.97",
                compute_capability="8.9",
            )
            for index in range(gpu_count)
        ],
        runtimes={
            "docker": True,
            "ollama": True,
            "kubectl": environment_type == "cluster",
            "ffmpeg": True,
            "llama_cpp": False,
            "vllm": environment_type != "personal_computer",
        },
        environment_type=environment_type,
        kubernetes=environment_type == "cluster",
    )


def test_graph_routes_personal_computer_and_builds_local_pipeline():
    graph = build_model_selector_graph(lambda: _inventory())

    result = graph.invoke(
        {
            "task": "把会议视频转成文字，区分说话人，然后生成中文摘要",
            "preferences": {"quality": "balanced", "max_candidates_per_stage": 2},
        }
    )

    assert "analyze_personal_computer" in result["trace"]
    assert len(result["recommendations"]) == 4
    assert result["deployment_plan"]["execution_mode"] == "local_pipeline"
    assert "串行" in " ".join(result["deployment_plan"]["next_actions"])


def test_graph_routes_multi_gpu_server():
    graph = build_model_selector_graph(lambda: _inventory("single_server"))

    result = graph.invoke({"task": "部署一个中文代码生成模型", "preferences": {}})

    assert "analyze_single_server" in result["trace"]
    assert result["deployment_plan"]["environment_summary"].startswith("单机服务器")
    assert result["recommendations"][0]["local_supported"] is True


def test_graph_recommends_remote_when_local_resources_do_not_fit():
    graph = build_model_selector_graph(lambda: _inventory())

    result = graph.invoke({"task": "根据文字描述生成一段视频", "preferences": {}})

    assert result["recommendations"][0]["local_supported"] is False
    assert result["deployment_plan"]["execution_mode"] == "remote_or_larger_server"
    assert any("服务器" in warning for warning in result["deployment_plan"]["warnings"])


def test_graph_uses_simulated_24gb_server_before_environment_routing():
    graph = build_model_selector_graph(lambda: _inventory())

    result = graph.invoke(
        {
            "task": "部署一个中文代码生成模型",
            "preferences": {},
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
        }
    )

    assert result["hardware"]["os_name"] == "Linux"
    assert result["hardware"]["ram_total_gb"] == 64
    assert result["hardware"]["gpus"][0]["total_vram_gb"] == 24
    assert result["hardware"]["metadata"]["hardware_source"] == "simulated"
    assert "analyze_single_server" in result["trace"]


def test_graph_can_simulate_two_48gb_gpus():
    graph = build_model_selector_graph(lambda: _inventory())

    result = graph.invoke(
        {
            "task": "根据文字描述生成一段视频",
            "preferences": {},
            "hardware_override": {
                "ram_total_gb": 128,
                "ram_available_gb": 112,
                "gpu_count": 2,
                "gpu_name": "NVIDIA L40S",
                "gpu_total_vram_gb": 48,
                "gpu_free_vram_gb": 44,
                "environment_type": "single_server",
            },
        }
    )

    assert len(result["hardware"]["gpus"]) == 2
    assert all(gpu["total_vram_gb"] == 48 for gpu in result["hardware"]["gpus"])
    assert result["recommendations"][0]["local_supported"] is True


def test_graph_merges_llmfit_candidate_for_text_stage():
    class FakeLlmfitClient:
        enabled = True

        def recommend(self, stage, inventory, *, limit):
            assert stage.capability == "code_generation"
            return [
                CandidateRecommendation(
                    model_id="llmfit/qwen3-8b",
                    display_name="Qwen3 8B via llmfit",
                    score=99,
                    fit_tier="ideal",
                    recommended_runtime="ollama",
                    parameter_size="8B",
                    quantization="Q4_K_M",
                    estimated_vram_gb=6.2,
                    min_ram_gb=12,
                    source="llmfit",
                    execution_mode="direct",
                    utilization={"peak_ratio": 0.3},
                    estimate_confidence="calibrated",
                    estimate_basis={"source": "llmfit"},
                    selected_variant={"id": "llmfit"},
                )
            ][:limit]

    graph = build_model_selector_graph(
        lambda: _inventory("single_server"),
        llmfit_client=FakeLlmfitClient(),
    )

    result = graph.invoke(
        {
            "task": "部署一个中文代码生成模型",
            "preferences": {"use_llmfit": True},
        }
    )

    assert result["recommendations"][0]["candidates"][0]["source"] == "llmfit"
    assert "llmfit_candidates" in result["trace"]

