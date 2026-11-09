from backend.agents.model_selector.catalog import load_catalog
from backend.agents.model_selector.matcher import recommend_models
from backend.agents.model_selector.state import (
    GPUDevice,
    HardwareInventory,
    ModelSpec,
)
from backend.agents.model_selector.task_classifier import classify_task


def _inventory(*, gpu_vram: float | None = 8.0, ram: float = 16.0) -> HardwareInventory:
    gpus = []
    if gpu_vram is not None:
        gpus = [
            GPUDevice(
                name="NVIDIA Test GPU",
                total_vram_gb=gpu_vram,
                free_vram_gb=max(gpu_vram - 1.5, 0),
                driver_version="595.97",
                compute_capability="8.9",
            )
        ]
    return HardwareInventory(
        os_name="Windows",
        os_version="11",
        architecture="AMD64",
        cpu_model="Test CPU",
        physical_cores=14,
        logical_cores=20,
        ram_total_gb=ram,
        ram_available_gb=ram * 0.6,
        disk_free_gb=200,
        gpus=gpus,
        runtimes={
            "docker": True,
            "ollama": True,
            "kubectl": False,
            "ffmpeg": True,
            "llama_cpp": False,
            "vllm": False,
        },
        environment_type="personal_computer",
    )


def test_catalog_covers_core_multimodal_capabilities():
    capabilities = {
        capability
        for model in load_catalog()
        for capability in model.capabilities
    }

    assert {
        "text_generation",
        "summarization",
        "image_understanding",
        "image_generation",
        "speech_to_text",
        "text_to_speech",
        "video_understanding",
        "video_generation",
    }.issubset(capabilities)


def test_catalog_has_at_least_ten_distinct_candidates_for_major_capabilities():
    counts: dict[str, int] = {}
    model_ids: dict[str, set[str]] = {}
    for model in load_catalog():
        for capability in model.capabilities:
            counts[capability] = counts.get(capability, 0) + 1
            model_ids.setdefault(capability, set()).add(model.model_id)

    for capability in [
        "text_generation",
        "summarization",
        "translation",
        "code_generation",
        "image_understanding",
        "video_understanding",
        "speech_to_text",
        "text_to_speech",
        "image_generation",
        "video_generation",
    ]:
        assert counts.get(capability, 0) >= 10, capability
        assert len(model_ids.get(capability, set())) >= 10, capability


def test_catalog_model_ids_are_unique_after_merging_catalog_files():
    model_ids = [model.model_id for model in load_catalog()]

    assert len(model_ids) == len(set(model_ids))


def test_major_tasks_return_three_distinct_model_candidates():
    tasks = {
        "总结这篇中文文章": "summarization",
        "编写一个 Python 服务": "code_generation",
        "理解这张图片内容": "image_understanding",
        "把这段录音转成文字": "speech_to_text",
        "把这段中文转成语音": "text_to_speech",
        "根据描述生成一张图片": "image_generation",
        "根据文字描述生成一段视频": "video_generation",
    }

    for task, capability in tasks.items():
        recommendations = recommend_models(
            classify_task(task),
            _inventory(),
            max_candidates_per_stage=3,
        )
        stage = next(
            item for item in recommendations if item.stage.capability == capability
        )
        assert len(stage.candidates) == 3, capability
        assert len({candidate.model_id for candidate in stage.candidates}) == 3


def test_recommends_hardware_specific_quantization_levels_for_8gb_gpu():
    recommendations = recommend_models(
        classify_task("总结一篇很长的中文文章"),
        _inventory(),
    )

    candidate = recommendations[0].candidates[0]
    assert candidate.model_id == "google/gemma-3-4b-it"
    assert candidate.fit_tier == "runnable"
    assert candidate.recommended_runtime == "ollama"
    assert recommendations[0].local_supported is True
    assert candidate.parameter_size == "4B"
    assert candidate.quantization == "Q3_K_M"
    assert candidate.estimated_vram_gb == 5.1
    assert candidate.min_ram_gb == 7.5
    assert {
        item.quantization
        for item in recommendations[0].candidates
    } == {"Q3_K_M", "Q5_K_M"}
    assert any("6.5GB" in reason for reason in candidate.hardware_reasons)
    assert candidate.tradeoffs
    assert candidate.source == "local_catalog"
    assert candidate.execution_mode in {"direct", "quantized", "cpu_offload", "cpu_only"}
    assert candidate.estimate_confidence in {"catalog", "heuristic"}
    assert candidate.estimate_basis
    assert "peak_ratio" in candidate.utilization
    assert candidate.selected_variant["id"]


def test_selects_cpu_offload_when_free_vram_is_too_low():
    inventory = _inventory(ram=32)
    inventory.gpus[0].free_vram_gb = 5

    recommendations = recommend_models(
        classify_task("总结一篇很长的中文文章"),
        inventory,
        quality="quality",
        max_candidates_per_stage=3,
    )
    candidate = next(
        item
        for item in recommendations[0].candidates
        if item.model_id == "Qwen/Qwen2.5-7B-Instruct"
    )

    assert candidate.fit_tier == "constrained"
    assert candidate.execution_mode == "cpu_offload"
    assert any("CPU Offload" in reason for reason in candidate.hardware_reasons)


def test_cpu_only_candidate_explains_cpu_and_memory_fit_without_gpu_claims():
    recommendations = recommend_models(
        classify_task("把这段中文转成语音"),
        _inventory(gpu_vram=None, ram=8),
    )

    candidate = recommendations[0].candidates[0]
    joined = " ".join(candidate.hardware_reasons)
    assert "CPU" in joined
    assert "4.8GB" in joined
    assert "GPU 显存满足" not in joined


def test_quality_and_speed_preferences_change_text_model_ranking():
    analysis = classify_task("总结一篇很长的中文文章")
    inventory = _inventory(gpu_vram=12, ram=24)

    speed = recommend_models(analysis, inventory, quality="speed")
    quality = recommend_models(analysis, inventory, quality="quality")

    assert speed[0].candidates[0].parameter_size == "1B"
    assert quality[0].candidates[0].model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert speed[0].candidates[0].model_id != quality[0].candidates[0].model_id


def test_image_generation_preferences_change_ranking_and_score_profile():
    analysis = classify_task("根据文字描述生成一张图片")
    inventory = _inventory(gpu_vram=12, ram=24)

    speed = recommend_models(
        analysis,
        inventory,
        quality="speed",
        max_candidates_per_stage=10,
    )[0].candidates
    balanced = recommend_models(
        analysis,
        inventory,
        quality="balanced",
        max_candidates_per_stage=10,
    )[0].candidates
    quality = recommend_models(
        analysis,
        inventory,
        quality="quality",
        max_candidates_per_stage=10,
    )[0].candidates

    assert speed[0].model_id == "runwayml/stable-diffusion-v1-5"
    assert quality[0].model_id == "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
    assert quality[0].quantization == "INT8 (bitsandbytes)"
    assert quality[0].fit_tier == "constrained"
    assert speed[0].model_id != quality[0].model_id
    assert speed[0].score_breakdown["weights"] != quality[0].score_breakdown["weights"]
    assert balanced[0].preference == "balanced"
    assert all(candidate.ranking_reasons for candidate in quality)


def test_recommends_each_stage_for_meeting_video_pipeline():
    recommendations = recommend_models(
        classify_task("把会议视频转成文字，区分说话人，然后生成中文摘要"),
        _inventory(),
    )

    assert [item.stage.capability for item in recommendations] == [
        "extract_audio",
        "speech_to_text",
        "speaker_diarization",
        "summarization",
    ]
    assert [item.candidates[0].model_id for item in recommendations] == [
        "ffmpeg",
        "Systran/faster-whisper-base",
        "pyannote/speaker-diarization-3.1",
        "google/gemma-3-4b-it",
    ]


def test_marks_video_generation_as_remote_when_8gb_is_insufficient():
    recommendations = recommend_models(
        classify_task("根据文字描述生成一段视频"),
        _inventory(),
    )

    stage = recommendations[0]
    assert stage.local_supported is False
    assert all(candidate.fit_tier == "unsupported" for candidate in stage.candidates)
    assert [candidate.score for candidate in stage.candidates] == sorted(
        [candidate.score for candidate in stage.candidates],
        reverse=True,
    )
    assert all(candidate.score_breakdown for candidate in stage.candidates)
    assert "服务器" in stage.fallback


def test_cpu_only_machine_gets_cpu_capable_tts_model():
    recommendations = recommend_models(
        classify_task("把这段中文转成语音"),
        _inventory(gpu_vram=None, ram=8),
    )

    candidate = recommendations[0].candidates[0]
    assert candidate.model_id == "hexgrad/Kokoro-82M"
    assert candidate.fit_tier in {"ideal", "runnable"}
    assert recommendations[0].local_supported is True


def test_cpu_only_machine_uses_q3_to_fit_one_billion_parameter_model():
    recommendations = recommend_models(
        classify_task("总结这篇中文文章"),
        _inventory(gpu_vram=None, ram=8),
    )

    candidate = recommendations[0].candidates[0]
    assert candidate.parameter_size == "1B"
    assert candidate.quantization == "Q3_K_M"
    assert candidate.execution_mode == "cpu_only"
    assert candidate.fit_tier == "constrained"
    assert recommendations[0].local_supported is True


def test_recommendation_explains_quantization_resource_reduction():
    model = ModelSpec(
        model_id="example/video-model",
        display_name="Example Video Model",
        description="test",
        capabilities=["video_generation"],
        modalities=["video"],
        runtimes=["diffusers"],
        min_vram_gb=8,
        recommended_vram_gb=12,
        estimated_vram_gb=12,
        min_ram_gb=16,
        parameter_size="4B",
        requires_gpu=True,
        supports_cpu=False,
    )

    recommendation = recommend_models(
        classify_task("根据文字描述生成一段视频"),
        _inventory(gpu_vram=9, ram=32),
        catalog=[model],
        max_candidates_per_stage=1,
        allow_offload=False,
    )[0]
    candidate = recommendation.candidates[0]

    assert candidate.fit_tier == "runnable"
    assert candidate.quantization == "INT4 (bitsandbytes)"
    assert any(
        "原始配置约 12.0GB" in reason and "量化后约 6.2GB" in reason
        for reason in candidate.hardware_reasons
    )
    assert any("组件兼容性" in warning for warning in candidate.warnings)
    assert any("48%" in tradeoff for tradeoff in candidate.tradeoffs)

