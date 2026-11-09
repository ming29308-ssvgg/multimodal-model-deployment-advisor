import pytest

from backend.agents.model_selector.task_classifier import classify_task


def test_classifies_text_summary_task():
    analysis = classify_task("总结这篇中文文章并列出三个关键结论")

    assert analysis.primary_modality == "text"
    assert analysis.modalities == ["text"]
    assert [stage.capability for stage in analysis.stages] == ["summarization"]


def test_decomposes_local_knowledge_base_question_answering():
    analysis = classify_task("部署一个本地中文知识库问答系统")

    assert analysis.primary_modality == "text"
    assert [stage.capability for stage in analysis.stages] == [
        "embedding",
        "reranking",
        "text_generation",
    ]


def test_classifies_image_generation_task():
    analysis = classify_task("根据这段描述生成一张产品宣传图片")

    assert analysis.primary_modality == "image"
    assert [stage.capability for stage in analysis.stages] == ["image_generation"]


def test_classifies_text_to_speech_task():
    analysis = classify_task("把这段中文转换成自然的语音")

    assert analysis.primary_modality == "audio"
    assert [stage.capability for stage in analysis.stages] == ["text_to_speech"]


def test_classifies_video_understanding_task():
    analysis = classify_task("理解这个视频发生了什么并回答问题")

    assert analysis.primary_modality == "video"
    assert [stage.capability for stage in analysis.stages] == ["video_understanding"]


def test_classifies_local_text_to_video_model_deployment():
    analysis = classify_task("我想在本地部署一个文生视频模型")

    assert analysis.primary_modality == "video"
    assert [stage.capability for stage in analysis.stages] == ["video_generation"]
    assert analysis.stages[0].input_modality == "text"
    assert analysis.stages[0].output_modality == "video"


def test_decomposes_meeting_video_into_pipeline():
    analysis = classify_task("把会议视频转成文字，区分说话人，然后生成中文摘要")

    assert analysis.primary_modality == "video"
    assert analysis.modalities == ["video", "audio", "text"]
    assert [stage.capability for stage in analysis.stages] == [
        "extract_audio",
        "speech_to_text",
        "speaker_diarization",
        "summarization",
    ]
    assert [stage.kind for stage in analysis.stages] == [
        "tool",
        "model",
        "model",
        "model",
    ]


def test_decomposes_general_multi_speaker_audio_into_pipeline():
    analysis = classify_task("把一段播客录音转成文字，区分说话人并生成摘要")

    assert analysis.primary_modality == "audio"
    assert analysis.modalities == ["audio", "text"]
    assert [stage.capability for stage in analysis.stages] == [
        "speech_to_text",
        "speaker_diarization",
        "summarization",
    ]


def test_rejects_empty_task():
    with pytest.raises(ValueError, match="任务描述不能为空"):
        classify_task("   ")

