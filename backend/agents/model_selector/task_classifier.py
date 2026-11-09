from collections.abc import Iterable

from .state import TaskAnalysis, TaskStage


def _contains(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _stage(
    capability: str,
    input_modality: str,
    output_modality: str,
    *,
    kind: str = "model",
) -> TaskStage:
    names = {
        "extract_audio": "提取视频音轨",
        "speech_to_text": "语音识别",
        "speaker_diarization": "说话人分离",
        "summarization": "文本摘要",
        "text_to_speech": "语音合成",
        "audio_understanding": "音频理解",
        "video_understanding": "视频理解",
        "video_generation": "视频生成",
        "image_generation": "图片生成",
        "image_understanding": "图片理解",
        "ocr": "OCR 文字识别",
        "code_generation": "代码生成",
        "embedding": "文本向量化",
        "reranking": "检索重排序",
        "translation": "文本翻译",
        "text_generation": "文本生成",
    }
    return TaskStage(
        name=names[capability],
        capability=capability,
        kind=kind,
        input_modality=input_modality,
        output_modality=output_modality,
    )


def classify_task(task: str) -> TaskAnalysis:
    normalized = task.strip().lower()
    if not normalized:
        raise ValueError("任务描述不能为空")

    stages: list[TaskStage] = []

    has_video = _contains(normalized, ["视频", "video"])
    has_image = _contains(normalized, ["图片", "图像", "照片", "image", "photo"])
    has_audio = _contains(normalized, ["音频", "录音", "语音", "audio", "speech"])
    wants_text_from_speech = _contains(
        normalized,
        ["转成文字", "转文字", "转写", "字幕", "speech to text", "transcribe"],
    )
    wants_summary = _contains(
        normalized,
        ["总结", "摘要", "概括", "summary", "summarize"],
    )
    wants_diarization = _contains(
        normalized,
        ["区分说话人", "说话人分离", "多说话人", "diarization", "speaker"],
    )
    wants_knowledge_base = _contains(
        normalized,
        ["知识库", "文档问答", "检索问答", "rag"],
    )
    wants_video_generation = has_video and _contains(
        normalized,
        ["生成", "制作", "文生视频", "create", "generate", "text to video"],
    )

    if has_video and wants_text_from_speech:
        stages.extend(
            [
                _stage("extract_audio", "video", "audio", kind="tool"),
                _stage("speech_to_text", "audio", "text"),
            ]
        )
        if wants_diarization:
            stages.append(_stage("speaker_diarization", "audio", "text"))
        if wants_summary:
            stages.append(_stage("summarization", "text", "text"))
    elif wants_video_generation:
        stages.append(_stage("video_generation", "text", "video"))
    elif has_video:
        stages.append(_stage("video_understanding", "video", "text"))
    elif _contains(normalized, ["生成图片", "生成一张", "绘制", "文生图", "text to image"]):
        stages.append(_stage("image_generation", "text", "image"))
    elif has_image and _contains(normalized, ["ocr", "识别文字", "提取文字"]):
        stages.append(_stage("ocr", "image", "text"))
    elif has_image:
        stages.append(_stage("image_understanding", "image", "text"))
    elif _contains(normalized, ["转换成自然的语音", "转成语音", "语音合成", "朗读", "text to speech"]):
        stages.append(_stage("text_to_speech", "text", "audio"))
    elif has_audio and wants_text_from_speech:
        stages.append(_stage("speech_to_text", "audio", "text"))
        if wants_diarization:
            stages.append(_stage("speaker_diarization", "audio", "text"))
        if wants_summary:
            stages.append(_stage("summarization", "text", "text"))
    elif has_audio:
        stages.append(_stage("audio_understanding", "audio", "text"))
    elif wants_knowledge_base:
        stages.extend(
            [
                _stage("embedding", "text", "text"),
                _stage("reranking", "text", "text"),
                _stage("text_generation", "text", "text"),
            ]
        )
    elif _contains(
        normalized,
        ["代码", "编程", "编写", "开发", "code", "python", "java", "javascript"],
    ):
        stages.append(_stage("code_generation", "text", "text"))
    elif _contains(normalized, ["embedding", "嵌入", "向量化", "向量检索"]):
        stages.append(_stage("embedding", "text", "text"))
    elif _contains(normalized, ["rerank", "重排序", "重排"]):
        stages.append(_stage("reranking", "text", "text"))
    elif _contains(normalized, ["翻译", "translate", "translation"]):
        stages.append(_stage("translation", "text", "text"))
    elif wants_summary:
        stages.append(_stage("summarization", "text", "text"))
    else:
        stages.append(_stage("text_generation", "text", "text"))

    modality_order = ["video", "image", "audio", "text"]
    used_modalities = {
        modality
        for stage in stages
        for modality in (stage.input_modality, stage.output_modality)
    }
    modalities = [modality for modality in modality_order if modality in used_modalities]

    if has_video or any(stage.output_modality == "video" for stage in stages):
        primary_modality = "video"
    elif has_image or any(stage.output_modality == "image" for stage in stages):
        primary_modality = "image"
    elif has_audio or any(stage.output_modality == "audio" for stage in stages):
        primary_modality = "audio"
    else:
        primary_modality = "text"

    return TaskAnalysis(
        original_task=task.strip(),
        primary_modality=primary_modality,
        modalities=modalities,
        stages=stages,
    )

