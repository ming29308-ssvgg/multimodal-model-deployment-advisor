import asyncio

import httpx

from backend.main import app


async def _get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        return await client.get(path)


def test_root_serves_model_navigator_page():
    response = asyncio.run(_get("/"))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "MODEL//NAV" in response.text
    assert 'id="task-input"' in response.text
    assert 'href="/static/app.css"' in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'id="target-environment"' in response.text
    assert 'id="allow-offload"' in response.text
    assert 'id="use-llmfit"' in response.text
    assert '<option value="10">10 个</option>' in response.text
    assert "candidate-source" in response.text
    assert "candidate-confidence" in response.text
    assert "candidate-score" in response.text
    assert "candidate-preference" in response.text
    assert "score-breakdown" in response.text
    assert "score-row-template" in response.text
    assert "ranking-reason-list" in response.text
    assert 'id="task-templates"' in response.text
    assert 'role="tablist"' in response.text
    for category in ["text", "image", "audio", "video", "multimodal"]:
        assert f'data-template-tab="{category}"' in response.text
    for template_name in [
        "知识库问答",
        "图片文字识别",
        "语音转写",
        "音频内容理解",
        "多说话人处理",
        "视频生成",
        "视频语音摘要",
    ]:
        assert template_name in response.text
    assert "会议音频" not in response.text


def test_static_frontend_assets_are_served():
    css_response = asyncio.run(_get("/static/app.css"))
    js_response = asyncio.run(_get("/static/app.js"))

    assert css_response.status_code == 200
    assert css_response.headers["content-type"].startswith("text/css")
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert 'addSpec(specs, "环境依赖", candidate.recommended_runtime)' in js_response.text
    assert 'addSpec(specs, "运行时", candidate.recommended_runtime)' not in js_response.text
    assert 'addSpec(specs, "量化策略", candidate.quantization || "无需额外量化")' in js_response.text
    assert 'addSpec(specs, "量化", candidate.quantization)' not in js_response.text


def test_swagger_docs_remain_available():
    response = asyncio.run(_get("/docs"))

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()
