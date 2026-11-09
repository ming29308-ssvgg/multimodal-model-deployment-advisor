from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.model_selector import router as model_selector_router


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Multimodal Model Selector Agent",
    version="0.1.0",
    description="自动检测部署环境并为文本、图片、音频和视频任务推荐模型。",
)
app.include_router(model_selector_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

