import json
import os
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .state import (
    CandidateRecommendation,
    HardwareInventory,
    TaskStage,
)


FetchJson = Callable[[str, float], dict[str, Any]]

USE_CASES = {
    "text_generation": "general",
    "summarization": "general",
    "translation": "general",
    "code_generation": "coding",
    "embedding": "embedding",
}

FIT_MAP = {
    "P": "ideal",
    "G": "runnable",
    "M": "constrained",
    "T": "unsupported",
    "U": "unsupported",
}

MODE_MAP = {
    "GPU": "direct",
    "CPU": "cpu_only",
    "HYBRID": "cpu_offload",
    "MOE": "cpu_offload",
}

CONFIDENCE_MAP = {
    "high": "calibrated",
    "medium": "heuristic",
    "low": "heuristic",
    "unknown": "unknown",
}


def _default_fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "model-nav/0.2",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class LlmfitClient:
    def __init__(
        self,
        base_url: str | None,
        *,
        timeout: float = 1.5,
        fetch_json: FetchJson = _default_fetch_json,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.fetch_json = fetch_json

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @classmethod
    def from_env(cls) -> "LlmfitClient":
        timeout_text = os.environ.get("LLMFIT_TIMEOUT_SECONDS", "1.5")
        try:
            timeout = max(float(timeout_text), 0.1)
        except ValueError:
            timeout = 1.5
        return cls(
            os.environ.get("LLMFIT_BASE_URL"),
            timeout=timeout,
        )

    def recommend(
        self,
        stage: TaskStage,
        inventory: HardwareInventory,
        *,
        limit: int,
    ) -> list[CandidateRecommendation]:
        use_case = USE_CASES.get(stage.capability)
        if (
            not self.enabled
            or not use_case
            or inventory.metadata.get("hardware_source") == "simulated"
        ):
            return []

        query = urlencode(
            {
                "use_case": use_case,
                "limit": max(1, min(limit, 10)),
            }
        )
        try:
            payload = self.fetch_json(
                f"{self.base_url}/api/v1/models/top?{query}",
                self.timeout,
            )
            rows = payload.get("models")
            if not isinstance(rows, list):
                return []
            return [
                candidate
                for row in rows
                if isinstance(row, dict)
                and (candidate := self._map_candidate(row, inventory)) is not None
            ]
        except (OSError, TimeoutError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return []

    def _map_candidate(
        self,
        row: dict[str, Any],
        inventory: HardwareInventory,
    ) -> CandidateRecommendation | None:
        name = str(row.get("name") or "").strip()
        if not name:
            return None
        model_id = str(row.get("id") or name)
        fit_tier = FIT_MAP.get(str(row.get("fit") or "U").upper(), "unsupported")
        run_mode = str(row.get("run_mode") or "GPU").upper()
        execution_mode = MODE_MAP.get(run_mode, "direct")
        memory_required = max(float(row.get("memory_required_gb") or 0), 0)
        memory_available = max(float(row.get("memory_available_gb") or 0), 0)
        utilization_pct = max(float(row.get("utilization_pct") or 0), 0)
        utilization_ratio = round(utilization_pct / 100, 4)
        runtimes = row.get("runtimes") or []
        runtime_names = [
            str(runtime)
            for runtime in runtimes
            if isinstance(runtime, (str, int, float))
        ]
        recommended_runtime = next(
            (
                runtime
                for runtime in runtime_names
                if inventory.runtimes.get(runtime, False)
            ),
            runtime_names[0] if runtime_names else "llmfit",
        )
        confidence_text = str(
            row.get("estimate_confidence") or "unknown"
        ).lower()
        confidence = CONFIDENCE_MAP.get(confidence_text, "unknown")
        score = int(round(float(row.get("score") or 0)))
        basis_detail = row.get("estimate_basis")

        return CandidateRecommendation(
            model_id=model_id,
            display_name=name,
            score=max(0, min(score, 100)),
            fit_tier=fit_tier,
            recommended_runtime=recommended_runtime,
            parameter_size=(
                str(row["parameter_count"])
                if row.get("parameter_count") is not None
                else None
            ),
            quantization=(
                str(row["best_quant"])
                if row.get("best_quant") is not None
                else None
            ),
            min_vram_gb=memory_required if execution_mode != "cpu_only" else 0,
            recommended_vram_gb=memory_required if execution_mode != "cpu_only" else 0,
            estimated_vram_gb=memory_required if execution_mode != "cpu_only" else 0,
            min_ram_gb=memory_required if execution_mode == "cpu_only" else 0,
            hardware_reasons=[
                f"llmfit 估算需要 {memory_required:.1f}GB，可用 "
                f"{memory_available:.1f}GB，利用率 {utilization_pct:.1f}%。",
                f"llmfit 推荐运行模式：{run_mode}。",
            ],
            tradeoffs=[
                "该候选来自可选 llmfit 服务，实际可用性取决于该服务的模型数据库与基准数据。"
            ],
            reasons=["由 llmfit 文本模型适配器返回。"],
            warnings=[] if fit_tier != "unsupported" else ["llmfit 判定该配置过紧或不可运行"],
            source="llmfit",
            execution_mode=execution_mode,
            utilization={
                "required_vram_gb": memory_required,
                "available_vram_gb": memory_available,
                "vram_ratio": utilization_ratio,
                "required_ram_gb": 0,
                "available_ram_gb": inventory.ram_available_gb,
                "ram_ratio": 0,
                "peak_ratio": utilization_ratio,
            },
            estimate_confidence=confidence,
            estimate_basis={
                "source": "llmfit",
                "detail": basis_detail,
            },
            selected_variant={
                "id": "llmfit-best",
                "label": f"llmfit {row.get('best_quant') or run_mode}",
                "mode": execution_mode,
                "quantization": row.get("best_quant"),
                "estimated_vram_gb": memory_required,
                "min_ram_gb": memory_required if execution_mode == "cpu_only" else 0,
                "quality_rank": score,
                "settings": {
                    "run_mode": run_mode,
                },
                "confidence": confidence,
            },
        )
