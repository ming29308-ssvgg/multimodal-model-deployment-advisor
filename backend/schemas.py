from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .agents.model_selector.state import (
    HardwareInventory,
    StageRecommendation,
    TaskAnalysis,
)


class RecommendationPreferences(BaseModel):
    quality: Literal["speed", "balanced", "quality"] = "balanced"
    max_candidates_per_stage: int = Field(default=3, ge=1, le=10)
    allow_offload: bool = True
    use_llmfit: bool = True


class HardwareOverride(BaseModel):
    name: str | None = None
    os_name: str | None = None
    os_version: str | None = None
    architecture: str | None = None
    cpu_model: str | None = None
    physical_cores: int | None = Field(default=None, ge=1)
    logical_cores: int | None = Field(default=None, ge=1)
    ram_total_gb: float | None = Field(default=None, gt=0)
    ram_available_gb: float | None = Field(default=None, ge=0)
    disk_free_gb: float | None = Field(default=None, ge=0)
    gpu_count: int | None = Field(default=None, ge=0, le=16)
    gpu_name: str | None = None
    gpu_vendor: str = "nvidia"
    gpu_total_vram_gb: float | None = Field(default=None, ge=0)
    gpu_free_vram_gb: float | None = Field(default=None, ge=0)
    environment_type: Literal[
        "personal_computer",
        "single_server",
        "cluster",
    ] | None = None
    runtimes: dict[str, bool] | None = None

    @model_validator(mode="after")
    def validate_available_resources(self) -> "HardwareOverride":
        if (
            self.ram_total_gb is not None
            and self.ram_available_gb is not None
            and self.ram_available_gb > self.ram_total_gb
        ):
            raise ValueError("可用内存不能大于总内存")
        if (
            self.gpu_total_vram_gb is not None
            and self.gpu_free_vram_gb is not None
            and self.gpu_free_vram_gb > self.gpu_total_vram_gb
        ):
            raise ValueError("可用显存不能大于总显存")
        return self


class RecommendationRequest(BaseModel):
    task: str = Field(min_length=1)
    preferences: RecommendationPreferences = Field(
        default_factory=RecommendationPreferences
    )
    hardware_override: HardwareOverride | None = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("任务描述不能为空")
        return value


class DeploymentPlan(BaseModel):
    environment_summary: str
    execution_mode: Literal[
        "local_single_model",
        "local_pipeline",
        "hybrid_pipeline",
        "remote_or_larger_server",
    ]
    steps: list[str]
    warnings: list[str]
    next_actions: list[str]


class RecommendationResponse(BaseModel):
    task_analysis: TaskAnalysis
    hardware: HardwareInventory
    recommendations: list[StageRecommendation]
    deployment_plan: DeploymentPlan
    trace: list[str]

