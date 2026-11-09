from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Modality = Literal["text", "image", "audio", "video"]
StageKind = Literal["model", "tool"]


class TaskStage(BaseModel):
    name: str
    capability: str
    kind: StageKind = "model"
    input_modality: Modality
    output_modality: Modality


class TaskAnalysis(BaseModel):
    original_task: str
    primary_modality: Modality
    modalities: list[Modality]
    stages: list[TaskStage] = Field(min_length=1)


EnvironmentType = Literal["personal_computer", "single_server", "cluster"]
FitTier = Literal["ideal", "runnable", "constrained", "unsupported"]
ExecutionMode = Literal["direct", "quantized", "cpu_offload", "cpu_only"]
EstimateConfidence = Literal[
    "benchmarked",
    "calibrated",
    "catalog",
    "heuristic",
    "unknown",
]


class GPUDevice(BaseModel):
    name: str
    vendor: str = "nvidia"
    total_vram_gb: float = Field(ge=0)
    free_vram_gb: float = Field(ge=0)
    driver_version: str | None = None
    compute_capability: str | None = None


class HardwareInventory(BaseModel):
    os_name: str
    os_version: str
    architecture: str
    cpu_model: str
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    ram_total_gb: float = Field(ge=0)
    ram_available_gb: float = Field(ge=0)
    disk_free_gb: float = Field(ge=0)
    gpus: list[GPUDevice] = Field(default_factory=list)
    runtimes: dict[str, bool] = Field(default_factory=dict)
    environment_type: EnvironmentType
    containerized: bool = False
    kubernetes: bool = False
    detection_warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionVariantSpec(BaseModel):
    id: str
    label: str
    mode: ExecutionMode
    quantization: str | None = None
    estimated_vram_gb: float = Field(default=0, ge=0)
    min_ram_gb: float = Field(default=1, ge=0)
    quality_rank: int = Field(default=50, ge=0, le=100)
    settings: dict[str, Any] = Field(default_factory=dict)


class ExecutionVariant(ExecutionVariantSpec):
    confidence: EstimateConfidence = "catalog"


class ResourceUtilization(BaseModel):
    required_vram_gb: float = 0
    available_vram_gb: float = 0
    vram_ratio: float | None = None
    required_ram_gb: float = 0
    available_ram_gb: float = 0
    ram_ratio: float | None = None
    peak_ratio: float | None = None


class FitEstimate(BaseModel):
    variant: ExecutionVariant
    fit_tier: FitTier
    utilization: ResourceUtilization
    confidence: EstimateConfidence
    basis: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    display_name: str
    kind: Literal["model", "tool"] = "model"
    description: str
    capabilities: list[str] = Field(min_length=1)
    modalities: list[Modality] = Field(min_length=1)
    runtimes: list[str] = Field(min_length=1)
    supported_os: list[str] = Field(default_factory=lambda: ["windows", "linux", "darwin"])
    min_vram_gb: float = Field(default=0, ge=0)
    recommended_vram_gb: float = Field(default=0, ge=0)
    estimated_vram_gb: float | None = Field(default=None, ge=0)
    min_ram_gb: float = Field(default=1, ge=0)
    quantization: str | None = None
    parameter_size: str | None = None
    requires_gpu: bool = False
    supports_cpu: bool = True
    supports_multi_gpu: bool = False
    requires_token: bool = False
    catalog_priority: int = Field(default=0, ge=0, le=50)
    command_templates: dict[str, str] = Field(default_factory=dict)
    execution_variants: list[ExecutionVariantSpec] = Field(default_factory=list)


class CandidateRecommendation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    display_name: str
    score: int
    fit_tier: FitTier
    recommended_runtime: str
    deployment_command: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    min_vram_gb: float = 0
    recommended_vram_gb: float = 0
    estimated_vram_gb: float = 0
    min_ram_gb: float = 0
    hardware_reasons: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source: Literal["local_catalog", "llmfit"] = "local_catalog"
    execution_mode: ExecutionMode = "direct"
    utilization: dict[str, Any] = Field(default_factory=dict)
    estimate_confidence: EstimateConfidence = "unknown"
    estimate_basis: dict[str, Any] = Field(default_factory=dict)
    selected_variant: dict[str, Any] = Field(default_factory=dict)
    preference: Literal["speed", "balanced", "quality"] = "balanced"
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    ranking_reasons: list[str] = Field(default_factory=list)


class StageRecommendation(BaseModel):
    stage: TaskStage
    candidates: list[CandidateRecommendation] = Field(default_factory=list)
    local_supported: bool
    fallback: str



