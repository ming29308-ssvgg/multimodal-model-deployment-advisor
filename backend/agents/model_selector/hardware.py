import csv
import io
import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import psutil

from .state import EnvironmentType, GPUDevice, HardwareInventory


def _mib_to_gib(value: str) -> float:
    number = float(value.lower().replace("mib", "").strip())
    return round(number / 1024, 2)


def parse_nvidia_smi_output(output: str) -> list[GPUDevice]:
    devices: list[GPUDevice] = []
    for row in csv.reader(io.StringIO(output.strip())):
        if not row or len(row) < 5:
            continue
        name, total, free, driver, compute = (part.strip() for part in row[:5])
        devices.append(
            GPUDevice(
                name=name,
                total_vram_gb=_mib_to_gib(total),
                free_vram_gb=_mib_to_gib(free),
                driver_version=driver or None,
                compute_capability=compute or None,
            )
        )
    return devices


def _query_nvidia_gpus() -> tuple[list[GPUDevice], str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], "nvidia-smi 不可用"

    command = [
        executable,
        "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi 执行失败: {exc}"

    if completed.returncode != 0:
        message = completed.stderr.strip() or f"退出码 {completed.returncode}"
        return [], f"nvidia-smi 执行失败: {message}"

    try:
        return parse_nvidia_smi_output(completed.stdout), None
    except (TypeError, ValueError) as exc:
        return [], f"nvidia-smi 输出解析失败: {exc}"


def detect_runtimes(
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, bool]:
    return {
        "python": True,
        "docker": which("docker") is not None,
        "ollama": which("ollama") is not None,
        "kubectl": which("kubectl") is not None,
        "ffmpeg": which("ffmpeg") is not None,
        "llama_cpp": which("llama-server") is not None or which("llama-cli") is not None,
        "vllm": which("vllm") is not None,
    }


def classify_environment(
    *,
    gpu_count: int,
    total_ram_gb: float,
    kubernetes: bool,
    logical_cores: int = 1,
    containerized: bool = False,
) -> EnvironmentType:
    if kubernetes:
        return "cluster"
    if gpu_count > 1 or total_ram_gb >= 48 or logical_cores >= 24 or containerized:
        return "single_server"
    return "personal_computer"


def _is_containerized() -> bool:
    if Path("/.dockerenv").exists():
        return True
    if platform.system() != "Linux":
        return False
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(marker in cgroup for marker in ("docker", "containerd", "kubepods"))


def _read_windows_cpu_model() -> str | None:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        try:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        finally:
            winreg.CloseKey(key)
        return str(value).strip() or None
    except (ImportError, OSError):
        return None


def _detect_cpu_model(os_name: str) -> str:
    if os_name == "Windows":
        registry_value = _read_windows_cpu_model()
        if registry_value:
            return registry_value
    detected = platform.processor().strip()
    if detected:
        return detected
    return os.environ.get("PROCESSOR_IDENTIFIER") or "unknown"


def _detect_architecture() -> str:
    return (
        platform.machine().strip()
        or os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or platform.architecture()[0]
        or "unknown"
    )


def detect_environment() -> HardwareInventory:
    memory = psutil.virtual_memory()
    logical_cores = psutil.cpu_count(logical=True) or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores
    disk = shutil.disk_usage(Path.cwd().anchor or Path.cwd())
    gpus, gpu_warning = _query_nvidia_gpus()
    runtimes = detect_runtimes()

    kubernetes = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
    containerized = _is_containerized()
    total_ram_gb = round(memory.total / 1024**3, 2)
    environment_type = classify_environment(
        gpu_count=len(gpus),
        total_ram_gb=total_ram_gb,
        kubernetes=kubernetes,
        logical_cores=logical_cores,
        containerized=containerized,
    )

    warnings = [gpu_warning] if gpu_warning else []
    os_name = platform.system() or "unknown"

    return HardwareInventory(
        os_name=os_name,
        os_version=platform.release() or "unknown",
        architecture=_detect_architecture(),
        cpu_model=_detect_cpu_model(os_name),
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        ram_total_gb=total_ram_gb,
        ram_available_gb=round(memory.available / 1024**3, 2),
        disk_free_gb=round(disk.free / 1024**3, 2),
        gpus=gpus,
        runtimes=runtimes,
        environment_type=environment_type,
        containerized=containerized,
        kubernetes=kubernetes,
        detection_warnings=warnings,
        metadata={
            "hostname": platform.node(),
            "hardware_source": "detected",
        },
    )


def apply_hardware_override(
    inventory: HardwareInventory,
    override: dict | None,
) -> HardwareInventory:
    if not override:
        metadata = dict(inventory.metadata)
        metadata.setdefault("hardware_source", "detected")
        return inventory.model_copy(update={"metadata": metadata})

    data = inventory.model_dump()
    scalar_fields = [
        "os_name",
        "os_version",
        "architecture",
        "cpu_model",
        "physical_cores",
        "logical_cores",
        "ram_total_gb",
        "ram_available_gb",
        "disk_free_gb",
        "environment_type",
        "runtimes",
    ]
    for field in scalar_fields:
        if override.get(field) is not None:
            data[field] = override[field]

    gpu_fields_present = any(
        override.get(field) is not None
        for field in [
            "gpu_count",
            "gpu_name",
            "gpu_total_vram_gb",
            "gpu_free_vram_gb",
        ]
    )
    if gpu_fields_present:
        existing_gpu = inventory.gpus[0] if inventory.gpus else None
        gpu_count = int(
            override.get("gpu_count")
            if override.get("gpu_count") is not None
            else len(inventory.gpus)
        )
        total_vram = float(
            override.get("gpu_total_vram_gb")
            if override.get("gpu_total_vram_gb") is not None
            else (existing_gpu.total_vram_gb if existing_gpu else 0)
        )
        free_vram = float(
            override.get("gpu_free_vram_gb")
            if override.get("gpu_free_vram_gb") is not None
            else (existing_gpu.free_vram_gb if existing_gpu else total_vram)
        )
        free_vram = min(free_vram, total_vram)
        gpu_name = str(
            override.get("gpu_name")
            or (existing_gpu.name if existing_gpu else "Simulated GPU")
        )
        gpu_vendor = str(override.get("gpu_vendor") or "nvidia")
        data["gpus"] = [
            GPUDevice(
                name=gpu_name,
                vendor=gpu_vendor,
                total_vram_gb=total_vram,
                free_vram_gb=free_vram,
            ).model_dump()
            for _ in range(gpu_count)
        ]

    if override.get("environment_type") is None:
        data["environment_type"] = classify_environment(
            gpu_count=len(data["gpus"]),
            total_ram_gb=float(data["ram_total_gb"]),
            kubernetes=bool(data["kubernetes"]),
            logical_cores=int(data["logical_cores"]),
            containerized=bool(data["containerized"]),
        )

    metadata = dict(data.get("metadata") or {})
    metadata["hardware_source"] = "simulated"
    if override.get("name"):
        metadata["simulation_name"] = str(override["name"])
    data["metadata"] = metadata
    return HardwareInventory.model_validate(data)

