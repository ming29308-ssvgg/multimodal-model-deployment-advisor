from types import SimpleNamespace

from backend.agents.model_selector import hardware
from backend.agents.model_selector.hardware import (
    classify_environment,
    detect_runtimes,
    parse_nvidia_smi_output,
)


def test_parses_nvidia_smi_csv_output():
    output = (
        "NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 6503 MiB, "
        "595.97, 8.9\n"
    )

    devices = parse_nvidia_smi_output(output)

    assert len(devices) == 1
    assert devices[0].name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert devices[0].total_vram_gb == 8.0
    assert devices[0].free_vram_gb == 6.35
    assert devices[0].driver_version == "595.97"
    assert devices[0].compute_capability == "8.9"


def test_classifies_personal_computer_server_and_cluster():
    assert classify_environment(gpu_count=1, total_ram_gb=16, kubernetes=False) == "personal_computer"
    assert classify_environment(gpu_count=4, total_ram_gb=128, kubernetes=False) == "single_server"
    assert classify_environment(gpu_count=1, total_ram_gb=32, kubernetes=True) == "cluster"


def test_detects_installed_runtimes_from_path_lookup():
    installed = {"docker", "ollama", "ffmpeg"}

    runtimes = detect_runtimes(which=lambda name: f"C:/{name}.exe" if name in installed else None)

    assert runtimes["docker"] is True
    assert runtimes["ollama"] is True
    assert runtimes["ffmpeg"] is True
    assert runtimes["kubectl"] is False
    assert runtimes["vllm"] is False
    assert runtimes["python"] is True


def test_windows_cpu_and_architecture_use_fallbacks(monkeypatch):
    monkeypatch.setattr(
        hardware.platform,
        "processor",
        lambda: "Intel64 Family 6 Model 186 Stepping 2, GenuineIntel",
    )
    monkeypatch.setattr(hardware.platform, "machine", lambda: "")
    monkeypatch.setattr(
        hardware,
        "_read_windows_cpu_model",
        lambda: "13th Gen Intel(R) Core(TM) i7-13700H",
    )
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")

    assert hardware._detect_cpu_model("Windows") == "13th Gen Intel(R) Core(TM) i7-13700H"
    assert hardware._detect_architecture() == "AMD64"


def test_detect_environment_degrades_to_cpu_only(monkeypatch):
    monkeypatch.setattr(hardware, "_query_nvidia_gpus", lambda: ([], "nvidia-smi 不可用"))
    monkeypatch.setattr(
        hardware,
        "detect_runtimes",
        lambda: {
            "docker": False,
            "ollama": False,
            "kubectl": False,
            "ffmpeg": True,
            "llama_cpp": False,
            "vllm": False,
        },
    )
    monkeypatch.setattr(
        hardware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
    )
    monkeypatch.setattr(hardware.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(
        hardware.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(total=500 * 1024**3, used=300 * 1024**3, free=200 * 1024**3),
    )
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.8")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "Test CPU")
    monkeypatch.setattr(hardware, "_read_windows_cpu_model", lambda: None)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    inventory = hardware.detect_environment()

    assert inventory.gpus == []
    assert inventory.environment_type == "personal_computer"
    assert inventory.ram_total_gb == 16.0
    assert inventory.runtimes["ffmpeg"] is True
    assert inventory.detection_warnings == ["nvidia-smi 不可用"]

