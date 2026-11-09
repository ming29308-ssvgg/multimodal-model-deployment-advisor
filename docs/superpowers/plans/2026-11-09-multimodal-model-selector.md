# Multimodal Model Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable LangGraph agent that understands a natural-language multimodal task, detects the current computer/server environment, and recommends compatible models and deployment strategies.

**Architecture:** A deterministic LangGraph workflow separates task classification, environment detection, environment-specific analysis, catalog matching, and plan generation. Hardware detection and scoring are pure or dependency-injected where possible so the core behavior remains testable without specific hardware.

**Tech Stack:** Python 3.12, LangGraph 1.x, FastAPI, Pydantic 2, psutil, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-multimodal-model-selector-design.md`

## Global Constraints

- The API accepts a task description and optional preferences; it never accepts hardware specifications.
- The graph and tests run without API keys and without downloading model weights.
- NVIDIA probing uses `nvidia-smi` when available and degrades safely when it is not.
- The catalog is local JSON and covers text, image, audio, video, and multimodal capabilities.
- Current-machine acceptance baseline is Windows 11, Python 3.12, RTX 4060 Laptop 8GB, and 16GB RAM.

## Review Focus

- Empty or whitespace-only tasks must return a validation error.
- Missing `nvidia-smi` must produce a valid CPU-only inventory.
- A complex meeting-video task must become a multi-stage pipeline.
- A model whose RAM or VRAM requirement exceeds the machine must not be marked `ideal`.
- When no local candidate fits, the plan must recommend a larger server or remote deployment instead of returning an empty explanation.

---

### Task 1: Project foundation and task classification

**Files:**
- Create: `pyproject.toml`
- Create: `backend/__init__.py`
- Create: `backend/agents/model_selector/__init__.py`
- Create: `backend/agents/model_selector/state.py`
- Create: `backend/agents/model_selector/task_classifier.py`
- Test: `tests/test_task_classifier.py`

**Interfaces:**
- Produces: `classify_task(task: str) -> TaskAnalysis`.
- Produces: Pydantic models `TaskAnalysis` and `TaskStage`.

- [ ] Write tests for text, image, audio, video, complex meeting-video, and empty task behavior.
- [ ] Run `pytest tests/test_task_classifier.py -v` and verify failure because the package does not exist.
- [ ] Implement the minimal classifier and models.
- [ ] Run `pytest tests/test_task_classifier.py -v` and verify all tests pass.

### Task 2: Automatic environment and hardware detection

**Files:**
- Create: `backend/agents/model_selector/hardware.py`
- Test: `tests/test_hardware.py`

**Interfaces:**
- Consumes: no user-supplied hardware.
- Produces: `detect_environment() -> HardwareInventory`.
- Produces: `parse_nvidia_smi_output(output: str) -> list[GPUDevice]`.

- [ ] Write tests for NVIDIA CSV parsing, CPU-only fallback, runtime discovery, and environment classification.
- [ ] Run `pytest tests/test_hardware.py -v` and verify failure for missing implementation.
- [ ] Implement detection with `psutil`, `platform`, `shutil.which`, and optional `nvidia-smi`.
- [ ] Run `pytest tests/test_hardware.py -v` and verify all tests pass.

### Task 3: Catalog and resource-aware matching

**Files:**
- Create: `backend/agents/model_selector/models.json`
- Create: `backend/agents/model_selector/catalog.py`
- Create: `backend/agents/model_selector/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `TaskAnalysis`, `HardwareInventory`, and catalog records.
- Produces: `recommend_models(...) -> list[StageRecommendation]`.

- [ ] Write tests that select suitable 3B text, speech recognition, vision, image generation, and pipeline candidates on representative hardware.
- [ ] Add tests proving oversized video generation is constrained or remote-only and CPU-only machines receive viable alternatives.
- [ ] Run `pytest tests/test_matcher.py -v` and verify failure for missing implementation.
- [ ] Implement catalog loading, compatibility filtering, scoring, fit tiers, and per-stage ranking.
- [ ] Run `pytest tests/test_matcher.py -v` and verify all tests pass.

### Task 4: LangGraph workflow, deployment plan, API, CLI, and documentation

**Files:**
- Create: `backend/agents/model_selector/nodes.py`
- Create: `backend/agents/model_selector/graph.py`
- Create: `backend/schemas.py`
- Create: `backend/api/__init__.py`
- Create: `backend/api/model_selector.py`
- Create: `backend/main.py`
- Create: `backend/cli.py`
- Create: `tests/test_graph.py`
- Create: `tests/test_api.py`
- Create: `.env.example`
- Create: `README.md`

**Interfaces:**
- Consumes: classifier, detector, and matcher interfaces from Tasks 1–3.
- Produces: compiled `model_selector_graph`.
- Produces: `POST /api/v1/model-selector/recommend`.
- Produces: `python -m backend.cli "<task>"`.

- [ ] Write graph tests for local, server, and no-fit routes with deterministic injected inventories.
- [ ] Write API tests for valid and empty tasks.
- [ ] Run the new tests and verify expected failures.
- [ ] Implement nodes, conditional routing, plan generation, FastAPI endpoint, and CLI.
- [ ] Document setup, examples, architecture, current limitations, and extension points.
- [ ] Run `pytest -v`.
- [ ] Run a CLI smoke test on the current machine.
- [ ] Start the FastAPI app and run a request-level smoke test.
- [ ] Run `python -m compileall backend`.

