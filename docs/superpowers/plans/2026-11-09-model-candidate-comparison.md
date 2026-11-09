# Model Candidate Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return multiple concrete model configurations per task stage with hardware-aware explanations and tradeoff comparisons.

**Architecture:** Extend the existing catalog and response models without changing the API path or request format. The matcher combines static model requirements with the detected inventory to produce resource fields, fit explanations, warnings, and tradeoffs.

**Tech Stack:** Python 3.12, Pydantic 2, LangGraph 1.x, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-11-09-model-candidate-comparison-design.md`

## Global Constraints

- Existing API request and route remain backward compatible.
- No model weights are downloaded or executed.
- Catalog entries use concrete model identifiers and conservative resource estimates.
- Every candidate explains its relationship to detected RAM, total VRAM, free VRAM, operating system, and runtime.
- Existing 24 tests remain green.

## Review Focus

- A stage with three catalog matches must return three distinct concrete model IDs.
- A candidate exceeding total VRAM must explain why it is constrained or unsupported.
- A candidate fitting total but not free VRAM must tell the user to release GPU memory.
- CPU-only candidates must not emit misleading GPU-fit explanations.
- API serialization must expose all new comparison fields.

---

### Task 1: Candidate comparison schema and hardware reasoning

**Files:**
- Modify: `backend/agents/model_selector/state.py`
- Modify: `backend/agents/model_selector/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `ModelSpec`, `HardwareInventory`.
- Produces: extended `CandidateRecommendation` fields and deterministic hardware explanations.

- [ ] Add failing tests for resource fields, free-VRAM constraints, CPU-only explanations, and tradeoffs.
- [ ] Run targeted tests and verify RED.
- [ ] Extend the response schema and matcher.
- [ ] Run targeted tests and verify GREEN.

### Task 2: Expand the multimodal catalog

**Files:**
- Modify: `backend/agents/model_selector/models.json`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: existing catalog loader.
- Produces: at least three candidates for major model capabilities where practical.

- [ ] Add failing catalog-coverage and ranking tests for text, ASR, image generation, vision, TTS, and video generation.
- [ ] Verify concrete model identifiers and published capability descriptions against primary model repositories/cards.
- [ ] Add conservative model configurations and deployment commands.
- [ ] Run matcher tests and verify GREEN.

### Task 3: API response and documentation

**Files:**
- Modify: `tests/test_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: extended recommendation models.
- Produces: backward-compatible Swagger response with multiple comparable candidates.

- [ ] Add a failing API test asserting multiple candidates and new comparison fields.
- [ ] Update examples and field descriptions.
- [ ] Run API and full-suite tests.
- [ ] Run CLI and real Uvicorn HTTP smoke tests.

