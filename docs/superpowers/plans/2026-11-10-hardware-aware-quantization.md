# Hardware-aware Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dynamically select and explain a model-specific quantization strategy that makes the best feasible use of detected hardware.

**Architecture:** Add a focused quantization strategy module that produces runtime/capability-specific execution variants. The existing fit engine evaluates those variants and the matcher exposes the selected strategy and resource reduction through the existing response model.

**Tech Stack:** Python, Pydantic, LangGraph, FastAPI, vanilla JavaScript, pytest

**Spec:** `docs/superpowers/specs/2026-11-10-hardware-aware-quantization-design.md`

## Global Constraints

- Preserve the existing API field names.
- Catalog-authored variants take precedence over inferred variants.
- Inferred estimates must use `heuristic` confidence.
- Do not apply text-only GGUF strategies to diffusion or media models.

## Review Focus

- A full-precision model that does not fit but INT4 does fit must be recommended as runnable.
- A model with explicit catalog variants must not receive conflicting inferred variants.
- CPU-only tools must receive a meaningful non-empty strategy.
- The selected strategy must describe its backend and estimated resource reduction.
- Existing ranking preferences must continue to operate on the selected variant.

---

### Task 1: Quantization strategy generation

**Files:**
- Create: `backend/agents/model_selector/quantization.py`
- Modify: `backend/agents/model_selector/fit_engine.py`
- Test: `tests/test_fit_engine.py`

**Interfaces:**
- Produces: `infer_execution_variants(model: ModelSpec, base: ExecutionVariant) -> list[ExecutionVariant]`
- Consumes: `ModelSpec`, `ExecutionVariant`

- [ ] Write failing tests for Diffusers INT4 fit, explicit variant precedence, Faster Whisper strategy, and CPU/tool fallback.
- [ ] Run the focused tests and confirm failures are caused by missing inferred variants.
- [ ] Implement family-specific strategy generation and integrate it into `build_execution_variants`.
- [ ] Run focused tests and confirm they pass.

### Task 2: Recommendation explanations and UI

**Files:**
- Modify: `backend/agents/model_selector/matcher.py`
- Modify: `backend/static/app.js`
- Test: `tests/test_matcher.py`
- Test: `tests/test_frontend.py`

**Interfaces:**
- Consumes: selected variant metadata from Task 1.
- Produces: user-visible strategy, resource-reduction explanation, and “量化策略” card label.

- [ ] Write failing matcher and frontend tests.
- [ ] Add selected-strategy reasoning and non-empty display behavior.
- [ ] Run focused tests.
- [ ] Run the complete test suite and JavaScript syntax check.
