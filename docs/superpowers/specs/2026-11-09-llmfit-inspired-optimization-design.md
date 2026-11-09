# LLMFit-inspired Model Recommendation Optimization

## Intent

Upgrade the current hardware-aware LangGraph Agent from fixed catalog thresholds to an explainable fit engine that can evaluate execution variants, simulate target machines, expose estimation confidence, and optionally reuse llmfit for text-model ranking.

## Chosen Architecture

Use a hybrid design:

- LangGraph remains responsible for natural-language task classification, multimodal pipeline decomposition, environment routing, and deployment planning.
- A local Python fit engine remains the source of truth for image, audio, video, tools, and fallback text recommendations.
- An optional llmfit REST adapter augments text generation, summarization, translation, coding, and embedding stages when `LLMFIT_BASE_URL` is configured and reachable.
- The adapter is fail-open: any timeout, invalid payload, or unavailable sidecar falls back to the local catalog without failing the recommendation request.

Rejected alternatives:

- Porting the Rust llmfit core into Python would duplicate a large and fast-moving codebase.
- Sending every multimodal task to llmfit would collapse video generation and video understanding into an unsafe broad category.
- Making llmfit mandatory would break the current zero-extra-service local demo.

## Goals

1. Generate and select explicit direct, quantized, and CPU-offload execution variants where the model supports them.
2. Compute fit from required resources divided by currently available resources, while preserving OS/runtime compatibility checks.
3. Return estimate confidence, calculation basis, utilization ratios, candidate source, and selected execution mode.
4. Allow API callers and the frontend to simulate a target workstation, server, or multi-GPU node without changing the actual machine.
5. Optionally query llmfit `/api/v1/models/top` for text-oriented stages and merge its candidates with local results.
6. Keep at least ten local catalog candidates for each primary capability family: text generation, summarization, translation, code, image understanding, video understanding, speech recognition, speech synthesis, image generation, and video generation.

## Fit Semantics

- `ideal`: peak utilization is at most 60%.
- `runnable`: peak utilization is above 60% and at most 85%.
- `constrained`: peak utilization is above 85% and at most 98%, or requires an explicit offload variant.
- `unsupported`: resource utilization exceeds 98%, the OS is incompatible, or a required GPU/runtime path does not exist.

Available memory is used for fit calculations. Total memory remains visible as capacity context.

## API Changes

`RecommendationRequest` gains:

- `preferences.allow_offload: bool = true`
- `preferences.use_llmfit: bool = true`
- `hardware_override: HardwareOverride | null`

`CandidateRecommendation` gains:

- `source`
- `execution_mode`
- `utilization`
- `estimate_confidence`
- `estimate_basis`
- `selected_variant`

The changes are additive; existing request bodies remain valid.

## Hardware Simulation

The backend merges a validated override onto detected hardware. The response returns the effective hardware and marks `metadata.hardware_source` as `detected` or `simulated`.

The first frontend version provides presets:

- Current machine
- 16GB single-GPU server
- 24GB single-GPU server
- 2×48GB server

The API supports explicit custom values for later UI expansion.

## llmfit Adapter

- Base URL comes from `LLMFIT_BASE_URL`.
- Only text generation, summarization, translation, code generation, and embedding stages are eligible.
- Use-case mapping: general, coding, or embedding.
- Request top runnable models with a short timeout.
- Parse only stable documented fields and ignore unknown fields.
- Simulated hardware uses the local engine because a remote llmfit node describes its own hardware.

## Frontend

Add a target-environment selector to the task console. Candidate cards show:

- candidate source;
- selected execution mode and quantization;
- VRAM/RAM utilization;
- estimate confidence;
- a compact explanation of the calculation basis.

The candidate-count selector supports up to ten results per stage. The
default remains three so ordinary responses stay concise.

## Catalog Coverage

The bundled catalog is the deterministic offline source of truth. It is split
into a compact core file and an extended file, loaded and validated as one
catalog. Each of the ten primary capability families has at least ten distinct
model IDs before optional llmfit candidates are merged.

The minimum is a catalog-availability guarantee, not a promise that every
machine can run all ten models. Hardware fit, operating-system compatibility,
runtime availability, and offload rules still determine the returned ranking
and support status.

## Non-goals

- Installing or starting llmfit automatically.
- Downloading or launching recommended models.
- Claiming benchmark accuracy for catalog or heuristic estimates.
- Replacing modality-specific video/image/audio estimators with token-oriented LLM formulas.

