# Hardware-aware quantization design

## Intent

Model recommendations must be based on the best deployable execution
strategy for the detected hardware, not only on the model's original
parameter size or full-precision memory estimate. The UI must always explain
the selected precision or quantization strategy.

## Design

- Keep the existing LangGraph workflow unchanged.
- Generate execution variants before fit evaluation.
- Prefer the highest-quality variant that fits available VRAM and RAM.
- Use runtime/capability-specific strategies:
  - Transformers language and understanding models: FP16, INT8, INT4.
  - Faster Whisper: FP16, INT8-FP16, INT8.
  - Diffusers image/video pipelines: FP16, INT8, INT4 where component
    quantization is supported; mark generated estimates as heuristic.
  - CPU tools and pipelines without a safe generic quantizer: explicitly
    report that quantization is unnecessary or not recommended.
- Preserve explicit catalog-authored `execution_variants` as authoritative.
- Treat a single catalog `quantization` value as a measured baseline rather
  than the only allowed strategy. Expand GGUF Q4 baselines to FP16, Q8, Q6,
  Q5, Q4 and Q3; expand Transformers INT4 baselines to FP16, INT8 and INT4.
- Include the original estimated resource requirement and selected backend
  in variant metadata so explanations can state why a lower precision was
  selected.

## Constraints

- Do not claim that every model repository ships every quantized artifact.
- Generated memory estimates are planning estimates, not benchmark results.
- Never apply GGUF rules to diffusion, audio, or video-generation pipelines.
- Existing API fields remain backward compatible.

## Approaches considered

1. Fill every catalog record manually. Accurate but difficult to maintain
   across 80 models and leaves future additions incomplete.
2. Infer one universal ratio from parameter count. Simple but technically
   wrong across model families.
3. Generate family-specific variants and allow catalog overrides. Chosen
   because it keeps the first version maintainable while preserving explicit
   model data when available.
