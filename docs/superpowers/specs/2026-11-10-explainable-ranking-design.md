# Explainable Preference Ranking Design

## Intent

Make the speed, balanced, and quality controls produce visibly different,
defensible rankings while keeping hardware compatibility authoritative.

## Approved Approach

Use a two-layer ranking system:

1. A hard compatibility gate keeps any ideal, runnable, or constrained
   candidate ahead of unsupported candidates.
2. A weighted 100-point score ranks candidates inside those groups.

The score has five components:

- hardware fit;
- inference speed and resource efficiency;
- expected task quality;
- runtime/deployment readiness;
- estimate confidence.

## Preference Weights

| Preference | Hardware | Speed | Quality | Readiness | Confidence |
|---|---:|---:|---:|---:|---:|
| speed | 35 | 35 | 15 | 10 | 5 |
| balanced | 35 | 20 | 25 | 10 | 10 |
| quality | 30 | 10 | 45 | 10 | 5 |

Weights sum to 100. Unsupported models can still be shown for server planning,
but cannot outrank a locally viable model.

## Component Semantics

- Hardware fit uses fit tier plus continuous peak RAM/VRAM utilization.
- Speed uses relative resource footprint, parameter size, and execution-mode
  penalties for CPU-only and CPU offload.
- Quality uses a relative capacity signal derived from parameter size,
  selected-variant quality rank, catalog priority, model resource class, and
  an external quality prior when llmfit supplies one.
- Readiness rewards an installed runtime and penalizes access-token
  requirements.
- Confidence maps benchmarked/calibrated/catalog/heuristic/unknown estimates to
  decreasing scores.

All inferred quality and speed values are explicitly heuristic. They are
ranking aids, not benchmark claims.

## API

`CandidateRecommendation` gains additive fields:

- `preference`;
- `score_breakdown`, containing component values, weights, and final score;
- `ranking_reasons`, containing concise explanations for the ranking.

Existing `score` remains the final weighted score for compatibility.

## Frontend

Candidate cards show:

- rank and total score prominently;
- active preference;
- five compact component bars;
- a short “why this rank” explanation.

Changing the preference and submitting again must change either ordering or
the visible score profile. The page keeps the existing dark map visual system.

## Non-goals

- Claiming benchmark-derived quality without benchmark data.
- Allowing quality preference to recommend an impossible local deployment over
  a viable model.
- Adding another frontend framework or build step.
