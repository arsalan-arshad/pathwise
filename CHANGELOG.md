# Changelog

All notable changes to `@arsalandev/pathwise`. Format follows [Keep a Changelog](https://keepachangelog.com/).

Every number here is labelled **real** (measured on a real pipeline), **synthetic** (a simulated workload), or **not
measured**. Synthetic numbers show that the mechanism works; they are not a forecast of your savings.

## [0.1.0] - 2026-11-13

First public release.

### Added
- **Planner core** (`createPlanner`): declare knobs (model, retrieval depth, reranker, thinking level, ...) and an SLA;
  Pathwise picks the cheapest plan predicted to meet it for each request and keeps learning from scored production runs.
  Zero runtime dependencies.
- **Adaptive calibration**: cheapest-first search with early stop, dominance pruning, near-duplicate shortcut and
  monotonicity audits (`calibration: { strategy: "exhaustive" }` restores a full sweep).
- **SLA guarantee** (`sla.confidence`): a leave-one-out replay raises the quality bar until a Wilson lower bound clears
  `minQuality`. If it cannot certify, the default `onUncertified: "safe"` runs one plan for every request; `"best-effort"`
  keeps the closest threshold. `scorerErrorRate` corrects for a noisy LLM judge.
- **`sla.fallbackPlan`**: when the SLA cannot be certified, run exactly the plan you trust today. `CalibrationReport` gains
  `policy` (`"routing"` or `"safe-plan"`) and `fixedPlan`. While it runs, `maxCostUsd` and `maxLatencyMs` are not enforced.
- **Adapters** (separate entry points, no SDK imports, structural types):
  - `@arsalandev/pathwise/vercel-ai`: wraps `generateText`.
  - `@arsalandev/pathwise/langchain`: wraps any chat model with `invoke`.
  - `@arsalandev/pathwise/embeddings`: real-embedding featurizer with a batched `warm()` cache (OpenAI-compatible and Voyage
    embedders on plain `fetch`).
  - `@arsalandev/pathwise/openai-compatible`: Chat Completions over plain `fetch` (OpenAI, Gemini's compatible endpoint,
    OpenRouter, Ollama, LM Studio, vLLM) with retries, timeout, abort and cost from the response's real `usage`.
- **Tooling**: `npm run bench:rag:table` (offline plan-table coverage), `npm run smoke:real-packages` (adapters against the
  real `ai` and `@langchain/core` packages), and the offline benches now run in CI.

### Measured on real data (one run)
HotpotQA dev, BM25 retrieval, two Gemini free-tier models (costs at paid list prices), 12 plans, 100 calibration and 200 test
questions, SLA = 95% of always-max F1 on the calibration questions (0.725). Details and limits in the README and
[`bench/real-rag`](bench/real-rag/README.md).
- Average-based Pathwise: F1 0.707 at $0.00066 per query, **60% cheaper than always-max** (F1 0.789, $0.00164), but it
  **missed the SLA** by 1.8 points.
- Guaranteed mode **did not certify** at 100 calibration questions and fell back to always-max: SLA met, **no savings**.
  With 100 questions even always-max cannot prove 95% of its own score.
- **Not shown:** that routing beats the best single fixed plan. Only part of the plan table was measured.
- One dataset, one provider, one run. HotpotQA may be in the models' training data.

### Synthetic only
- Simulated support-bot workload, 1,000 queries, 90% SLA: always-max $33.00 vs Pathwise $4.93 at 100% accuracy (85% cheaper);
  adaptive calibration costs $1.67 (463 of 1,620 runs) instead of $17.74.
- SLA guarantee: 0 misses in 320 simulated runs (16 scenarios x 20 workloads), against 45 for the average-based mode; the
  guarantee costs more when calibration evidence is thin.
- `decide()` takes under 1 ms per request (about 0.07 ms in these runs).
- Embeddings: a fake embedder that encodes difficulty beat hashed text features, and a pure-noise embedder was worse than
  hashed features. This says nothing about real embedding models.

### Not measured or not tested
- The real-data effect of real embeddings.
- The adapters against live providers. `vercel-ai` and `langchain` were smoke-tested on Nov 13, 2026 against `ai` 7.0.171 and
  `@langchain/core` 1.4.0 using their own mock/fake models. `embeddings` and `openai-compatible` were tested only against
  fake servers on localhost.
- Free-tier latency was erratic, so no latency result is reported from real data.

### Fixed during development
- Calibration pruned the best plans on queries that every plan fails, which made guaranteed mode serve a worse, dearer plan on
  the first real run (F1 0.669 at $0.00144 per query). Found by that run, fixed before release.
- A regex in the OpenAI-compatible adapter was flagged by CodeQL (polynomial ReDoS); replaced with a loop.
