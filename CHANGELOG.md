# Changelog

All notable changes to `@arsalandev/pathwise`. Format follows [Keep a Changelog](https://keepachangelog.com/).

Every number here is labelled **real** (measured on a real pipeline), **synthetic** (a simulated workload), or **not
measured**. Synthetic numbers show that the mechanism works; they are not a forecast of your savings.

## [0.2.0] - 2026-11-17

What is new, in one list. Every number is labelled REAL, SYNTHETIC or not measured. The headline claims are limited by what was measured: see "Known limits" at the end of this entry.

### Added
- **`planner.recommend(samples)`**: races the plans on the SAME questions (successive elimination with Wilson bounds) and returns the cheapest plan whose mean quality meets `sla.minQuality`, or `undefined` (never an invented plan). Racing is published prior art, not a new method. REAL (HotpotQA, 100 questions, 200 shuffles, one seed, in-sample): same plan as the full table in 195/200 orders at SLA 0.725 with about 37% fewer runs (760 of 1,200); it was never certified in that setting. SYNTHETIC checks in `npm run bench:recommend`.
- **`planner.certify(freshSamples)`**: runs the frozen policy on FRESH questions and puts ONE Wilson lower bound on its mean quality (no threshold search, no replay, no learning; `decide()` and `slaCertified` untouched). SYNTHETIC with known truth (0/1 scores, SLA 0.9, confidence 0.9): a plan exactly at the SLA is falsely certified in about 12% of runs at 100 fresh questions (by design about 10%), a plan 2 points below in about 3%, and a plan 5 points above is certified in 77% of runs with 100 fresh questions and 88% with 150. REAL HotpotQA with 30 fresh questions certified 11 of 200 splits at SLA 0.725.
- **`pathwise` command line** (`npx @arsalandev/pathwise recommend|certify --log file.jsonl`): the same statistics over a JSONL log of measured runs; zero dependencies, no key, no network. SYNTHETIC checks only.
- **`@arsalandev/pathwise/postgres`**: store `exportState()` in your own Postgres through a pg-compatible client (no `pg` dependency). Tested against an in-memory fake (SYNTHETIC) and once by the author against a local Postgres (Postgres.app, 3 records); not tested against a hosted database.
- **Bench tools** (`bench/real-rag`, not in the package): `bench:rag:analyze`, `bench:rag:fill`, `bench:rag:embed-compare`, `bench:rag:probe`, `bench:rag:certify`, `bench:rag:certify:sweep`, `bench:rag:single-plan`, `smoke:ollama`, `smoke:postgres`, and the committed result files under `bench/real-rag/results`.

### Known limits (read before relying on a number)
- `slaCertified` (the replay claim from `sla.confidence`) is optimistic with a calibration set of 100 or fewer 0/1 scores (SYNTHETIC, known truth: 17% to 40% false certification against 10% nominal) and, when its internal monotonicity flag is false, counts unmeasured plans as zero, so it gets RARER as the calibration set grows (SYNTHETIC: best plan truly 5 points above the SLA, 0/1 scores: certified in 51% / 32% / 13% / 5% of runs at n = 40 / 100 / 300 / 1000 while a fresh-question bound certified 19% / 59% / 88% / 97%). Treat it as a replay estimate; use `certify()` on fresh questions for the measured check. Nothing in `decide()` or the bound changed in this release.
- `certify()` covers traffic like the fresh sample, not drift; use the fresh questions once per frozen policy; `scorerErrorRate` is not corrected in `recommend()` or `certify()` (a warning says so); under about 60 fresh questions a thin margin cannot certify, and a margin of a couple of points needs several hundred.
- REAL results are one dataset family at a time (HotpotQA dev, MuSiQue dev), two Gemini free-tier models at list prices, one seed. Routing did not beat one well-chosen fixed plan on HotpotQA; the MuSiQue probe (cheapest vs dearest plan, 300 questions) found no clear signal (AUC 0.546 / 0.549 against a 0.65 gate) and that pair cannot show headroom. Nothing here shows routing is useless on other workloads.
- The 0.1.0 entry below is history and is unchanged.

### Measured on real data (Nov 15, 2026)
Same setup as the 0.1.0 entry below (HotpotQA dev, BM25, two Gemini free-tier models at list prices, 12 plans, 100 calibration and 200 test questions, one seed, one sample per pair), now with the full calibration table.
- **A single fixed plan beat Pathwise's routing on this data.** Best fixed plan chosen on calibration only (small, top-k 10, thinking low): test F1 0.756 at $0.00055 per query, meets 0.725. Always-max: F1 0.789 at $0.00164. Pathwise average-based: F1 0.707 at $0.00066 (missed). Guaranteed mode fell back to always-max ($0.00164); the fixed plan was cheaper by $0.00110 per query (95% bootstrap interval 0.00097 to 0.00123). For the guaranteed mode the cost interval excludes 0; the average-based comparison is point estimates only (a recorded run, no interval).
- **Real embeddings did not change the result.** Local `nomic-embed-text` vs hashed text features, calibration-only cross-validation: F1 0.654 vs 0.653 at $0.00051 per query. Test set not measured.
- Not certifiable at 100 questions: the fixed plan's 90% lower bound is 0.678 against a 0.725 target.
- One dataset, one provider, one seed. Other datasets and pipelines are not measured. Supersedes the 0.1.0 lines "Not shown: that routing beats the best single fixed plan" and "Not measured: the real-data effect of real embeddings".

### Also added
- Issue forms and `SECURITY.md` (private vulnerability reporting), `permissions: contents: read` for CI.
- Quick start (`docs/quickstart.md`) with a calibration-size table (computed, not measured).
- Benchmark tools: `bench:rag:fill` (live, free Gemini models only), `bench:rag:analyze` (best fixed plan vs routing, offline), `bench:rag:embed-compare` (local Ollama embeddings vs hashed features), `smoke:ollama` (written; not confirmed run against a real server).

### Security
- `recommend()` and `certify()` reject a negative `costUsd`. `openAIEmbed` and `voyageEmbed` remove the configured API key (6+ characters) from HTTP error text, as `pathwise/openai-compatible` does.

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
- Guaranteed mode **did not certify** at 100 calibration questions and fell back to its safe plan, which here was
  always-max (`medium|high|topK10`): SLA met, **no savings**. A Wilson lower bound calculation (not a field in the results
  files) says that with 100 questions even always-max cannot prove 95% of its own score (0.706 < 0.725).
- **Not shown:** that routing beats the best single fixed plan. Only part of the plan table was measured.
- One dataset, one provider, one run. HotpotQA may be in the models' training data.

### Synthetic only
- Simulated support-bot workload, 1,000 queries, 90% SLA: always-max $33.00 vs Pathwise $4.93 at 100% accuracy (85% cheaper);
  adaptive calibration costs $1.67 (463 of 1,620 runs) instead of $17.74.
- SLA guarantee: 0 misses in 320 simulated runs (16 scenarios x 20 workloads), against 45 for the average-based mode; the
  guarantee costs more when calibration evidence is thin.
- `decide()` takes under 1 ms per request (about 0.07 ms, measured on the fake-model adapter benches).
- Embeddings: a fake embedder that encodes difficulty beat hashed text features, and a pure-noise embedder was worse than
  hashed features. This says nothing about real embedding models.

### Not measured or not tested
- The real-data effect of real embeddings.
- The adapters against live providers. `vercel-ai` and `langchain` were smoke-tested on Nov 13, 2026 against `ai` 7.0.171 and
  `@langchain/core` 1.4.0 using their own mock/fake models. `embeddings` and `openai-compatible` were tested only against
  fake servers on localhost.
- Latency on real data: free-tier latency was erratic (queueing, 503s), so the latency figures in the results files are
  not meaningful and no latency result is claimed.

### Fixed during development
- Calibration pruned the best plans on queries that both compared plans fail (ties at 0 counted as evidence that the cheaper
  plan is never worse), which made guaranteed mode serve a worse, dearer plan in an early real run (F1 0.669 at $0.00144 per
  query). Found by that run, fixed before release.
- A regex in the OpenAI-compatible adapter was flagged by CodeQL (polynomial ReDoS); replaced with a loop.
