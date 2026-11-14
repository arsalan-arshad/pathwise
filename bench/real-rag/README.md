# Real-RAG benchmark

Tests Pathwise on **real Gemini API calls** and a **real eval set** (HotpotQA dev, distractor setting).
Everything else in this repo is a simulation; this is where the numbers become real.

- **Pipeline:** BM25 over each question's 10 paragraphs (2 contain the answer) -> Gemini answers from the top-k.
- **Knobs:** model x top-k (2, 5, 10) x thinking (low, high): 12 plans with the free set's 2 models, 18 with 3.
- **Models, free set (default, costs $0):** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`,
  all on the Gemini API free tier. (`gemini-3.8-flash` was tried first but was overloaded and hit its daily free
  quota within a few calls.) Override any tier with `PATHWISE_BENCH_LARGE_MODEL=...` (also `_SMALL_`, `_MEDIUM_`).
  **The free set runs only small + medium by default**: free-tier Flash models (`large`) allowed only about 20-30
  calls a day in our runs, Flash-Lite models hundreds. Add the large tier with `PATHWISE_BENCH_TIERS=small,medium,large`. **Paid set** (`PATHWISE_BENCH_MODELS=paid`, needs billing):
  `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-4-argon`.
- **Quality:** token F1 against the gold answer (official HotpotQA rules). Deterministic, so no judge noise.
- **Cost:** real token usage x official paid-tier list prices (`config.ts`, checked Nov 8, 2026), thinking billed as
  output. On the free tier you pay nothing; the report shows what the same calls cost at list price, which is
  what the savings comparison needs.
- **Free-tier friendly:** requests-per-minute throttle (default 10, `--rpm`), automatic slow-down on per-minute
  limits, and a clean stop when the daily quota runs out. Run the same command the next day: every finished call
  is cached, so it continues where it stopped.
- **Safety:** response cache, persistent ledger, and a budget cap (default $25, `PATHWISE_BENCH_BUDGET`) that only
  matters on the paid set.

The dataset (7,405 questions) is fetched from the Hugging Face dataset viewer on first run (about 75 requests;
the original host, curtis.ml.cmu.edu, no longer resolves). It stays local (`data/`, CC BY-SA 4.0, gitignored).

## Run it for free

```bash
export GEMINI_API_KEY=...            # free key from Google AI Studio; never committed or logged

npm run bench:rag:smoke              # 3 tiny calls: checks the key, model ids and usage fields
npm run bench:rag:pilot              # 12 plans x 20 questions (240 calls, ~25 min at 10/min): suggests --sla
npm run bench:rag -- --sla-of-max 0.95   # main run: 100 calibration + 200 test questions
```

`--sla-of-max 0.95` sets the SLA to 95% of the "always max" plan's mean F1 on the calibration questions
(never the test set): "at least 95% of my most expensive setting's quality". A fixed `--sla 0.8` also works,
but a bar picked from a small pilot can be unreachable on the full set (that happened on our first run).

Check your free-tier limits at https://aistudio.google.com/rate-limit and pass `--rpm` to match.
Options: `--calibration N`, `--test N`, `--confidence 0.9`, `--concurrency 2`, `--rpm 10`.
Offline dry run with a fake model (no key, no calls): prefix any command with `PATHWISE_BENCH_MOCK=1`.
Results land in `results/` as Markdown and JSON.

Replay a calibration from the cache without any API call (it stops at the first response that was never cached):
`npm run bench:rag:replay -- --sla 0.725`. It prints what calibration learned per plan and why plans were pruned.
`PATHWISE_BENCH_OFFLINE=1` makes any command cache-only.

Free-tier note: Google's terms for unpaid API use may allow it to use prompts to improve its products. This
benchmark only sends public HotpotQA text; do not send private data on a free key.

## Filling the table (free tier)

Only part of the (question, plan) table is measured, so we cannot yet say whether routing beats the best single fixed plan. `bench:rag:fill` measures the missing pairs. It uses only `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite` (it refuses any other model, the paid set, and `PATHWISE_BENCH_OFFLINE=1` with `--yes-live`).

```bash
export GEMINI_API_KEY=...     # your own shell only; the tool never prints or stores it
cd ~/work/AI-architecture-patterns/pathwise && git checkout main && git pull && npm ci

# 1. What is missing (no call is made). Expected before any fill: 501 of 1,200 calibration pairs, small 235 / medium 266.
npm run bench:rag:fill -- --phase calibration

# 2. Fill the calibration table. If it stops with "daily quota used up", rerun the SAME command after the reset
#    (about 13:15 PKT); it resumes from the cache. Repeat until it prints "remaining 0".
caffeinate -i npm run bench:rag:fill -- --phase calibration --yes-live

# 3. Fill ONLY plans you chose on the calibration set (up to 4 per run) on the 200 test questions.
caffeinate -i npm run bench:rag:fill -- --phase test --plans "model=medium|thinking=low|topK=10" --yes-live
```

- **Throttle:** `--rpm` is the total requests per minute across both models (default 10, a conservative free-tier setting; raise it only after checking https://aistudio.google.com/rate-limit). 501 calls take about 51 minutes at 10 per minute when no daily limit interferes.
- **Daily limit:** a model whose daily quota is used up is stopped cleanly, the other model goes on, and the tool prints the remaining calls per model. A model that keeps failing after the retries is treated the same way. Exit code 0.
- **Resumable:** every response is cached as it arrives (`cache/live-free/`). Ctrl-C leaves a valid cache.
- **Ledger:** the persistent list-price ledger and its ceiling (`PATHWISE_BENCH_BUDGET`, default $25) still apply; the free tier bills $0. The dry run prints the list-price cost of the missing pairs as a ceiling for a paid key, not a forecast.
- **Test pairs** are only ever filled for plans you name, so nothing is chosen from test results.
- **Checks:** `npm run bench:rag:fill:check` runs the tool in mock mode against a temp cache (refusals, quota stop, resume). It needs the dataset downloaded.

## Known limits

- HotpotQA is public and old: models may have seen it in training, which flatters small models.
- The free set's largest model is `gemini-3.6-flash`, not a frontier model, so the price gap between plans is
  smaller than in a real stack that tops out at a frontier model. Savings measured here are likely an underestimate.
- One dataset, one provider, one run: treat the result as a first real data point, not a general claim.
- Gemini 4 / 3.6+ ignore temperature, so answers vary between runs; the cache freezes one sample.
