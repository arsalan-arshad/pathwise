# Real-RAG benchmark

Tests Pathwise on **real Gemini API calls** and a **real eval set** (HotpotQA dev, distractor setting).
Everything else in this repo is a simulation; this is where the numbers become real.

- **Pipeline:** BM25 over each question's 10 paragraphs (2 contain the answer) -> Gemini answers from the top-k.
- **Knobs (18 plans):** model x top-k (2, 5, 10) x thinking (low, high).
- **Models, free set (default, costs $0):** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`,
  all on the Gemini API free tier. (`gemini-3.8-flash` was tried first but was overloaded and hit its daily free
  quota within a few calls.) Override any tier with `PATHWISE_BENCH_LARGE_MODEL=...` (also `_SMALL_`, `_MEDIUM_`). **Paid set** (`PATHWISE_BENCH_MODELS=paid`, needs billing):
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
npm run bench:rag:pilot              # 18 plans x 20 questions (360 calls, ~40 min at 10/min): suggests --sla
npm run bench:rag -- --sla 0.XX      # main run: 100 calibration + 200 test questions
```

Check your free-tier limits at https://aistudio.google.com/rate-limit and pass `--rpm` to match.
Options: `--calibration N`, `--test N`, `--confidence 0.9`, `--concurrency 2`, `--rpm 10`.
Offline dry run with a fake model (no key, no calls): prefix any command with `PATHWISE_BENCH_MOCK=1`.
Results land in `results/` as Markdown and JSON.

Free-tier note: Google's terms for unpaid API use may allow it to use prompts to improve its products. This
benchmark only sends public HotpotQA text; do not send private data on a free key.

## Known limits

- HotpotQA is public and old: models may have seen it in training, which flatters small models.
- The free set's largest model is `gemini-3.6-flash`, not a frontier model, so the price gap between plans is
  smaller than in a real stack that tops out at a frontier model. Savings measured here are likely an underestimate.
- One dataset, one provider, one run: treat the result as a first real data point, not a general claim.
- Gemini 4 / 3.6+ ignore temperature, so answers vary between runs; the cache freezes one sample.
