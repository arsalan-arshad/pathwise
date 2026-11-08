# Real-RAG benchmark

Tests Pathwise on **real Gemini API calls** and a **real eval set** (HotpotQA dev, distractor setting).
Everything else in this repo is a simulation; this is where the numbers become real.

- **Pipeline:** BM25 over each question's 10 paragraphs (2 contain the answer) -> Gemini answers from the top-k.
- **Knobs (18 plans):** model (`gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-4-argon`) x top-k (2, 5, 10) x thinking (low, high).
- **Quality:** token F1 against the gold answer (official HotpotQA rules). Deterministic, so no judge noise.
- **Cost:** real token usage x official prices (`config.ts`, checked Nov 8, 2026). Thinking tokens billed as output.
- **Safety:** hard budget cap (default $25, `PATHWISE_BENCH_BUDGET`), persistent spend ledger, response cache
  (re-runs are free; every strategy is still charged the cost its calls originally had).

Prices change: `gemini-4-argon` doubles on Dec 1, 2026 and `gemini-3.8-flash` on Jan 1, 2027. `config.ts` applies them by date.

## Run it (about 30-45 minutes, under $25)

```bash
export GEMINI_API_KEY=...            # your key, from Google AI Studio; never committed or logged

npm run bench:rag:smoke              # 3 tiny calls: checks the key, model ids and usage fields
npm run bench:rag:pilot              # 18 plans x 40 questions: per-plan quality and cost, suggests --sla
npm run bench:rag -- --sla 0.XX      # main run: 200 calibration + 500 test questions
```

Options: `--calibration N`, `--test N`, `--confidence 0.9`, `--concurrency 6`.
Offline dry run with a fake model (no key, no cost): prefix any command with `PATHWISE_BENCH_MOCK=1`.

Results land in `results/` as Markdown and JSON. The downloaded data (`data/`, CC BY-SA 4.0) and the
response cache (`cache/`) stay local and are gitignored.

## Known limits

- HotpotQA is public and old: models may have seen it in training, which flatters small models.
- One dataset, one provider, one run: treat the result as a first real data point, not a general claim.
- Gemini 4 / 3.6+ ignore temperature, so answers vary between runs; the cache freezes one sample.
