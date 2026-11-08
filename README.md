<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner/readme-banner-dark.png">
  <img alt="Pathwise: the cheapest path to the answer" src="assets/banner/readme-banner-light.png">
</picture>

[![ci](https://github.com/arsalan-arshad/pathwise/actions/workflows/ci.yml/badge.svg)](https://github.com/arsalan-arshad/pathwise/actions/workflows/ci.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A query planner for RAG and agent pipelines.** Declare the quality you need; Pathwise picks the cheapest way to deliver it, per request.

Databases have had query optimizers for 50 years. AI pipelines don't: teams hard-code one model, one `topK`, one reranker setting for every request, and pay frontier-model prices for "what are your opening hours?".

## How it works

1. **Mark your knobs.** Model, retrieval depth, reranker, agent step budget: anything that trades cost for quality.
2. **Calibrate.** Pathwise runs plans on a sample of your real queries and scores them: cheapest plans first, stopping at the first plan that wins each query, and pruning plans a cheaper one never loses to. About 9% of the cost of running every plan (`calibration: { strategy: "exhaustive" }` restores that).
3. **Plan per request.** For each new query it finds similar past queries, predicts each plan's quality, cost and latency, and runs the cheapest plan that meets your SLA.
4. **Keep learning.** A fraction of production runs are scored and fed back, so it adapts as models and prices change.

```ts
import { createPlanner } from "pathwise";

const planner = createPlanner({
  space: { model: ["small", "medium", "large"], topK: [2, 5, 10], rerank: [false, true] },
  pipeline: async (query, plan) => {
    const answer = await myRag(query, plan); // your existing code, now parameterised
    return { output: answer, costUsd: answer.cost };
  },
  scorer: (query, answer) => judge(query, answer), // 0..1: eval set or LLM judge
  sla: { minQuality: 0.9, maxLatencyMs: 3000 },
});

await planner.calibrate(sampleQueries);
const { output, plan } = await planner.run(userQuery);
```

## SLA guarantee

By default Pathwise picks the cheapest plan whose *average* predicted quality meets `minQuality`.
Set `confidence` to turn that into a guarantee you can check:

```ts
const planner = createPlanner({
  // ...
  sla: { minQuality: 0.95, confidence: 0.9 }, // 90% confidence that delivered quality >= 95%
  scorerErrorRate: 0.1, // your LLM judge disagrees with human labels 10% of the time
});

const report = await planner.calibrate(sampleQueries);
report.slaCertified; // false = not enough evidence yet; Pathwise plays safe (or set onUncertified: "best-effort")
```

How it works: after calibration, Pathwise replays every calibration query as if it had never seen it,
and raises its internal quality bar until the replayed results clear `minQuality` at the requested
confidence (Wilson lower bound). If no bar does, it runs the best plan for every request until you
add calibration samples. With a noisy judge, set `scorerErrorRate`: calibration then majority-votes
each verdict, confirms passes with extra votes, and corrects scores for the judge's error rate.

Simulated benchmark (`npm run bench:sla`, **synthetic**, 20 random workloads per row, 1,000 production queries,
"messy" = question text predicts difficulty poorly, "noisy" = judge wrong 10% of the time):

| Scenario (95% SLA) | Calibration samples | SLA misses: average-based | SLA misses: guaranteed | Cost: average-based | Cost: guaranteed |
|---|---|---|---|---|---|
| messy | 90 | 17/20 | **0/20** | $7.12 | $30.28 |
| messy | 300 | 17/20 | **0/20** | $6.84 | $14.06 |
| messy + noisy judge | 300 | 4/20 | **0/20** | $21.41 | $23.08 |
| clean | 300 | 0/20 | 0/20 | $5.16 | $5.16 |

The guarantee costs money when evidence is thin and gets cheaper with more calibration data.
Across all 16 scenarios we ran (2 workloads x noise on/off x 90/95% SLA x 90/300 samples) the guaranteed mode missed 0 of 320 runs;
the average-based mode missed 45. Real pipelines will differ.

## Demo

```bash
npm install
npm run demo
```

Simulated support-bot workload, 1,000 queries, 90% accuracy SLA:

| Strategy | Accuracy | Cost |
|---|---|---|
| Always max (large, topK 10, rerank) | 100.0% | $33.00 |
| Typical default (large, topK 5) | 100.0% | $26.00 |
| Always cheap (small, topK 2) | 52.8% | $0.34 |
| **Pathwise planner** | **100.0%** | **$4.93** |

The workload is synthetic; real savings depend on how varied your traffic is.

## Roadmap

- [x] Smarter calibration: cheapest-first search with early stop, dominance pruning, monotonicity audits
- [ ] Real embeddings featurizer (OpenAI / Voyage / local) instead of hashed text features
- [ ] Adapters for Vercel AI SDK, LangChain.js, LlamaIndex.TS
- [ ] Persistent store (SQLite / Postgres) for learned data
- [ ] Confidence bounds on predicted quality (guarantee the SLA, not just the average)
- [ ] Hosted dashboard: dollars saved, SLA met, plan mix over time

## Author

Built by **Arsalan Arshad** at [arsalandev.studio](https://arsalandev.studio).
If you want help with a Pathwise integration or a custom AI pipeline, [start a conversation](https://arsalandev.studio).

## License

MIT © [Arsalan Arshad](https://arsalandev.studio)
