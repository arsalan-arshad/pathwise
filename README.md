# Pathwise

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

## License

MIT
