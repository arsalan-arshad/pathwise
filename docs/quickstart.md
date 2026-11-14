# Quick start

From install to a planner serving traffic. Everything on this page runs with no model, no network and no API key
(`npm run example:quickstart` in a clone). The numbers it prints come from a **simulated** pipeline: they show the mechanism,
not what you will save.

## 1. Install

```bash
npm install @arsalandev/pathwise
```

Zero runtime dependencies. Node 22 or newer.

## 2. Declare, calibrate, serve

A sketch with placeholders (`myRag`, `judge`, `sampleQuestions` are your code; types are illustrative):

```ts
import { createPlanner } from "@arsalandev/pathwise";

const planner = createPlanner<string, MyAnswer>({
  // The knobs: everything that trades cost for quality in your pipeline.
  space: { model: ["small", "large"], topK: [2, 5] },

  // Your existing pipeline, parameterised by a plan. Return the output and what the call cost in USD.
  pipeline: async (question, plan) => {
    const answer = await myRag(question, plan); // your code
    return { output: answer, costUsd: answer.costUsd };
  },

  // Score an output from 0 to 1: an eval-set check or an LLM judge.
  scorer: (question, answer) => judge(question, answer),

  // The service level. `confidence` turns "on average" into a bound you can check, and `fallbackPlan` is what runs if
  // calibration cannot certify it (use the plan you trust today).
  sla: { minQuality: 0.8, confidence: 0.9, fallbackPlan: { model: "large", topK: 5 } },
});

// Calibrate on a sample of REAL questions (more on how many below), then read the report.
const report = await planner.calibrate(sampleQuestions);

// Serve traffic: each request gets the cheapest plan predicted to meet the SLA.
const { output, plan, costUsd } = await planner.run(question);
```

A runnable version, with a simulated pipeline and scorer in place of the placeholders, is [`examples/quickstart.ts`](../examples/quickstart.ts); CI typechecks it. It prints,
for a simulated 300-question workload (SYNTHETIC): calibration used 217 runs for $1.014 and the planner served the remaining 180
questions at $1.030 against $1.980 for always using the large model with `topK` 5, with 100.0% accuracy. Real pipelines will differ.

Already using the Vercel AI SDK, LangChain.js or any OpenAI-compatible server? The pipeline function is a few lines with
[the adapters](../README.md#vercel-ai-sdk): `@arsalandev/pathwise/vercel-ai`, [`/langchain`](../README.md#langchainjs), [`/openai-compatible`](../README.md#openai-compatible-endpoints-no-sdk).

## 3. Read the calibration report

| Field | Meaning |
|---|---|
| `slaCertified` | `true`: a leave-one-out replay of your calibration questions clears `minQuality` at `confidence`. `false`: not enough evidence yet. |
| `certifiedQuality` | The lower confidence bound that was reached. Compare it with `minQuality`. |
| `policy` | `"routing"`: each request gets the cheapest plan predicted to meet the SLA. `"safe-plan"`: not certified, so every request gets one plan. |
| `fixedPlan` | That one plan: your `fallbackPlan` if you set one, otherwise the plan with the best replayed quality. |
| `runs`, `costUsd` | What calibration cost. Adaptive calibration usually needs a fraction of the runs a full sweep would. |

**"Not certified" is not a failure.** It means the data cannot prove the SLA at that confidence, and Pathwise plays safe: it never
silently serves a cheaper plan it cannot back. On our one real-data run (HotpotQA, 100 calibration questions, an SLA of 95% of
always-max's own score) it did not certify and fell back to its safe plan, which turned out to be always-max (no `fallbackPlan` was set), with no savings. See the README, "First
result on real data".

## 4. How many calibration questions?

Certifying needs enough questions that the *lower* confidence bound clears your target. The table gives the number of
calibration questions at which a single plan whose observed quality equals its true quality, a given margin above the SLA, just clears the
lower bound at `confidence: 0.9` (the planner's Wilson formula, written for pass/fail scores; computed for this page, not measured on a
pipeline). Sampling noise means you will clear it at that n only about half the time, so plan on more:

| SLA (`minQuality`) | plan 2 points above | plan 5 points above | plan 10 points above |
|---|---|---|---|
| 0.80 | 657 | 106 | 27 |
| 0.90 | 370 | 60 | 15 |
| 0.95 | 196 | 32 | n/a (needs quality above 1.0) |

(A plan at quality 1.00, in the 0.90 row's last column and the 0.95 row's middle column, never fails a question; real plans do not.)

Read it this way:
- **A tight margin needs hundreds of questions.** A 95%-of-always-max SLA is the hard case: even the most expensive plan sits only a few points
  above the target, so a bound at n=100 falls short. Our real run: always-max scored 0.764 and the target was 0.725 (95% of it); the
  same formula applied to that mean F1 (a graded score, treated like a pass rate) gives 0.706 at n=100, 0.723 at n=200 and 0.731 at n=300.
- **A looser target certifies much sooner.** Aim for a target with real margin below your best plan (for example 90% of always-max's score, not 95%).
- The planner's real check replays its *routing* leave-one-out, not one fixed plan, and judges are noisy and scores often graded, so all of
  that needs more questions than the table. Treat it as "do not expect to certify with fewer than this".

## 5. Production settings that matter

- **Noisy judge:** if the scorer is an LLM, set `scorerErrorRate` to how often it disagrees with human labels on a labelled sample.
  Without it a judge that is wrong 10% of the time makes even perfect plans look 90% correct. Pair it with `confidence`.
- **Keep learning:** `feedbackRate` (default 0.1) is the fraction of live requests that are scored and fed back. Set it to 0 for fully
  deterministic runs.
- **Constraints:** `maxCostUsd` and `maxLatencyMs` in `sla` filter plans when routing. They are **not** enforced while a safe plan or
  `fallbackPlan` is running.
- **Better matching:** the default text featurizer is hashed words. `@arsalandev/pathwise/embeddings` adds real embeddings; its effect on
  real data is not measured yet.
- **Calibrate again** when models, prices or traffic change. The guarantee covers traffic like the calibration sample, not drift.

## 6. What this is not

Synthetic numbers are labelled as such everywhere in these docs. The one real-data result so far is a single HotpotQA run, and it
did not show that routing beats the best single fixed plan. Measure on your own traffic before relying on a saving.
