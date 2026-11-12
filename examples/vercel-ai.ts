/**
 * Vercel AI SDK adapter example: runs with a FAKE generateText (no `ai` package, no key, no network).
 * With the real SDK, replace `fakeGenerateText` by `import { generateText } from "ai"` and put real models in `models`.
 * Simulated: a model answers correctly when its capability (+ retrieval bonus) >= the question's hidden difficulty.
 */
import { performance } from "node:perf_hooks";
import { createPlanner, hashedTextFeaturizer } from "../src/index.js";
import { vercelAIPipeline, usageCostUsd, type GenerateTextFn } from "../src/adapters/vercel-ai.js";
import { createWorkload, type Query } from "./workload.js";

const CAP: Record<string, number> = { small: 0.35, medium: 0.6, large: 0.85 };
const TOPK_BONUS: Record<number, number> = { 2: 0, 5: 0.05, 10: 0.08 };

// Fake `generateText`: the "model" is just its name; the prompt carries topK and the hidden difficulty.
const fakeGenerateText: GenerateTextFn = async ({ model, prompt }) => {
  const [topK, difficulty] = (prompt ?? "").split("|").map(Number);
  const ok = CAP[model as string] + TOPK_BONUS[topK] >= difficulty;
  return { text: ok ? "right" : "wrong", usage: { inputTokens: 300 + topK * 200, outputTokens: 40, outputTokenDetails: { reasoningTokens: 0 } } };
};

const models = {
  small: { model: "small", price: { inputPerMTok: 0.1, outputPerMTok: 0.4 } },
  medium: { model: "medium", price: { inputPerMTok: 0.75, outputPerMTok: 3.75 } },
  large: { model: "large", price: { inputPerMTok: 2, outputPerMTok: 10 } },
};

function checkUsageCost() {
  const price = { inputPerMTok: 1, outputPerMTok: 10 };
  const near = (a: number, b: number) => Math.abs(a - b) < 1e-12;
  // v7 shape: outputTokens already includes reasoning, so reasoning must not be added again.
  if (!near(usageCostUsd({ inputTokens: 1000, outputTokens: 500, outputTokenDetails: { reasoningTokens: 300 } }, price), 0.006)) throw new Error("v7 usage cost");
  // Flat legacy shape: reasoning reported separately.
  if (!near(usageCostUsd({ promptTokens: 1000, completionTokens: 500, reasoningTokens: 300 }, price), 0.009)) throw new Error("flat usage cost");
  if (usageCostUsd(undefined, price) !== 0) throw new Error("missing usage");
}

async function main() {
  checkUsageCost();
  const { makeQueries } = createWorkload(42);
  const feats = hashedTextFeaturizer();
  const pipeline = vercelAIPipeline<Query>({
    generateText: fakeGenerateText,
    models,
    prepare: (q, plan) => ({ prompt: `${plan.topK}|${q.difficulty}` }),
  });
  const planner = createPlanner<Query, string>({
    space: { model: ["small", "medium", "large"], topK: [2, 5, 10] },
    pipeline,
    scorer: (_q, out) => (out === "right" ? 1 : 0),
    sla: { minQuality: 0.9 },
    featurize: (q) => feats(q.text),
    feedbackRate: 0,
  });
  const calib = await planner.calibrate(makeQueries(90));
  console.log(`calibrated: ${calib.runs} runs, $${calib.costUsd.toFixed(4)} (synthetic)`);

  const prod = makeQueries(500);
  let right = 0;
  let cost = 0;
  for (const q of prod) {
    const r = await planner.run(q);
    if (r.output === "right") right++;
    cost += r.costUsd;
  }
  const maxPlan = { model: "large", topK: 10 };
  const maxCost = usageCostUsd({ inputTokens: 300 + 10 * 200, outputTokens: 40 }, models.large.price) * prod.length;
  console.log(`production (synthetic): accuracy ${(right / prod.length * 100).toFixed(1)}%, $${cost.toFixed(4)} vs always-${maxPlan.model}/topK${maxPlan.topK} $${maxCost.toFixed(4)}`);

  // Overhead: adapter pipeline call vs the raw fake call, and decide().
  const N = 20000;
  const q = prod[0];
  const plan = { model: "small", topK: 5 };
  let t = performance.now();
  for (let i = 0; i < N; i++) await fakeGenerateText({ model: "small", prompt: `5|${q.difficulty}` });
  const raw = (performance.now() - t) / N;
  t = performance.now();
  for (let i = 0; i < N; i++) await pipeline(q, plan);
  const wrapped = (performance.now() - t) / N;
  t = performance.now();
  for (let i = 0; i < N; i++) planner.decide(q);
  const decide = (performance.now() - t) / N;
  console.log(`overhead per call (ms): adapter +${(wrapped - raw).toFixed(4)}, decide() ${decide.toFixed(4)}`);
}
main();
