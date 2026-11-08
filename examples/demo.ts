/**
 * Demo: a simulated customer-support RAG pipeline.
 *
 * Knobs: which model, how many chunks to retrieve, reranker on/off.
 * Easy questions are answered correctly by a small model with little context;
 * hard multi-hop questions need the big model, more context and reranking.
 * The planner has to learn this from calibration data alone.
 */
import { createPlanner, hashedTextFeaturizer, planKey, type Plan } from "../src/index.js";

const textFeatures = hashedTextFeaturizer();

// ---------- deterministic randomness so results are reproducible ----------
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = rng(42);
const pick = <T>(xs: T[]) => xs[Math.floor(rand() * xs.length)];
const between = (a: number, b: number) => a + rand() * (b - a);

// ---------- synthetic query workload ----------
const products = ["starter", "team", "business", "enterprise"];
const countries = ["germany", "brazil", "japan", "canada", "pakistan"];

const tiers = {
  easy: {
    difficulty: [0.15, 0.3],
    templates: [
      () => "what are your opening hours",
      () => "how do i reset my password",
      () => `where is my order ${Math.floor(rand() * 90000 + 10000)}`,
      () => `what is the price of the ${pick(products)} plan`,
      () => `do you ship to ${pick(countries)}`,
    ],
  },
  medium: {
    difficulty: [0.42, 0.6],
    templates: [
      () => `how do i migrate my ${pick(products)} workspace to a new region without downtime`,
      () => `why is my invoice higher this month after upgrading to ${pick(products)}`,
      () => `can i combine the ${pick(products)} plan with an annual discount for a team of ${Math.floor(between(5, 200))}`,
    ],
  },
  hard: {
    difficulty: [0.7, 0.9],
    templates: [
      () =>
        `compare the refund policy for enterprise contracts signed before and after the 2025 terms update and explain which applies to a ${pick(countries)} customer who upgraded mid cycle`,
      () =>
        `given our ${pick(products)} usage logs explain why latency spikes when sso and audit logging are both enabled across multiple regions and what the contract says about credits`,
    ],
  },
} as const;

interface Query {
  text: string;
  difficulty: number; // hidden ground truth, the planner never sees it
}

function makeQueries(n: number): Query[] {
  const mix: Array<keyof typeof tiers> = ["easy", "easy", "easy", "medium", "medium", "hard"];
  return Array.from({ length: n }, () => {
    const tier = tiers[pick(mix)];
    return { text: pick([...tier.templates])(), difficulty: between(tier.difficulty[0], tier.difficulty[1]) };
  });
}

// ---------- the simulated pipeline ----------
const MODEL = {
  small: { capability: 0.35, callUsd: 0.0003, perChunkUsd: 0.00002, latencyMs: 400 },
  medium: { capability: 0.6, callUsd: 0.003, perChunkUsd: 0.0002, latencyMs: 900 },
  large: { capability: 0.85, callUsd: 0.02, perChunkUsd: 0.0012, latencyMs: 2200 },
} as const;
const TOPK_BONUS: Record<number, number> = { 2: 0, 5: 0.05, 10: 0.08 };

async function ragPipeline(q: Query, plan: Plan) {
  const m = MODEL[plan.model as keyof typeof MODEL];
  const topK = plan.topK as number;
  const rerank = plan.rerank as boolean;
  const skill = m.capability + TOPK_BONUS[topK] + (rerank ? 0.05 : 0);
  return {
    output: { correct: skill >= q.difficulty },
    costUsd: m.callUsd + topK * m.perChunkUsd + (rerank ? 0.001 : 0),
    latencyMs: m.latencyMs + topK * 30 + (rerank ? 250 : 0),
  };
}

// ---------- run the experiment ----------
type Strategy = "exhaustive" | "adaptive";

async function evaluate(strategy: Strategy, calibration: Query[], production: Query[]) {
  const planner = createPlanner<Query, { correct: boolean }>({
    space: { model: ["small", "medium", "large"], topK: [2, 5, 10], rerank: [false, true] },
    pipeline: ragPipeline,
    scorer: (_q, out) => (out.correct ? 1 : 0),
    sla: { minQuality: 0.9 },
    // The planner only ever sees the question text, never the hidden difficulty.
    featurize: (q) => textFeatures(q.text),
    feedbackRate: 0, // keep the demo deterministic
    calibration: { strategy },
  });
  const calib = await planner.calibrate(calibration);
  let correct = 0;
  for (const q of production) {
    const r = await planner.run(q);
    if (r.output.correct) correct++;
  }
  return { planner, calib, stats: planner.report(), accuracy: correct / production.length };
}

async function main() {
  const calibration = makeQueries(90);
  const production = makeQueries(1000);

  const before = await evaluate("exhaustive", calibration, production);
  const after = await evaluate("adaptive", calibration, production);
  const { planner, calib, stats } = after;

  // Baselines: fixed plans most teams ship with
  const baselines: Array<[string, Plan]> = [
    ["Always max (large, topK 10, rerank)", { model: "large", topK: 10, rerank: true }],
    ["Typical default (large, topK 5)", { model: "large", topK: 5, rerank: false }],
    ["Always cheap (small, topK 2)", { model: "small", topK: 2, rerank: false }],
  ];

  const usd = (x: number) => `$${x.toFixed(2)}`;
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  console.log(`\nSYNTHETIC workload: ${production.length} production queries, SLA = 90% accuracy\n`);
  console.log("Strategy".padEnd(40), "Accuracy".padEnd(10), "Cost");

  let maxCost = 0;
  for (const [name, plan] of baselines) {
    let correct = 0, cost = 0;
    for (const q of production) {
      const r = await ragPipeline(q, plan);
      if (r.output.correct) correct++;
      cost += r.costUsd;
    }
    if (maxCost === 0) maxCost = cost;
    console.log(name.padEnd(40), pct(correct / production.length).padEnd(10), usd(cost));
  }
  console.log("Pathwise planner".padEnd(40), pct(after.accuracy).padEnd(10), usd(stats.totalCostUsd));

  console.log(`\nSavings vs always-max: ${pct(1 - stats.totalCostUsd / maxCost)}`);
  console.log(`One-off calibration cost: ${usd(stats.calibrationCostUsd)}`);
  console.log(`Fallbacks to max quality: ${stats.fallbacks}`);
  console.log("\nPlans the planner chose:");
  Object.entries(stats.planUsage)
    .sort((a, b) => b[1] - a[1])
    .forEach(([k, n]) => console.log(`  ${String(n).padStart(4)}  ${k}`));

  // Sanity: show one decision
  const example = production.find((q) => q.text.startsWith("compare"))!;
  const d = planner.decide(example);
  console.log(`\nExample hard query -> ${planKey(d.plan)} (predicted quality ${pct(d.estimate.quality)})`);

  // Calibration: exhaustive (v0.1) vs adaptive, same samples, same production traffic
  console.log(`\nCalibration strategies (${calibration.length} samples, ${production.length} queries, SYNTHETIC)\n`);
  console.log(
    "Strategy".padEnd(12), "Runs".padEnd(11), "Calib cost".padEnd(12),
    "Prod cost".padEnd(11), "Accuracy".padEnd(10), "Payback",
  );
  for (const [name, r] of [["exhaustive", before], ["adaptive", after]] as const) {
    const savedPerQuery = (maxCost - r.stats.totalCostUsd) / production.length;
    const payback = savedPerQuery > 0 ? `${Math.ceil(r.stats.calibrationCostUsd / savedPerQuery)} queries` : "never";
    console.log(
      name.padEnd(12),
      `${r.calib.runs}/${r.calib.exhaustiveRuns}`.padEnd(11),
      usd(r.stats.calibrationCostUsd).padEnd(12),
      usd(r.stats.totalCostUsd).padEnd(11),
      pct(r.accuracy).padEnd(10),
      payback,
    );
  }
  console.log(`\nAdaptive: ${calib.seedQueries} seed sweeps, ${calib.shortcutQueries} shortcut queries, ` +
    `${calib.audits} audits, monotonicity violations ${pct(calib.monotonicityViolationRate)} ` +
    `(${calib.assumeMonotone ? "skipped pricier plans credited" : "no credit: conservative"})`);
  console.log(`Pruned as dominated: ${calib.prunedPlans.length ? calib.prunedPlans.join(", ") : "none"}`);
}

main();
