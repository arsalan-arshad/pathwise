/**
 * Demo: a simulated customer-support RAG pipeline (see workload.ts).
 * The planner has to learn which plan each kind of question needs from calibration data alone.
 */
import { createPlanner, hashedTextFeaturizer, planKey, type Plan } from "../src/index.js";
import { createWorkload, ragPipeline, type Query } from "./workload.js";

const textFeatures = hashedTextFeaturizer();
const { makeQueries } = createWorkload(42);

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
