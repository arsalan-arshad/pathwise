/**
 * Regression bench for premature pruning (SYNTHETIC, labeled as such).
 *
 * Real HotpotQA run, Nov 10 2026: queries that every plan fails are common, and they tie at 0. Early in
 * calibration those ties were counted as evidence that a cheap plan is "never worse" than a better, pricier
 * one, and the better plan was pruned for good. The planner then served the worse plan to everyone.
 *
 * Workload: 45% hopeless queries (no plan passes), 30% easy (any plan passes), 25% hard (only the pricier
 * model passes). Hopeless queries come first, the adversarial order. Only the pricier model can meet the SLA (0.5): small alone reaches 0.30, large 0.55.
 *
 * Usage: npx tsx examples/prune-bench.ts [seeds=20] [calibrationSamples=100]
 */
import { createPlanner } from "../src/index.js";
import { rng } from "./workload.js";

const SEEDS = Number(process.argv[2] ?? 20);
const CALIBRATION = Number(process.argv[3] ?? 100);
const SLA = 0.5;

interface Q { id: number; kind: "hopeless" | "easy" | "hard" }

async function once(seed: number) {
  const rand = rng(seed);
  const make = (n: number, hopelessFirst: boolean): Q[] => {
    const qs: Q[] = Array.from({ length: n }, (_, id) => {
      const u = rand();
      return { id, kind: u < 0.45 ? "hopeless" : u < 0.75 ? "easy" : "hard" };
    });
    if (hopelessFirst) qs.sort((a, b) => Number(b.kind === "hopeless") - Number(a.kind === "hopeless"));
    return qs;
  };
  const planner = createPlanner<Q, { ok: boolean }>({
    space: { model: ["small", "large"], topK: [2, 5] },
    pipeline: async (q, plan) => {
      const ok = q.kind === "easy" || (q.kind === "hard" && plan.model === "large");
      return { output: { ok }, costUsd: (plan.model === "large" ? 3 : 1) + (plan.topK as number) * 0.1, latencyMs: 100 };
    },
    scorer: (_q, out) => (out.ok ? 1 : 0),
    sla: { minQuality: SLA },
    // Position-derived features: text carries no signal here, so routing cannot help and plan choice is the test.
    featurize: (q) => [Math.sin(q.id * 12.9898), Math.cos(q.id * 78.233)],
    feedbackRate: 0,
    concurrency: 1,
  });
  const calib = await planner.calibrate(make(CALIBRATION, true));
  const prod = make(1000, false);
  let ok = 0;
  for (const q of prod) if ((await planner.run(q)).output.ok) ok++;
  return {
    prunedLarge: calib.prunedPlans.some((k) => k.includes("large")),
    accuracy: ok / prod.length,
    calibRuns: calib.runs,
    prodCost: planner.report().totalCostUsd,
  };
}

const rows: Awaited<ReturnType<typeof once>>[] = [];
for (let s = 1; s <= SEEDS; s++) rows.push(await once(s));
const mean = (f: (r: (typeof rows)[number]) => number) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
console.log(`SYNTHETIC premature-pruning bench, ${SEEDS} seeds, ${CALIBRATION} calibration queries, SLA ${SLA}`);
console.log(`pricier model pruned in ${rows.filter((r) => r.prunedLarge).length}/${SEEDS} runs`);
console.log(`production accuracy ${mean((r) => r.accuracy).toFixed(3)} (SLA ${SLA}; best possible 0.550 = easy + hard share; the small model alone gets 0.300), calibration runs ${mean((r) => r.calibRuns).toFixed(0)}, production cost ${mean((r) => r.prodCost).toFixed(0)}`);

// Regression bench: pruning the pricier model is the bug this guards against, so it must fail CI.
const pruned = rows.filter((r) => r.prunedLarge).length;
if (pruned > 0) {
  console.error(`FAIL: the pricier model was pruned in ${pruned}/${SEEDS} runs`);
  process.exit(1);
}
