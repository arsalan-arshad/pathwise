/**
 * SLA benchmark: how often does the planner actually miss its quality SLA?
 *
 * Stresses the planner two ways (both SYNTHETIC):
 *  - "messy" workload: difficulty jitters so question text predicts it less well;
 *  - noisy scorer: calibration verdicts are flipped with some probability, like an LLM judge.
 * Production accuracy is always measured against ground truth, never the noisy scorer.
 *
 * Usage: npx tsx examples/sla-bench.ts [seeds=20] [confidence=0.9] [safe|best-effort] [calibrationSamples=90]
 */
import { createPlanner, hashedTextFeaturizer } from "../src/index.js";
import { createWorkload, ragPipeline, rng, SPACE, type Query } from "./workload.js";

const SEEDS = Number(process.argv[2] ?? 20);
const CONFIDENCE = Number(process.argv[3] ?? 0.9);
const ON_UNCERTIFIED = (process.argv[4] ?? "safe") as "safe" | "best-effort";
const CALIBRATION = Number(process.argv[5] ?? 90);
const textFeatures = hashedTextFeaturizer();

interface Config {
  workload: "clean" | "messy";
  noise: number;
  minQuality: number;
}

type Mode = "mean" | "corrected" | "guaranteed";

async function runOnce(seed: number, cfg: Config, mode: Mode) {
  const confidence = mode === "guaranteed" ? CONFIDENCE : undefined;
  const { makeQueries } = createWorkload(seed, { jitter: cfg.workload === "messy" ? 0.15 : 0 });
  const calibration = makeQueries(CALIBRATION);
  const production = makeQueries(1000);
  const flip = rng(seed * 7919 + 1);
  const planner = createPlanner<Query, { correct: boolean }>({
    space: { ...SPACE, model: [...SPACE.model], topK: [...SPACE.topK], rerank: [...SPACE.rerank] },
    pipeline: ragPipeline,
    scorer: (_q, out) => {
      const truth = out.correct ? 1 : 0;
      return flip() < cfg.noise ? 1 - truth : truth;
    },
    sla: confidence === undefined ? { minQuality: cfg.minQuality } : { minQuality: cfg.minQuality, confidence, onUncertified: ON_UNCERTIFIED },
    featurize: (q) => textFeatures(q.text),
    feedbackRate: 0,
    // Guaranteed mode also tells the planner the judge's error rate (measured on labelled data in real life).
    scorerErrorRate: mode === "mean" ? 0 : cfg.noise,
  });
  const calib = await planner.calibrate(calibration);
  let correct = 0;
  for (const q of production) if ((await planner.run(q)).output.correct) correct++;
  const stats = planner.report();
  return {
    accuracy: correct / production.length,
    prodCost: stats.totalCostUsd,
    calibCost: stats.calibrationCostUsd,
    threshold: calib.qualityThreshold,
    certified: calib.slaCertified,
    judgeCalls: stats.scorerCalls,
  };
}

const configs: Config[] = [];
for (const workload of ["clean", "messy"] as const)
  for (const noise of [0, 0.1])
    for (const minQuality of [0.9, 0.95]) configs.push({ workload, noise, minQuality });

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const usd = (x: number) => `$${x.toFixed(2)}`;
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

console.log(`SYNTHETIC SLA benchmark: ${SEEDS} seeds per row, ${CALIBRATION} calibration + 1000 production queries`);
console.log(`mean = v0.1 decisions on mean predicted quality`);
console.log(`corrected = mean, plus scorerErrorRate = the judge's true error rate (3-vote calibration scoring)`);
console.log(`guaranteed = corrected, plus sla.confidence ${CONFIDENCE}, onUncertified "${ON_UNCERTIFIED}"\n`);
console.log(
  "Workload".padEnd(9), "Noise".padEnd(6), "SLA".padEnd(5), "Mode".padEnd(11),
  "Misses".padEnd(8), "Certified".padEnd(10), "Worst acc".padEnd(10), "Mean acc".padEnd(9), "Prod cost".padEnd(10), "Calib cost".padEnd(11), "Judge calls",
);
for (const cfg of configs) {
  const modes: Mode[] = cfg.noise > 0 ? ["mean", "corrected", "guaranteed"] : ["mean", "guaranteed"];
  for (const mode of modes) {
    const runs = [];
    for (let s = 1; s <= SEEDS; s++) runs.push(await runOnce(s, cfg, mode));
    const misses = runs.filter((r) => r.accuracy < cfg.minQuality).length;
    console.log(
      cfg.workload.padEnd(9), pct(cfg.noise).padEnd(6), pct(cfg.minQuality).padEnd(5), mode.padEnd(11),
      `${misses}/${SEEDS}`.padEnd(8),
      (mode === "guaranteed" ? `${runs.filter((r) => r.certified).length}/${SEEDS}` : "-").padEnd(10),
      pct(Math.min(...runs.map((r) => r.accuracy))).padEnd(10),
      pct(mean(runs.map((r) => r.accuracy))).padEnd(9),
      usd(mean(runs.map((r) => r.prodCost))).padEnd(10),
      usd(mean(runs.map((r) => r.calibCost))).padEnd(11),
      String(Math.round(mean(runs.map((r) => r.judgeCalls)))),
    );
  }
}
