/**
 * Offline replay of a calibration from the response cache. Costs nothing: it never calls the API.
 *
 *   PATHWISE_BENCH_OFFLINE=1 npx tsx bench/real-rag/replay.ts --sla 0.725 [--calibration 100] [--confidence 0.9]
 *
 * Re-runs Pathwise's calibration exactly as run.ts does, then shows what it learned per plan. If the run
 * needs a response that was never cached it stops and says how many calls a real run would make.
 */
import { createPlanner, hashedTextFeaturizer, planKey } from "../../src/index.js";
import { SPACE } from "./config.js";
import { loadHotpot, sample, type Question } from "./dataset.js";
import { OfflineMiss, offlineMisses } from "./gemini.js";
import { ragPipeline, scorer, type Answer } from "./pipeline.js";

process.env.PATHWISE_BENCH_OFFLINE = "1";
const args = process.argv.slice(2);
const arg = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const SLA = arg("sla", 0.725);
const CALIBRATION = arg("calibration", 100);
const CONFIDENCE = arg("confidence", 0.9);

const { calibration } = sample(await loadHotpot(), CALIBRATION, 0, arg("seed", 7));
const featurize = hashedTextFeaturizer();
const planner = createPlanner<Question, Answer>({
  space: SPACE,
  pipeline: ragPipeline,
  scorer,
  sla: CONFIDENCE > 0 ? { minQuality: SLA, confidence: CONFIDENCE } : { minQuality: SLA },
  featurize: (q) => featurize(q.question),
  feedbackRate: 0,
  concurrency: 2,
});

try {
  const report = await planner.calibrate(calibration);
  console.log(report);
} catch (err) {
  if (!(err instanceof OfflineMiss)) throw err;
  console.log(`Stopped: ${err.message}\n(${offlineMisses.size} distinct uncached call(s) seen before stopping; a real run would make at least that many.)`);
  process.exit(0);
}

// What calibration measured, per plan.
const records = planner.exportState();
const keys = [...new Set(records.flatMap((r) => r.observations.map((o) => o.planKey)))];
console.log("\nplan                                   runs  mean-score  pass-rate  mean-cost");
for (const k of keys.sort()) {
  const obs = records.flatMap((r) => r.observations.filter((o) => o.planKey === k));
  const mean = (f: (o: (typeof obs)[number]) => number) => obs.reduce((s, o) => s + f(o), 0) / obs.length;
  console.log(`${k.padEnd(38)} ${String(obs.length).padStart(4)}  ${mean((o) => o.quality).toFixed(3).padStart(9)}  ${mean((o) => (o.quality >= SLA ? 1 : 0)).toFixed(3).padStart(8)}  $${mean((o) => o.costUsd).toFixed(5)}`);
}
console.log("\nplans never run:", [...planKeys()].filter((k) => !keys.includes(k)).join(", ") || "none");
function planKeys() {
  const out = new Set<string>();
  for (const m of ["small", "medium"]) for (const t of ["low", "high"]) for (const k of [2, 5, 10]) out.add(planKey({ model: m, thinking: t, topK: k }));
  return out;
}

// Why was each pruned plan pruned? Its cheaper "dominator" and the shared queries it was judged on.
{
  const prof = new Map<string, number>();
  for (const k of keys) {
    const obs = records.flatMap((r) => r.observations.filter((o) => o.planKey === k));
    prof.set(k, obs.reduce((s, o) => s + o.costUsd, 0) / obs.length);
  }
  const q = (r: (typeof records)[number], k: string) => {
    const o = r.observations.filter((x) => x.planKey === k);
    return o.length ? o.reduce((s, x) => s + x.quality, 0) / o.length : undefined;
  };
  const shared = records.filter((r) => r.observations.length > 1);
  console.log("\nplan pruned -> cheaper plan that never scored lower on shared queries");
  for (const a of (planner as unknown as { dominated: Set<string> }).dominated ?? []) {
    for (const b of keys) {
      if (b === a || !(prof.get(b)! < prof.get(a)!)) continue;
      const pairs = shared.map((r) => [q(r, a), q(r, b)] as const).filter(([x, y]) => x !== undefined && y !== undefined) as [number, number][];
      if (pairs.length >= 8 && pairs.every(([x, y]) => y >= x)) {
        const informative = pairs.filter(([x, y]) => x >= SLA || y >= SLA).length;
        console.log(`  ${a} <- ${b}: ${pairs.length} shared queries, only ${informative} where either plan passed`);
        break;
      }
    }
  }
}
