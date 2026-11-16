/**
 * Throwaway DIAGNOSTIC (SYNTHETIC, known truth; no cache, no network, no key; instrument only, nothing in src/ is changed):
 * why does the planner's in-sample certification get RARER as the calibration set grows?
 *
 *   npx tsx bench/real-rag/certify-diagnose.ts [--reps 100] [--sizes 100,1000]
 *
 * World: the T-027 "power" world (0/1 scores, random features, 3 plans cost 1/2/5, best plan truly SLA + 0.05, SLA 0.9, confidence 0.9).
 * Per repetition it reads the planner's private state through a cast (read-only) and replays the calibration records at the
 * threshold the planner picked, exactly as tuneThreshold() does: which plan the replay chooses, and whether that plan was MEASURED on
 * that question or credited (winner credit) or counted as zero ("unmeasured, assume it fails", see realized()).
 * Counterfactuals through the same harness: (a) exhaustive calibration (every plan measured on every question); (b) the same adaptive
 * records but the true best plan forced as the replay choice for every question (with the planner's own realized()), and (b') the same
 * with winner credit forced on. Reports which restores certification at n = 1000.
 */
import { createPlanner, enumeratePlans, planKey } from "../../src/index.js";
import { normalQuantile } from "../../src/stats.js";
import { mulberry32 } from "./analyze-core.js";
import { wilson } from "./single-plan-core.js";

const SPACE = { model: ["cheap", "mid", "max"] } as const;
const IDS = enumeratePlans(SPACE).map((p) => planKey(p));
const COSTS = [1, 2, 5];
const SLOPES = [0.4, 0.3, 0.1];
const SLA = 0.9;
const CONF = 0.9;
const MEANS = [0.55, 0.8, SLA + 0.05];
const clip = (x: number) => Math.min(0.995, Math.max(0.01, x));
const prob = (p: number, d: number) => clip(MEANS[p] + SLOPES[p] * (0.5 - d));
const BEST = 2; // the true best (dearest) plan

const args = process.argv.slice(2);
const opt = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const REPS = Number(opt("reps") ?? 100);
const SIZES = (opt("sizes") ?? "100,1000").split(",").map(Number);
const z = normalQuantile(CONF);

interface Internals {
  records: Array<{ features: number[]; observations: Array<{ planKey: string; quality: number }>; winner?: string }>;
  estimateFeatures(f: number[], exclude?: unknown): unknown[];
  choose(all: unknown[], minQuality: number): { estimate: { planKey: string } };
  realized(r: unknown, key: string, forceCredit?: boolean): number;
}

async function rep(n: number, seed: number, exhaustive: boolean) {
  const rnd = mulberry32(seed);
  const d = Array.from({ length: n }, () => rnd());
  const features: number[][] = d.map(() => [rnd() + 0.05, rnd() + 0.05]);
  const f1 = IDS.map((_, p) => d.map((x) => (rnd() < prob(p, x) ? 1 : 0)));
  const planner = createPlanner<number, number>({
    space: SPACE,
    pipeline: async (q, plan) => {
      const p = IDS.indexOf(planKey(plan));
      return { output: f1[p][q], costUsd: COSTS[p] };
    },
    scorer: (_q, o) => o,
    sla: { minQuality: SLA, confidence: CONF },
    featurize: (q) => features[q],
    feedbackRate: 0,
    concurrency: 1,
    ...(exhaustive ? { calibration: { strategy: "exhaustive" as const } } : {}),
  });
  const report = await planner.calibrate(Array.from({ length: n }, (_, i) => i));
  const I = planner as unknown as Internals;
  const replay = I.records.filter((r) => r.winner !== undefined || r.observations.length > 1);
  const t = report.qualityThreshold;
  const chosenCount = [0, 0, 0];
  let measured = 0;
  let credited = 0;
  let zeroed = 0;
  let sum = 0;
  for (const r of replay) {
    const key = I.choose(I.estimateFeatures(r.features, r), t).estimate.planKey;
    chosenCount[IDS.indexOf(key)]++;
    const was = r.observations.some((o) => o.planKey === key && !Number.isNaN(o.quality));
    const q = I.realized(r, key);
    sum += q;
    if (was) measured++;
    else if (q > 0 && r.winner !== undefined && q === r.observations.find((o) => o.planKey === r.winner)?.quality) credited++;
    else zeroed++;
  }
  const boundOf = (mean: number) => wilson(mean, replay.length, z).lo;
  const meanReplay = sum / Math.max(1, replay.length);
  // counterfactual (b): force the true best plan as the replay choice for every record, with realized() as the planner has it
  const bestKey = IDS[BEST];
  const forced = replay.reduce((a, r) => a + I.realized(r, bestKey), 0) / Math.max(1, replay.length);
  const forcedCredit = replay.reduce((a, r) => a + I.realized(r, bestKey, true), 0) / Math.max(1, replay.length);
  const bestMeasured = replay.filter((r) => r.observations.some((o) => o.planKey === bestKey && !Number.isNaN(o.quality))).length;
  // exact truth of the frozen policy over a grid of difficulties
  const G = 400;
  const base = features.length;
  for (let k = 0; k < G; k++) features.push([rnd() + 0.05, rnd() + 0.05]);
  let truth = 0;
  for (let k = 0; k < G; k++) truth += prob(IDS.indexOf(planKey(planner.decide(base + k).plan)), (k + 0.5) / G);
  truth /= G;
  return {
    certified: report.slaCertified,
    assumeMonotone: report.assumeMonotone,
    violation: report.monotonicityViolationRate,
    pruned: report.prunedPlans.length,
    runs: report.runs,
    threshold: t,
    boundReported: report.certifiedQuality,
    replayN: replay.length,
    meanReplay,
    boundReplay: boundOf(meanReplay),
    chosen: chosenCount.map((c) => c / Math.max(1, replay.length)),
    measuredShare: measured / Math.max(1, replay.length),
    creditedShare: credited / Math.max(1, replay.length),
    zeroShare: zeroed / Math.max(1, replay.length),
    bestMeasuredShare: bestMeasured / Math.max(1, replay.length),
    forcedMean: forced,
    forcedBound: boundOf(forced),
    forcedCreditMean: forcedCredit,
    forcedCreditBound: boundOf(forcedCredit),
    safe: report.policy === "safe-plan",
    truth,
  };
}

const m = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const pc = (x: number) => `${(100 * x).toFixed(1)}%`;
console.log(`== certify diagnose (SYNTHETIC, known truth): power world, 0/1 scores, random features, SLA ${SLA}, confidence ${CONF}, best plan (max, cost 5) truly ${MEANS[2]} (cheap 0.55, mid 0.80), ${REPS} independent repetitions per row`);
for (const n of SIZES) {
  for (const mode of ["adaptive (default)", "exhaustive (counterfactual a: every plan measured on every question)"] as const) {
    const rs: Array<Awaited<ReturnType<typeof rep>>> = [];
    for (let r = 0; r < REPS; r++) rs.push(await rep(n, 424242 + n * 13 + r, mode.startsWith("exhaustive")));
    const share = (f: (x: Awaited<ReturnType<typeof rep>>) => boolean) => pc(rs.filter(f).length / rs.length);
    console.log(`\n-- n = ${n}, calibration ${mode}`);
    console.log(`in-sample certified ${share((x) => x.certified)}; safe-plan policy ${share((x) => x.safe)}; assumeMonotone true ${share((x) => x.assumeMonotone)} (mean violation rate ${pc(m(rs.map((x) => x.violation)))}); pruned plans mean ${m(rs.map((x) => x.pruned)).toFixed(2)}; calibration runs mean ${m(rs.map((x) => x.runs)).toFixed(0)} of ${3 * n}`);
    console.log(`threshold picked: mean ${m(rs.map((x) => x.threshold)).toFixed(3)}; the planner's reported bound (certifiedQuality) mean ${m(rs.map((x) => x.boundReported)).toFixed(3)} vs SLA ${SLA}; frozen policy's TRUE expected quality mean ${m(rs.map((x) => x.truth)).toFixed(3)}`);
    console.log(`replay at that threshold (${m(rs.map((x) => x.replayN)).toFixed(0)} records): replayed mean quality ${m(rs.map((x) => x.meanReplay)).toFixed(3)}; chosen plan share cheap/mid/max ${m(rs.map((x) => x.chosen[0])).toFixed(2)}/${m(rs.map((x) => x.chosen[1])).toFixed(2)}/${m(rs.map((x) => x.chosen[2])).toFixed(2)}; chosen plan MEASURED on that question ${pc(m(rs.map((x) => x.measuredShare)))}, winner-credited ${pc(m(rs.map((x) => x.creditedShare)))}, counted as ZERO (unmeasured) ${pc(m(rs.map((x) => x.zeroShare)))}`);
    console.log(`the true best plan (max) was measured on ${pc(m(rs.map((x) => x.bestMeasuredShare)))} of the replay questions`);
    console.log(`counterfactual (b) force the true best plan as the replay choice for every question: replayed mean ${m(rs.map((x) => x.forcedMean)).toFixed(3)}, Wilson bound mean ${m(rs.map((x) => x.forcedBound)).toFixed(3)}, bound >= SLA in ${share((x) => x.forcedBound >= SLA)}; with winner credit forced on (b'): mean ${m(rs.map((x) => x.forcedCreditMean)).toFixed(3)}, bound >= SLA in ${share((x) => x.forcedCreditBound >= SLA)}`);
  }
}
console.log("\n(SYNTHETIC, one world family, instrument only: the planner was not changed. Private fields are read through a cast.)");
