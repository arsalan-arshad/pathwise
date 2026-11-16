/**
 * Checks for planner.recommend() (SYNTHETIC workload from examples/workload.ts and generated scores; no network).
 *   npm run bench:recommend
 */
import { createPlanner, hashedTextFeaturizer, planKey } from "../src/index.js";
import { createWorkload, ragPipeline, SPACE } from "./workload.js";

let failures = 0;
const check = (ok: boolean, msg: string) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failures++;
};
type Q = { text: string; difficulty: number };
const feats = hashedTextFeaturizer();
const mk = (sla: { minQuality: number; confidence?: number }) =>
  createPlanner<Q, { correct: boolean }>({
    space: SPACE,
    pipeline: ragPipeline,
    scorer: (_q, o) => (o.correct ? 1 : 0),
    sla,
    featurize: (q) => feats(q.text),
    feedbackRate: 0,
    concurrency: 4,
  });

// 1. Recovers the cheapest feasible plan on the simulated workload; compare with a full table over the same samples.
let agree = 0;
let below = 0;
let certified = 0;
let none = 0;
let calls = 0;
let full = 0;
const SEEDS = 20;
for (let seed = 1; seed <= SEEDS; seed++) {
  const { makeQueries } = createWorkload(seed, {});
  const samples = makeQueries(120);
  const planner = mk({ minQuality: 0.8 });
  // Full table baseline: mean quality and cost per plan over all samples.
  const table = await Promise.all(
    planner.plans.map(async (p) => {
      let q = 0;
      let c = 0;
      for (const s of samples) {
        const r = await ragPipeline(s, p);
        q += r.output.correct ? 1 : 0;
        c += r.costUsd;
      }
      return { key: planKey(p), q: q / samples.length, c: c / samples.length };
    }),
  );
  const fullBest = table.filter((t) => t.q >= 0.8).sort((a, b) => a.c - b.c)[0];
  const rec = await planner.recommend(samples, { seed });
  full += planner.plans.length * samples.length;
  if (!rec) {
    none++;
    continue;
  }
  calls += rec.callsUsed;
  if (rec.certified) certified++;
  if (fullBest && rec.planKey === fullBest.key) agree++;
  if ((table.find((t) => t.key === rec.planKey)?.q ?? 0) < 0.8) below++;
}
console.log(`     (SYNTHETIC, ${SEEDS} workloads x 120 samples, SLA 0.8: same plan as the full table ${agree}/${SEEDS}, none ${none}, below SLA ${below}, certified ${certified}, calls ${calls} of ${full})`);
check(agree >= SEEDS * 0.8, "recommend() returns the full-table plan in at least 80% of simulated workloads");
check(calls < full, "racing uses fewer runs than the full table");

// 2. Nothing feasible -> undefined, never an invented plan.
{
  const { makeQueries } = createWorkload(3, {});
  const planner = mk({ minQuality: 1.01 });
  check((await planner.recommend(makeQueries(40))) === undefined, "an unreachable SLA returns undefined");
}
// 3. Certification needs evidence: tiny samples cannot certify; the warning says so.
{
  const { makeQueries } = createWorkload(4, {});
  const rec = await mk({ minQuality: 0.6 }).recommend(makeQueries(12), { confidence: 0.9 });
  check(!!rec && !rec.certified && rec.warnings.some((w) => /not certified/.test(w)) && rec.warnings.some((w) => /only 12 samples/.test(w)), "12 samples: not certified, and the warnings say so");
}
// 4. basis "full-table" when nothing can be eliminated before the questions run out (minSamples above the sample count).
{
  const { makeQueries } = createWorkload(5, {});
  const planner = mk({ minQuality: 0.5 });
  const rec = await planner.recommend(makeQueries(15), { minSamples: 50 });
  check(!!rec && rec.basis === "full-table" && rec.callsUsed === rec.fullTableCalls, "with no early elimination the basis is full-table and calls equal the full table");
}
// 5. Determinism, and decide() is untouched by recommend().
{
  const { makeQueries } = createWorkload(6, {});
  const samples = makeQueries(60);
  const a = mk({ minQuality: 0.8 });
  await a.calibrate(samples);
  const probe = makeQueries(30);
  const before = probe.map((q) => planKey(a.decide(q).plan));
  const r1 = await a.recommend(samples, { seed: 3 });
  const r2 = await mk({ minQuality: 0.8 }).recommend(samples, { seed: 3 });
  const after = probe.map((q) => planKey(a.decide(q).plan));
  check(JSON.stringify(before) === JSON.stringify(after), "decide() returns the same plans before and after recommend()");
  check(JSON.stringify(r1) === JSON.stringify(r2), "recommend() is deterministic for the same samples and seed");
  const t0 = performance.now();
  for (let i = 0; i < 2000; i++) a.decide(probe[i % probe.length]);
  const ms = (performance.now() - t0) / 2000;
  console.log(`     (decide(): ${ms.toFixed(4)} ms per call, SYNTHETIC workload, this machine)`);
  check(ms < 5, "decide() stays under 5 ms (the printed number is the measurement; 5 ms keeps a slow CI runner from flaking)");
}
// 6. A failing pipeline run propagates (no silent partial answer).
{
  const { makeQueries } = createWorkload(7, {});
  const p = createPlanner<Q, { correct: boolean }>({ space: SPACE, pipeline: async () => { throw new Error("boom"); }, scorer: () => 1, sla: { minQuality: 0.5 } });
  let msg = "";
  try { await p.recommend(makeQueries(5)); } catch (e) { msg = (e as Error).message; }
  check(msg === "boom", "a pipeline error is thrown, not swallowed");
}
// 7. Bad inputs fail loudly: NaN scores and empty samples never produce a plan.
{
  const { makeQueries } = createWorkload(8, {});
  const nan = createPlanner<Q, { correct: boolean }>({ space: SPACE, pipeline: ragPipeline, scorer: () => NaN, sla: { minQuality: 0.5 } });
  let m1 = "";
  try { await nan.recommend(makeQueries(12)); } catch (e) { m1 = (e as Error).message; }
  check(/non-finite/.test(m1), "a NaN scorer is an error, not a plan with NaN quality");
  let m2 = "";
  try { await mk({ minQuality: 0.5 }).recommend([]); } catch (e) { m2 = (e as Error).message; }
  check(/at least one sample/.test(m2), "empty samples are an error");
}
// 8. One survivor keeps being measured until certified: a certain plan certifies before the samples run out.
{
  const one = createPlanner<Q, { correct: boolean }>({ space: { model: ["only"] }, pipeline: async () => ({ output: { correct: true }, costUsd: 1 }), scorer: () => 1, sla: { minQuality: 0.7 } });
  const rec = await one.recommend(createWorkload(9, {}).makeQueries(60), { confidence: 0.9 });
  check(!!rec && rec.certified && rec.samplesUsed < 60, "a single always-right plan certifies with fewer samples than available");
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
