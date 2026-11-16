/**
 * Checks for planner.certify() (SYNTHETIC: generated Bernoulli scores with a known true mean; no model, no network, no key).
 *   npm run bench:certify
 */
import { createPlanner, planKey } from "../src/index.js";

let failures = 0;
const check = (ok: boolean, msg: string) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failures++;
};
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A planner over ONE plan whose outcomes come from a lookup we refill: the served policy is that plan for everything. */
function oneplan(opts: { minQuality: number; confidence?: number; scorerErrorRate?: number } = { minQuality: 0.9 }) {
  const table: number[] = [];
  const p = createPlanner<number, number>({
    space: { model: ["only"] },
    pipeline: async (q) => ({ output: table[q], costUsd: 2 }),
    scorer: (_q, o) => o,
    sla: { minQuality: opts.minQuality, ...(opts.confidence ? { confidence: opts.confidence } : {}) },
    ...(opts.scorerErrorRate !== undefined ? { scorerErrorRate: opts.scorerErrorRate } : {}), // an explicit undefined would override the planner's default
    featurize: (q) => [1, q % 7, (q * 3) % 5],
    feedbackRate: 0,
    concurrency: 4,
  });
  return { p, table };
}
const bern = (r: () => number, m: number) => (r() < m ? 1 : 0);

// ---- 1. statistics: SYNTHETIC grid, 0/1 scores, one frozen plan with a known true mean, 1000 repetitions per cell
{
  const REPS = 1000;
  const rate = async (m: number, n: number) => {
    const { p, table } = oneplan({ minQuality: 0.9, confidence: 0.9 });
    const r = rng(1000 + Math.round(m * 1000) + n);
    table.length = 0;
    for (let i = 0; i < 30; i++) table.push(bern(r, m));
    await p.calibrate(Array.from({ length: 30 }, (_, i) => i)); // the policy exists (one plan), calibration data irrelevant to certify()
    let ok = 0;
    for (let rep = 0; rep < REPS; rep++) {
      table.length = 30;
      for (let i = 0; i < n; i++) table.push(bern(r, m));
      const c = await p.certify(Array.from({ length: n }, (_, i) => 30 + i));
      if (c.certified) ok++;
    }
    return ok / REPS;
  };
  const atSla = await rate(0.9, 100);
  const below = await rate(0.88, 100);
  const power150 = await rate(0.95, 150);
  const power100 = await rate(0.95, 100);
  console.log(`     (SYNTHETIC, ${REPS} repetitions per cell, 0/1 scores, SLA 0.9, confidence 0.9: true 0.90 at n=100 certified ${(100 * atSla).toFixed(1)}%; true 0.88 at n=100 ${(100 * below).toFixed(1)}%; true 0.95 at n=100 ${(100 * power100).toFixed(1)}%, at n=150 ${(100 * power150).toFixed(1)}%; standard error about 1 point at 10%)`);
  check(atSla <= 0.12, "false certification of a plan exactly at the SLA is at most 12% (by design about 10%)");
  check(below <= 0.08, "false certification of a plan truly 2 points below the SLA is at most 8%");
  check(power150 >= 0.7, "a plan 5 points above the SLA is certified in at least 70% of runs at n = 150");
}

// ---- 2. behaviour
{
  const { p, table } = oneplan({ minQuality: 0.5, confidence: 0.9 });
  const r = rng(7);
  for (let i = 0; i < 200; i++) table.push(bern(r, 0.9));
  await p.calibrate(Array.from({ length: 40 }, (_, i) => i));
  const probe = Array.from({ length: 30 }, (_, i) => i);
  const before = probe.map((q) => planKey(p.decide(q).plan));
  const fresh = Array.from({ length: 100 }, (_, i) => 100 + i);
  const a = await p.certify(fresh);
  const b = await p.certify(fresh);
  const after = probe.map((q) => planKey(p.decide(q).plan));
  check(JSON.stringify(before) === JSON.stringify(after), "decide() outputs are identical before and after certify()");
  check(JSON.stringify(a) === JSON.stringify(b), "certify() twice on the same samples gives the same result (no randomness, no learning)");
  check(Math.abs(Object.values(a.planShares).reduce((x, y) => x + y, 0) - 1) < 1e-12 && a.fellBackToSafe >= 0 && a.fellBackToSafe <= 1, "planShares sum to 1 and fellBackToSafe is a share");
  check(a.n === 100 && Math.abs(a.costUsd - 200) < 1e-9, "n and costUsd count exactly the fresh runs (100 runs at cost 2)");
  check(a.certified && a.lowerBound < a.meanQuality && a.lowerBound >= 0.5, "a plan well above a loose SLA is certified and the bound sits below the mean");
  check(a.warnings.some((w) => /ONCE per frozen policy/.test(w)) && a.warnings.some((w) => /drift is not covered/.test(w)), "the use-once and drift warnings are always present");
  const small = await p.certify(fresh.slice(0, 20));
  check(small.warnings.some((w) => /only 20 fresh samples/.test(w)), "fewer than 60 samples warns");
  const strict = await p.certify(fresh, { confidence: 0.99 });
  check(strict.lowerBound < a.lowerBound && strict.confidence === 0.99, "a higher confidence gives a lower bound");
}
// not certified warning, and the safe-plan share
{
  const { p, table } = oneplan({ minQuality: 0.99, confidence: 0.9 });
  const r = rng(11);
  for (let i = 0; i < 300; i++) table.push(bern(r, 0.7));
  const rep = await p.calibrate(Array.from({ length: 60 }, (_, i) => i));
  const c = await p.certify(Array.from({ length: 100 }, (_, i) => 100 + i));
  check(!c.certified && c.warnings.some((w) => /not certified/.test(w)), "an unreachable SLA is not certified and says so");
  check(rep.policy === "safe-plan" ? c.fellBackToSafe === 1 : c.fellBackToSafe <= 1, "in safe mode every fresh question counts as served by the safe plan (fellBackToSafe = 1)");
}
// throw cases and counters
{
  const { p, table } = oneplan({ minQuality: 0.5 });
  for (let i = 0; i < 50; i++) table.push(1);
  await p.calibrate(Array.from({ length: 20 }, (_, i) => i));
  let m1 = "";
  try { await p.certify([]); } catch (e) { m1 = (e as Error).message; }
  check(/at least one fresh sample/.test(m1), "empty samples throw");
  const nan = createPlanner<number, number>({ space: { model: ["only"] }, pipeline: async () => ({ output: 1, costUsd: 1 }), scorer: () => NaN, sla: { minQuality: 0.5 }, featurize: (q) => [1, q], feedbackRate: 0 });
  // calibrate with a good scorer first by importing state from `p`
  nan.importState(p.exportState());
  let m2 = "";
  try { await nan.certify([1, 2, 3]); } catch (e) { m2 = (e as Error).message; }
  check(/non-finite/.test(m2), "a NaN scorer throws");
  const boom = createPlanner<number, number>({ space: { model: ["only"] }, pipeline: async () => { throw new Error("boom"); }, scorer: () => 1, sla: { minQuality: 0.5 }, featurize: (q) => [1, q], feedbackRate: 0 });
  boom.importState(p.exportState());
  let m3 = "";
  try { await boom.certify([1, 2, 3]); } catch (e) { m3 = (e as Error).message; }
  check(m3 === "boom", "a pipeline error propagates");
  const fresh = createPlanner<number, number>({ space: { model: ["only"] }, pipeline: async () => ({ output: 1, costUsd: 1 }), scorer: () => 1, sla: { minQuality: 0.5 } });
  let m4 = "";
  try { await fresh.certify([1]); } catch (e) { m4 = (e as Error).message; }
  check(/call calibrate\(\) first/.test(m4), "certify() before any calibration fails like decide()");
  let m5 = "";
  try { await p.certify([30, 31], { confidence: 1.5 }); } catch (e) { m5 = (e as Error).message; }
  check(/confidence must be in/.test(m5), "a confidence outside (0, 1) throws");
  const noisy = oneplan({ minQuality: 0.5, scorerErrorRate: 0.1 });
  for (let i = 0; i < 100; i++) noisy.table.push(1);
  await noisy.p.calibrate(Array.from({ length: 20 }, (_, i) => i));
  const w = await noisy.p.certify(Array.from({ length: 70 }, (_, i) => 20 + i));
  check(w.warnings.some((x) => /scorerErrorRate is set but certify\(\) does not correct/.test(x)), "scorerErrorRate > 0 warns that it is not corrected");
  const before = p.report().calibrationRuns;
  await p.certify(Array.from({ length: 10 }, (_, i) => 20 + i));
  check(p.report().calibrationRuns === before + 10, "the stats counters grow by the runs certify() made (documented)");
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
