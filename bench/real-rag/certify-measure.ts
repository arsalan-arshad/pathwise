/**
 * T-029 prototype and measurements (bench-only, NO change to src/): "certify what you serve, by measurement on fresh questions".
 *
 *   npx tsx bench/real-rag/certify-measure.ts [--reps 300] [--real]
 *
 * `certifyByMeasurement` is the API sketch as a bench function: run the policy that would be SERVED (a decide function, or one
 * recommended plan) on FRESH questions with the real pipeline and scorer, then one Wilson lower bound of the mean quality
 * at `confidence`. No threshold search, no replay, no credit for unmeasured plans.
 * Parts (all SYNTHETIC with known truth unless marked REAL):
 *   1. n_fresh needed: the smallest n whose Wilson bound clears the SLA when the observed mean equals the true mean (computed,
 *      not measured; about half the time more is needed);
 *   2. certification rate and false-certification rate of a fixed plan vs n_fresh, 0/1 and graded scores (2000 repetitions);
 *   3. composition: replay claim vs calibrate + certify decide() vs recommend() + certify the recommended plan, per world and size;
 *   4. (--real) the T-025 HotpotQA splits (REAL, offline): recommend() on A, certify on B, miss on C.
 */
import { createPlanner, enumeratePlans, planKey, type Plan } from "../../src/index.js";
import { normalQuantile } from "../../src/stats.js";
import { mulberry32 } from "./analyze-core.js";
import { wilson } from "./single-plan-core.js";

// ------------------------------------------------------------------ the prototype
export interface CertifyResult {
  meanQuality: number;
  lowerBound: number;
  certified: boolean;
  n: number;
  costUsd: number;
  planShares: Record<string, number>;
  fellBackToSafe: boolean;
}
export async function certifyByMeasurement<I>(a: {
  decide: (input: I) => Plan;
  run: (input: I, plan: Plan) => Promise<{ quality: number; costUsd: number }>;
  fresh: I[];
  minQuality: number;
  confidence: number;
  fellBackToSafe?: boolean;
}): Promise<CertifyResult> {
  if (a.fresh.length === 0) throw new Error("certifyByMeasurement needs at least one fresh question");
  let sum = 0;
  let cost = 0;
  const shares: Record<string, number> = {};
  for (const q of a.fresh) {
    const plan = a.decide(q);
    const r = await a.run(q, plan);
    if (!Number.isFinite(r.quality) || !Number.isFinite(r.costUsd)) throw new Error("non-finite quality or cost");
    sum += Math.min(1, Math.max(0, r.quality));
    cost += r.costUsd;
    shares[planKey(plan)] = (shares[planKey(plan)] ?? 0) + 1 / a.fresh.length;
  }
  const mean = sum / a.fresh.length;
  const lower = wilson(mean, a.fresh.length, normalQuantile(a.confidence)).lo;
  return { meanQuality: mean, lowerBound: lower, certified: lower >= a.minQuality, n: a.fresh.length, costUsd: cost, planShares: shares, fellBackToSafe: a.fellBackToSafe ?? false };
}

// ------------------------------------------------------------------ synthetic world (same family as certify-sweep.ts)
const SPACE = { model: ["cheap", "mid", "max"] } as const;
const PLANS = enumeratePlans(SPACE);
const IDS = PLANS.map((p) => planKey(p));
const COSTS = [1, 2, 5];
const SLOPES = [0.4, 0.3, 0.1];
const SLA = 0.9;
const CONF = 0.9;
const clip = (x: number) => Math.min(0.995, Math.max(0.01, x));
type World = "below" | "at" | "power";
const meansOf = (w: World) => [0.55, 0.8, w === "below" ? SLA - 0.02 : w === "at" ? SLA : SLA + 0.05];
const prob = (m: number[], p: number, d: number) => clip(m[p] + SLOPES[p] * (0.5 - d));
const exact = (m: number[]) => m.map((_, p) => Array.from({ length: 4000 }, (_, k) => prob(m, p, (k + 0.5) / 4000)).reduce((x, y) => x + y, 0) / 4000);
function gaussian(r: () => number) { return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
function gamma(shape: number, r: () => number): number {
  if (shape < 1) return gamma(shape + 1, r) * Math.pow(r(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do { x = gaussian(r); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
const beta = (a: number, b: number, r: () => number) => { const x = gamma(a, r); return x / (x + gamma(b, r)); };
type Scores = "binary" | "graded";
const score = (s: number, kind: Scores, r: () => number) => (kind === "binary" ? (r() < s ? 1 : 0) : beta(s * 4, (1 - s) * 4, r));

const args = process.argv.slice(2);
const REPS = Number(args[args.indexOf("--reps") + 1] || 300);
const pct = (x: number, n: number) => `${((100 * x) / n).toFixed(1).padStart(5)}%`;
const z = normalQuantile(CONF);

// ------------------------------------------------------------------ part 1: n needed (computed)
console.log(`== certify-measure (SYNTHETIC unless marked REAL), SLA ${SLA}, confidence ${CONF}, ${REPS} repetitions for part 3`);
console.log("\n-- part 1: fresh questions needed so that the Wilson lower bound clears the SLA when the OBSERVED mean equals the true mean (computed, not measured; about half the time more is needed). The bound depends on the mean only, so 0/1 and graded scores need the same n at the same observed mean.");
for (const margin of [0.02, 0.05, 0.08, 0.1]) {
  let n = 1;
  while (wilson(SLA + margin, n, z).lo < SLA && n < 100000) n++;
  console.log(`  true quality SLA + ${margin.toFixed(2)} (${(SLA + margin).toFixed(2)}): n >= ${n}`);
}

// ------------------------------------------------------------------ part 2: fixed plan, certification vs n_fresh
console.log("\n-- part 2: a fixed plan certified by measurement on n fresh questions (iid scores with the stated true mean; 2000 repetitions per row)");
console.log("true mean | scores | " + [20, 30, 50, 75, 100, 150, 200, 300, 500].map((n) => String(n).padStart(6)).join(" "));
for (const m of [0.88, 0.9, 0.95, 0.98]) {
  for (const kind of ["binary", "graded"] as Scores[]) {
    const r = mulberry32(900 + Math.round(m * 1000));
    const cells: string[] = [];
    for (const n of [20, 30, 50, 75, 100, 150, 200, 300, 500]) {
      let ok = 0;
      for (let i = 0; i < 2000; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += score(m, kind, r);
        if (wilson(s / n, n, z).lo >= SLA) ok++;
      }
      cells.push(pct(ok, 2000));
    }
    console.log(`${m.toFixed(2)}      | ${kind.padEnd(6)} | ${cells.join(" ")}   (${m < SLA ? "FALSE certification rate" : m === SLA ? "at the SLA: certification = false-certification" : "certification rate (power)"})`);
  }
}

// ------------------------------------------------------------------ part 3: composition on the synthetic world
async function cellRep(world: World, scores: Scores, nCal: number, nFresh: number, seed: number) {
  const r = mulberry32(seed);
  const m = meansOf(world);
  const tm = exact(m);
  const mk = (n: number) => {
    const d = Array.from({ length: n }, () => r());
    return { d, features: d.map(() => [r() + 0.05, r() + 0.05]), f1: IDS.map((_, p) => d.map((x) => score(prob(m, p, x), scores, r))) };
  };
  const cal = mk(nCal);
  const fresh = mk(nFresh);
  const feats = [...cal.features, ...fresh.features];
  const f1 = cal.f1.map((row, p) => [...row, ...fresh.f1[p]]);
  const make = () =>
    createPlanner<number, number>({
      space: SPACE,
      pipeline: async (q, plan) => ({ output: f1[IDS.indexOf(planKey(plan))][q], costUsd: COSTS[IDS.indexOf(planKey(plan))] }),
      scorer: (_q, o) => o,
      sla: { minQuality: SLA, confidence: CONF },
      featurize: (q) => feats[q],
      feedbackRate: 0,
      concurrency: 1,
    });
  const run = async (q: number, plan: Plan) => {
    const p = IDS.indexOf(planKey(plan));
    return { quality: f1[p][q], costUsd: COSTS[p] };
  };
  const freshIdx = Array.from({ length: nFresh }, (_, i) => nCal + i);
  const calIdx = Array.from({ length: nCal }, (_, i) => i);
  // (i) replay claim + (ii) calibrate then certify decide() on fresh
  const planner = make();
  const rep = await planner.calibrate(calIdx);
  const grid = 400;
  const base = feats.length;
  for (let k = 0; k < grid; k++) feats.push([r() + 0.05, r() + 0.05]);
  let truth = 0;
  for (let k = 0; k < grid; k++) truth += prob(m, IDS.indexOf(planKey(planner.decide(base + k).plan)), (k + 0.5) / grid);
  truth /= grid;
  const cert = await certifyByMeasurement({ decide: (q: number) => planner.decide(q).plan, run, fresh: freshIdx, minQuality: SLA, confidence: CONF, fellBackToSafe: rep.policy === "safe-plan" });
  // (iii) recommend then certify the one recommended plan
  const p2 = make();
  const rec = await p2.recommend(calIdx, { seed });
  let recCert: CertifyResult | undefined;
  if (rec) recCert = await certifyByMeasurement({ decide: () => rec.plan, run, fresh: freshIdx, minQuality: SLA, confidence: CONF });
  return {
    replay: { cert: rep.slaCertified, bad: truth < SLA, calls: rep.runs },
    dec: { cert: cert.certified, bad: truth < SLA, calls: rep.runs + nFresh, cost: cert.costUsd },
    rec: { plan: !!rec, cert: !!recCert?.certified, bad: rec ? tm[IDS.indexOf(rec.planKey)] < SLA : false, calls: rec ? rec.callsUsed + nFresh : 0, cost: recCert?.costUsd ?? 0 },
  };
}
console.log("\n-- part 3: composition (SYNTHETIC, known truth). Worlds: best plan truly SLA-0.02 (below), = SLA (at), SLA+0.05 (power). 'bad' = certified while the served policy's / the recommended plan's exact true quality < SLA. n_fresh = 100.");
console.log("world  scores  n_cal | replay claim: cert / FALSE  calls | calibrate+certify(decide): cert / FALSE  calls | recommend+certify(plan): returned / cert / FALSE  calls");
for (const world of ["below", "at", "power"] as World[])
  for (const scores of ["binary", "graded"] as Scores[])
    for (const nCal of [100, 300]) {
      const acc = { rc: 0, rb: 0, rcalls: 0, dc: 0, db: 0, dcalls: 0, p: 0, c: 0, b: 0, ccalls: 0 };
      for (let i = 0; i < REPS; i++) {
        const x = await cellRep(world, scores, nCal, 100, 31000 + nCal * 17 + i);
        acc.rc += x.replay.cert ? 1 : 0; acc.rb += x.replay.cert && x.replay.bad ? 1 : 0; acc.rcalls += x.replay.calls;
        acc.dc += x.dec.cert ? 1 : 0; acc.db += x.dec.cert && x.dec.bad ? 1 : 0; acc.dcalls += x.dec.calls;
        acc.p += x.rec.plan ? 1 : 0; acc.c += x.rec.cert ? 1 : 0; acc.b += x.rec.cert && x.rec.bad ? 1 : 0; acc.ccalls += x.rec.calls;
      }
      console.log(`${world.padEnd(6)} ${scores.padEnd(7)} ${String(nCal).padStart(4)}  | ${pct(acc.rc, REPS)} / ${pct(acc.rb, REPS)}  ${(acc.rcalls / REPS).toFixed(0).padStart(5)} | ${pct(acc.dc, REPS)} / ${pct(acc.db, REPS)}  ${(acc.dcalls / REPS).toFixed(0).padStart(5)} | ${pct(acc.p, REPS)} / ${pct(acc.c, REPS)} / ${pct(acc.b, REPS)}  ${(acc.ccalls / REPS).toFixed(0).padStart(5)}`);
    }
console.log("(calls = pipeline runs: replay claim = adaptive calibration; calibrate+certify = calibration + n_fresh served-policy runs; recommend+certify = racing calls + n_fresh runs of one plan. Standard error about 1.7 points at 10% and 2.9 at 50%.)");

// ------------------------------------------------------------------ part 4: REAL HotpotQA splits (T-025 protocol, recommend on A)
if (args.includes("--real")) {
  process.env.PATHWISE_BENCH_OFFLINE = "1";
  const { hashedTextFeaturizer } = await import("../../src/index.js");
  const { BASELINES, SPACE: RSPACE } = await import("./config.js");
  const { loadHotpot, sample } = await import("./dataset.js");
  const { OfflineMiss } = await import("./gemini.js");
  const { ragPipeline, scorer } = await import("./pipeline.js");
  const core = await import("./analyze-core.js");
  const cc = await import("./certify-core.js");
  const { calibration } = sample(await loadHotpot(), 100, 200, 7); // the test split is drawn but never read
  const rplans = enumeratePlans(RSPACE);
  const rids = rplans.map((p) => planKey(p));
  const table: import("./analyze-core.js").Table = { plans: rids, f1: [], cost: [] };
  let missing = 0;
  for (const plan of rplans) {
    const f: number[] = [];
    const c: number[] = [];
    for (const q of calibration) {
      try { const x = await ragPipeline(q, plan); f.push(scorer(q, x.output)); c.push(x.costUsd); } catch (e) { if (!(e instanceof OfflineMiss)) throw e; missing++; }
    }
    table.f1.push(f); table.cost.push(c);
  }
  console.log("\n-- part 4 (REAL, offline, HotpotQA, 100 fully-tabled questions, 200 seeded splits 40/30/30, test never read): recommend() on A, certify that plan on B (one Wilson bound), miss = mean on C below the SLA");
  if (missing > 0) console.log(`incomplete table (${missing} missing): nothing measured`);
  else {
    const hashed = hashedTextFeaturizer();
    const feats = calibration.map((q) => hashed(q.question));
    const maxPlan = planKey(BASELINES.max);
    for (const sla of [0.725, 0.687, 0.6]) {
      let ret = 0, cert = 0, missCert = 0, calls = 0;
      for (let s = 0; s < 200; s++) {
        const sp = cc.split3(100, [40, 30, 30], 7000 + s);
        const planner = createPlanner<number, number>({
          space: RSPACE,
          pipeline: async (q, plan) => { const p = rids.indexOf(planKey(plan)); return { output: table.f1[p][q], costUsd: table.cost[p][q] }; },
          scorer: (_q, o) => o, sla: { minQuality: sla, confidence: CONF }, featurize: (q) => feats[q], feedbackRate: 0, concurrency: 1,
        });
        const rec = await planner.recommend(sp.A, { seed: s });
        if (!rec) continue;
        ret++;
        calls += rec.callsUsed + sp.B.length;
        const p = rids.indexOf(rec.planKey);
        const c = await certifyByMeasurement({ decide: () => rec.plan, run: async (q) => ({ quality: table.f1[p][q], costUsd: table.cost[p][q] }), fresh: sp.B, minQuality: sla, confidence: CONF });
        if (c.certified) { cert++; if (core.meanOf(sp.C.map((q) => table.f1[p][q])) < sla) missCert++; }
      }
      console.log(`SLA ${sla}${sla === 0.6 ? " (LOOSE, harness check, not a product claim)" : ""}: recommend returned a plan in ${ret}/200; certified on B ${cert}/200; of those, mean on C below the SLA ${missCert}/${cert}; mean calls ${ret ? (calls / ret).toFixed(0) : "n/a"} (A=40 racing + 30 fresh)`);
    }
    void maxPlan;
    console.log("(compare T-025 with the replay planner: in-sample certified 3/200 and split-certified 18/200 at 0.725; 7 and 47 at 0.687; 3 and 124 at 0.60. 'Miss on C' is a 30-question mean: not comparable to 10%.)");
  }
}
