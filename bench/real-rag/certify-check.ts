/**
 * SYNTHETIC checks for the split-based certification harness: tables are generated from KNOWN true quality per plan (Bernoulli
 * scores), so a certification can be called right or wrong. No cache, no network, no key.
 *   npm run bench:rag:certify:check
 */
import { enumeratePlans, planKey } from "../../src/index.js";
import { mulberry32, type Table } from "./analyze-core.js";
import { aggregate, runSplit, split3, type SplitResult } from "./certify-core.js";

let failures = 0;
const check = (msg: string, ok: boolean, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}${!ok && detail ? `\n     ${detail}` : ""}`);
  if (!ok) failures++;
};
const SPACE = { model: ["cheap", "mid", "max"] } as const;
const ids = enumeratePlans(SPACE).map((p) => planKey(p));
const maxPlan = ids[2];

/** One synthetic workload: N questions, plan p scores 1 with probability trueQ[p] (cost fixed per plan), random features. */
function world(trueQ: number[], costs: number[], n: number, seed: number): { table: Table; features: number[][] } {
  const rnd = mulberry32(seed);
  return {
    table: { plans: ids, f1: trueQ.map((m) => Array.from({ length: n }, () => (rnd() < m ? 1 : 0))), cost: costs.map((c) => Array.from({ length: n }, () => c)) },
    features: Array.from({ length: n }, () => Array.from({ length: 8 }, () => rnd())),
  };
}
async function many(trueQ: number[], costs: number[], sla: number, splits: number): Promise<SplitResult[]> {
  const rs: SplitResult[] = [];
  const truth = Object.fromEntries(ids.map((id, i) => [id, trueQ[i]]));
  for (let s = 0; s < splits; s++) {
    const w = world(trueQ, costs, 100, 500 + s); // a FRESH world per split: independent draws
    rs.push(await runSplit({ ...w, space: SPACE, ...split3(100, [40, 30, 30], 9000 + s), sla, confidence: 0.9, maxPlan, trueQuality: truth }));
  }
  return rs;
}
const SPLITS = 300;
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

check("split3 gives disjoint A/B/C of the asked sizes, seeded", (() => {
  const a = split3(100, [40, 30, 30], 1);
  const all = new Set([...a.A, ...a.B, ...a.C]);
  return a.A.length === 40 && a.B.length === 30 && a.C.length === 30 && all.size === 100 && JSON.stringify(a) === JSON.stringify(split3(100, [40, 30, 30], 1)) && JSON.stringify(a) !== JSON.stringify(split3(100, [40, 30, 30], 2));
})());

// Null case: the best plan is truly BELOW the SLA, so every certification is false.
{
  const rs = await many([0.5, 0.85, 0.88], [1, 2, 5], 0.9, SPLITS);
  const g = aggregate(rs);
  console.log(`     (SYNTHETIC null, ${SPLITS} splits, best plan truly 0.88 < SLA 0.90: false-certification rate in-sample ${pct(g.falseCertInSample)}, split ${pct(g.falseCertSplit)}; always-max fallback ${pct(g.fallbackShare)})`);
  check("null case: the split-certified false-certification rate stays near or below 1 - confidence (<= 15%)", g.falseCertSplit <= 0.15, pct(g.falseCertSplit));
}
// Null case where the plan sits exactly AT the SLA: certification should be rare as well.
{
  const rs = await many([0.5, 0.85, 0.9], [1, 2, 5], 0.9, SPLITS);
  const g = aggregate(rs);
  console.log(`     (SYNTHETIC, best plan truly 0.90 = SLA 0.90: certified in-sample ${pct(g.inSampleCertRate)}, split ${pct(g.splitCertRate)})`);
  check("a plan exactly at the SLA is certified rarely (split rate <= 15%)", g.splitCertRate <= 0.15, pct(g.splitCertRate));
}
// Power case: the best cheap plan is truly well above the SLA.
{
  const rs = await many([0.5, 0.93, 0.97], [1, 2, 5], 0.7, SPLITS);
  const g = aggregate(rs);
  console.log(`     (SYNTHETIC power, ${SPLITS} splits, mid plan truly 0.93 vs SLA 0.70: certification in-sample ${pct(g.inSampleCertRate)}, split ${pct(g.splitCertRate)}; miss among split-certified ${Number.isNaN(g.missAmongSplit) ? "n/a" : pct(g.missAmongSplit)}; fallback ${pct(g.fallbackShare)})`);
  check("power case: a plan truly well above the SLA is split-certified in more than half of the splits", g.splitCertRate > 0.5, pct(g.splitCertRate));
  check("power case: the best fixed plan chosen on A is the cheap feasible plan most of the time", rs.filter((r) => r.fixed.plan === ids[1]).length > SPLITS * 0.5);
}
// Determinism.
{
  const w = world([0.5, 0.93, 0.97], [1, 2, 5], 100, 1);
  const args = { ...w, space: SPACE, ...split3(100, [40, 30, 30], 3), sla: 0.7, confidence: 0.9, maxPlan };
  check("runSplit is deterministic", JSON.stringify(await runSplit(args)) === JSON.stringify(await runSplit(args)));
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC known-truth worlds)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
