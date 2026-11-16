/**
 * Pure helpers for certify.ts: the split-based certification harness. No I/O.
 *
 * Question: when the planner says "certified" (sla.confidence), how often is that true on FRESH questions? And how often does
 * it certify at all? `tuneThreshold()` searches the threshold and computes the Wilson bound on the same leave-one-out rows
 * (no holdout, no correction), so its bound can be optimistic. This harness splits the questions into A / B / C:
 *   A: calibrate a fresh planner (what the planner itself says is the "in-sample" certification),
 *   B: freeze the planner's policy (decide() on each B question, no search), realized quality of those decisions on B,
 *      one Wilson lower bound at `confidence` >= SLA  => "split-certified",
 *   C: untouched final test: mean quality and cost of the frozen policy; SLA miss = mean quality on C < SLA.
 * It does NOT change decide(), tuneThreshold() or the bound.
 */
import { createPlanner, enumeratePlans, planKey, type KnobSpace } from "../../src/index.js";
import { meanOf, mulberry32, type Table } from "./analyze-core.js";
import { normalQuantile } from "../../src/stats.js";
import { wilson } from "./single-plan-core.js";

export interface SplitArgs {
  table: Table;
  /** Feature vector per question (same length as the table's question dimension). */
  features: number[][];
  space: KnobSpace;
  A: number[];
  B: number[];
  C: number[];
  sla: number;
  confidence: number;
  /** Plan id of always-max (the safe fallback to compare against). */
  maxPlan: string;
  /** Known true mean quality per plan id (SYNTHETIC checks only): then miss = true quality of the policy below the SLA. */
  trueQuality?: Record<string, number>;
}

export interface SplitResult {
  inSampleCertified: boolean;
  policy: "routing" | "safe-plan";
  safePlan?: string;
  threshold: number;
  splitCertified: boolean;
  /** Wilson lower bound of the realized quality on B (the frozen policy, one bound, no search). */
  boundB: number;
  meanB: number;
  meanC: number;
  /** meanC < sla (real data), or the policy's TRUE quality < sla (synthetic). */
  missC: boolean;
  costC: number;
  /** Calibration cost on A at list price, per C question (so a policy must repay it). */
  calibCostPerC: number;
  calibRuns: number;
  /** Policy served always-max to everyone (safe fallback), so a certification is not a saving. */
  fellBackToMax: boolean;
  /** Best fixed plan chosen on A (cheapest with mean quality >= sla on A; if none, always-max), evaluated like the policy. */
  fixed: { plan: string; meanC: number; costC: number; missC: boolean; splitCertified: boolean; boundB: number };
  maxCostC: number;
  maxMeanC: number;
  sla: number;
}

export async function runSplit(a: SplitArgs): Promise<SplitResult> {
  const { table } = a;
  const plans = enumeratePlans(a.space);
  const ids = plans.map((p) => planKey(p));
  const idx = new Map(ids.map((id, i) => [id, i]));
  const pi = (id: string) => {
    const i = table.plans.indexOf(id);
    if (i < 0) throw new Error(`plan ${id} is not in the table`);
    return i;
  };
  const planner = createPlanner<number, number>({
    space: a.space,
    pipeline: async (q, plan) => {
      const p = pi(planKey(plan));
      return { output: table.f1[p][q], costUsd: table.cost[p][q] };
    },
    scorer: (_q, out) => out,
    sla: { minQuality: a.sla, confidence: a.confidence },
    featurize: (q) => a.features[q],
    feedbackRate: 0,
    concurrency: 1,
  });
  const report = await planner.calibrate(a.A);
  const realized = (qs: number[]) => qs.map((q) => table.f1[pi(planKey(planner.decide(q).plan))][q]);
  const realizedCost = (qs: number[]) => qs.map((q) => table.cost[pi(planKey(planner.decide(q).plan))][q]);
  const fB = realized(a.B);
  const fC = realized(a.C);
  const z = normalQuantile(a.confidence);
  const boundB = wilson(meanOf(fB), fB.length, z).lo;
  const decisions = a.C.map((q) => planKey(planner.decide(q).plan));
  const trueQ = a.trueQuality ? meanOf(decisions.map((d) => a.trueQuality![d])) : undefined;
  const meanC = meanOf(fC);
  const missC = trueQ !== undefined ? trueQ < a.sla : meanC < a.sla;

  // best fixed plan chosen on A only
  const meanOn = (p: number, qs: number[]) => meanOf(qs.map((q) => table.f1[p][q]));
  const costOn = (p: number, qs: number[]) => meanOf(qs.map((q) => table.cost[p][q]));
  let best = pi(a.maxPlan);
  for (let p = 0; p < table.plans.length; p++) {
    if (meanOn(p, a.A) < a.sla) continue;
    if (meanOn(best, a.A) < a.sla || costOn(p, a.A) < costOn(best, a.A) - 1e-12) best = p;
  }
  const fixedB = a.B.map((q) => table.f1[best][q]);
  const fixedMeanC = meanOn(best, a.C);
  const fixedTrue = a.trueQuality ? a.trueQuality[table.plans[best]] : undefined;
  const maxI = pi(a.maxPlan);
  void idx;
  return {
    inSampleCertified: report.slaCertified,
    policy: report.policy,
    safePlan: report.fixedPlan,
    threshold: report.qualityThreshold,
    splitCertified: boundB >= a.sla,
    boundB,
    meanB: meanOf(fB),
    meanC,
    missC,
    costC: meanOf(realizedCost(a.C)),
    calibCostPerC: report.costUsd / a.C.length,
    calibRuns: report.runs,
    fellBackToMax: report.policy === "safe-plan" && report.fixedPlan === a.maxPlan,
    fixed: {
      plan: table.plans[best],
      meanC: fixedMeanC,
      costC: costOn(best, a.C),
      missC: fixedTrue !== undefined ? fixedTrue < a.sla : fixedMeanC < a.sla,
      splitCertified: wilson(meanOf(fixedB), fixedB.length, z).lo >= a.sla,
      boundB: wilson(meanOf(fixedB), fixedB.length, z).lo,
    },
    maxCostC: costOn(maxI, a.C),
    maxMeanC: meanOn(maxI, a.C),
    sla: a.sla,
  };
}

/** A seeded split of 0..n-1 into A / B / C of the given sizes. */
export function split3(n: number, sizes: [number, number, number], seed: number): { A: number[]; B: number[]; C: number[] } {
  if (sizes[0] + sizes[1] + sizes[2] > n) throw new Error("split sizes exceed n");
  const rnd = mulberry32(seed);
  const o = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [o[i], o[j]] = [o[j], o[i]];
  }
  return { A: o.slice(0, sizes[0]), B: o.slice(sizes[0], sizes[0] + sizes[1]), C: o.slice(sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]) };
}

export interface Aggregate {
  n: number;
  inSampleCertRate: number;
  splitCertRate: number;
  /** Among in-sample certified runs: share whose C result missed (NaN when none certified). */
  missAmongInSample: number;
  missAmongSplit: number;
  nInSample: number;
  nSplit: number;
  /** Share of ALL splits that are certified and miss (false certification rate). */
  falseCertInSample: number;
  falseCertSplit: number;
  fallbackShare: number;
  certifiedSaving: number;
  certifiedOnlyFallback: number;
  meanF1C: number;
  costC: number;
  costWithCalibration: number;
  fixedMeanF1: number;
  fixedCost: number;
  maxMeanF1: number;
  maxCost: number;
  fixedMissRate: number;
  routingMissRate: number;
  /** Always-max's own SLA-miss rate on C: the noise floor of the 'mean on a small C below the SLA' definition. */
  maxMissRate: number;
}

export function aggregate(rs: SplitResult[]): Aggregate {
  const n = rs.length;
  const share = (xs: boolean[]) => xs.filter(Boolean).length / Math.max(1, xs.length);
  const ins = rs.filter((r) => r.inSampleCertified);
  const spl = rs.filter((r) => r.splitCertified);
  const cert = rs.filter((r) => r.inSampleCertified || r.splitCertified);
  return {
    n,
    inSampleCertRate: ins.length / n,
    splitCertRate: spl.length / n,
    missAmongInSample: ins.length ? share(ins.map((r) => r.missC)) : NaN,
    missAmongSplit: spl.length ? share(spl.map((r) => r.missC)) : NaN,
    nInSample: ins.length,
    nSplit: spl.length,
    falseCertInSample: share(rs.map((r) => r.inSampleCertified && r.missC)),
    falseCertSplit: share(rs.map((r) => r.splitCertified && r.missC)),
    fallbackShare: share(rs.map((r) => r.fellBackToMax)),
    certifiedSaving: cert.filter((r) => !r.fellBackToMax && r.costC < r.maxCostC * 0.99).length,
    certifiedOnlyFallback: cert.filter((r) => r.fellBackToMax).length,
    meanF1C: meanOf(rs.map((r) => r.meanC)),
    costC: meanOf(rs.map((r) => r.costC)),
    costWithCalibration: meanOf(rs.map((r) => r.costC + r.calibCostPerC)),
    fixedMeanF1: meanOf(rs.map((r) => r.fixed.meanC)),
    fixedCost: meanOf(rs.map((r) => r.fixed.costC)),
    maxMeanF1: meanOf(rs.map((r) => r.maxMeanC)),
    maxCost: meanOf(rs.map((r) => r.maxCostC)),
    fixedMissRate: share(rs.map((r) => r.fixed.missC)),
    routingMissRate: share(rs.map((r) => r.missC)),
    maxMissRate: share(rs.map((r) => r.maxMeanC < r.sla)),
  };
}
