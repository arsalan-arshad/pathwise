/**
 * Plan recommendation by racing (successive elimination). Pure logic: the caller injects `evaluate`. Zero dependencies.
 * Prior art: best-arm identification and configuration search for LLM/RAG pipelines are published work; nothing here is a
 * novel method.
 */
import { planKey } from "./space.js";
import { normalQuantile, wilsonLower, wilsonUpper } from "./stats.js";
import type { Plan, Recommendation } from "./types.js";

export interface RecommendOptions {
  /** Probability for the Wilson lower bound behind `certified` (default sla.confidence, else 0.9). */
  confidence?: number;
  /** Confidence of the bounds used only to ELIMINATE plans early (default 0.975, stricter than `confidence`: a wrong early drop cannot be undone). */
  eliminationConfidence?: number;
  /** Questions every plan is measured on before any plan can be eliminated. Default 10. */
  minSamples?: number;
  /** Seed for the order in which questions are drawn. Default 7. */
  seed?: number;
}

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

export interface RecommendArgs<I> {
  plans: Plan[];
  samples: I[];
  minQuality: number;
  confidence: number;
  options: RecommendOptions;
  concurrency: number;
  evaluate: (input: I, plan: Plan) => Promise<{ quality: number; costUsd: number }>;
}

export async function recommendPlan<I>(a: RecommendArgs<I>): Promise<Recommendation | undefined> {
  const { plans, samples, minQuality, confidence } = a;
  if (samples.length === 0) throw new Error("recommend() needs at least one sample");
  if (!(confidence > 0 && confidence < 1)) throw new Error("confidence must be in (0, 1)");
  const minSamples = Math.max(1, a.options.minSamples ?? 10);
  const z = normalQuantile(confidence);
  const elim = a.options.eliminationConfidence ?? 0.975;
  if (!(elim > 0 && elim < 1)) throw new Error("eliminationConfidence must be in (0, 1)");
  const zElim = normalQuantile(elim);
  const keys = plans.map((p) => planKey(p));
  const order = samples.map((_, i) => i);
  const rnd = rng(a.options.seed ?? 7);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  let alive = plans.map((_, i) => i);
  const sumQ = plans.map(() => 0);
  const sumC = plans.map(() => 0);
  let calls = 0;
  let n = 0;
  for (const qi of order) {
    const todo = alive.map((p) => async () => {
      const r = await a.evaluate(samples[qi], plans[p]);
      sumQ[p] += r.quality;
      sumC[p] += r.costUsd;
    });
    let next = 0;
    await Promise.all(
      Array.from({ length: Math.min(Math.max(1, a.concurrency), todo.length) }, async () => {
        while (next < todo.length) await todo[next++]();
      }),
    );
    calls += alive.length;
    n++;
    if (n < minSamples) continue;
    const b = alive.map((p) => ({ p, m: sumQ[p] / n, c: sumC[p] / n, lo: wilsonLower(sumQ[p] / n, n, zElim), hi: wilsonUpper(sumQ[p] / n, n, zElim) }));
    const kept = b.filter((x) => {
      if (x.hi < minQuality) return false; // cannot meet the SLA
      return !b.some((y) => y !== x && y.lo >= minQuality && (y.c < x.c - 1e-12 || (Math.abs(y.c - x.c) <= 1e-12 && y.m > x.m))); // a cheaper plan already clears it
    });
    if (kept.length === 0) {
      alive = [];
      break;
    }
    alive = kept.map((x) => x.p);
    if (alive.length === 1) {
      // One plan is left: keep measuring only it, until it is certified or the samples run out (more evidence can still certify it).
      const x = b.find((y) => y.p === alive[0])!;
      if (wilsonLower(sumQ[alive[0]] / n, n, z) >= minQuality || x.hi < minQuality) break;
    }
  }
  const warnings: string[] = [];
  if (n === 0) return undefined;
  let best = -1;
  for (const p of alive) {
    if (sumQ[p] / n < minQuality) continue;
    if (best < 0 || sumC[p] / n < sumC[best] / n - 1e-12) best = p;
  }
  if (best < 0) return undefined;
  const mean = sumQ[best] / n;
  const lower = wilsonLower(mean, n, z);
  const certified = lower >= minQuality;
  const fullTableCalls = plans.length * n;
  if (!certified) warnings.push(`not certified: the Wilson lower bound ${lower.toFixed(3)} at confidence ${confidence} is below minQuality ${minQuality} with ${n} samples; this plan only meets the SLA on the observed mean.`);
  if (n < 30) warnings.push(`only ${n} samples: the recommendation is a rough estimate.`);
  warnings.push("elimination looks at the bounds after every sample, so the confidence applies to the final bound, not to the whole search.");
  const certifiedOthers = alive.filter((p) => p !== best && wilsonLower(sumQ[p] / n, n, z) >= minQuality);
  if (!certified && certifiedOthers.length > 0) warnings.push(`a dearer surviving plan (${certifiedOthers.map((p) => keys[p]).join(", ")}) is certified; the recommended plan is the cheapest whose observed mean meets the SLA.`);
  const survivors = alive.length;
  return {
    plan: plans[best],
    planKey: keys[best],
    meanQuality: mean,
    lowerBound: lower,
    certified,
    confidence,
    costPerRunUsd: sumC[best] / n,
    callsUsed: calls,
    fullTableCalls,
    samplesUsed: n,
    basis: calls === fullTableCalls && survivors === plans.length ? "full-table" : "racing",
    warnings,
  };
}
