/**
 * Certification by measurement: run the policy that is actually served on FRESH questions and put ONE Wilson lower bound on its
 * mean quality. No threshold search, no replay, no credit for unmeasured plans. Pure logic: the caller injects `decide` and
 * `evaluate`. Zero dependencies. Prior art: a held-out confidence bound on a fixed policy is ordinary practice, not a new method.
 */
import { normalQuantile, wilsonLower } from "./stats.js";
import type { Certification, Plan } from "./types.js";

export interface CertifyOptions {
  /** Probability of the Wilson lower bound (default sla.confidence, else 0.9). */
  confidence?: number;
}

export interface CertifyArgs<I> {
  samples: I[];
  minQuality: number;
  confidence: number;
  concurrency: number;
  /** The frozen policy, exactly as production would call it (no learning). `fallback` = served by the safe or best-effort fallback plan. */
  decide: (input: I) => { plan: Plan; planKey: string; fallback: boolean };
  evaluate: (input: I, plan: Plan) => Promise<{ quality: number; costUsd: number; clamped?: boolean }>;
  /** sla.scorerErrorRate: when above 0 a warning says the bound is on the raw judge scores (not corrected). */
  scorerErrorRate: number;
}

export async function certifyPolicy<I>(a: CertifyArgs<I>): Promise<Certification> {
  const { samples, minQuality, confidence } = a;
  if (!Array.isArray(samples)) throw new Error("certify() needs an array of fresh samples");
  if (samples.length === 0) throw new Error("certify() needs at least one fresh sample");
  if (!(minQuality >= 0 && minQuality <= 1)) throw new Error("certify(): sla.minQuality must be a number between 0 and 1");
  if (!(confidence > 0 && confidence < 1)) throw new Error("confidence must be in (0, 1)");
  // decide() first, for every input, sequentially: it is synchronous and cheap, and it keeps the result independent of evaluation order.
  const chosen = samples.map((s) => a.decide(s));
  const results: Array<{ quality: number; costUsd: number; clamped?: boolean }> = new Array(samples.length);
  let next = 0;
  let aborted = false; // after the first error no worker starts another run (no spend after the rejection)
  await Promise.all(
    Array.from({ length: Math.min(Math.max(1, a.concurrency), samples.length) }, async () => {
      while (!aborted && next < samples.length) {
        const i = next++;
        try {
          results[i] = await a.evaluate(samples[i], chosen[i].plan);
        } catch (err) {
          aborted = true;
          throw err;
        }
      }
    }),
  );
  // Reduce in input order, so the floating-point sums and the key order of planShares do not depend on timing.
  let sum = 0;
  let cost = 0;
  let fallbacks = 0;
  let clampedCount = 0;
  const counts: Record<string, number> = {};
  for (let i = 0; i < samples.length; i++) {
    sum += results[i].quality;
    cost += results[i].costUsd;
    if (chosen[i].fallback) fallbacks++;
    if (results[i].clamped) clampedCount++;
    counts[chosen[i].planKey] = (counts[chosen[i].planKey] ?? 0) + 1;
  }
  const n = samples.length;
  const planShares: Record<string, number> = {};
  for (const k of Object.keys(counts).sort()) planShares[k] = counts[k] / n;
  const meanQuality = sum / n;
  const lowerBound = wilsonLower(meanQuality, n, normalQuantile(confidence));
  const certified = lowerBound >= minQuality;
  const warnings: string[] = [];
  if (n < 60) warnings.push(`only ${n} fresh samples: a plan a few points above the SLA usually cannot certify with this few (about 60 are needed even when the observed mean is 5 points above 0.9 at confidence 0.9; see bench/real-rag/results/certify-by-measurement-design-2026-11-17.md).`);
  if (!certified) warnings.push(`not certified: the Wilson lower bound ${lowerBound.toFixed(3)} at confidence ${confidence} is below minQuality ${minQuality} (observed mean ${meanQuality.toFixed(3)} over ${n} samples).`);
  warnings.push("use these fresh questions ONCE per frozen policy: if you change anything after seeing the result and certify on the same questions again, that is a replay, not a certificate; reserve a new set.");
  warnings.push("the fresh questions must be independent of the calibration questions and drawn like the traffic you will serve; drift is not covered.");
  if (clampedCount > 0) warnings.push(`${clampedCount} of ${n} scores were outside [0, 1] and were clamped: check the scorer (a scorer that returns percentages would otherwise look perfect).`);
  if (a.scorerErrorRate > 0) warnings.push("scorerErrorRate is set but certify() does not correct for it: the bound is on the raw judge scores.");
  return { meanQuality, lowerBound, certified, confidence, n, costUsd: cost, planShares, fellBackToSafe: fallbacks / n, warnings };
}
