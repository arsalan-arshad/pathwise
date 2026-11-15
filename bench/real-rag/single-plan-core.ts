/**
 * Pure helpers for single-plan.ts: racing (successive elimination) to find the cheapest plan whose mean F1 meets an SLA.
 * No I/O. NOT a novelty claim: configuration search and best-arm identification for LLM/RAG pipelines are published work.
 */
import { mulberry32, type Table } from "./analyze-core.js";

/** Wilson interval bounds for a mean in [0,1] treated like a proportion (as the planner's own certification does). */
export function wilson(m: number, n: number, z: number): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 1 };
  const z2 = z * z;
  const centre = m + z2 / (2 * n);
  const margin = z * Math.sqrt((m * (1 - m)) / n + z2 / (4 * n * n));
  const d = 1 + z2 / n;
  return { lo: Math.max(0, (centre - margin) / d), hi: Math.min(1, (centre + margin) / d) };
}

export interface RaceResult {
  /** The chosen plan (cheapest feasible among the survivors), or null when no surviving plan meets the SLA. */
  plan: string | null;
  /** (question, plan) pairs evaluated. */
  calls: number;
  questionsUsed: number;
  survivors: number;
}

/**
 * Draws questions in `order`, evaluates every surviving plan on each, and after `minQuestions` eliminates a plan when
 *  - its Wilson upper bound is below the SLA (it cannot meet it), or
 *  - another plan's Wilson lower bound already meets the SLA at a lower cost (it is dominated by a feasible plan).
 * Stops when one plan is left or the questions run out; then picks the cheapest survivor whose observed mean meets the SLA.
 * The bounds are looked at after every question, so the stated per-look error is not a family-wise guarantee.
 */
export function race(table: Table, order: number[], sla: number, z: number, minQuestions: number): RaceResult {
  const P = table.plans.length;
  let alive = Array.from({ length: P }, (_, i) => i);
  const sumF = new Array<number>(P).fill(0);
  const sumC = new Array<number>(P).fill(0);
  let calls = 0;
  let n = 0;
  for (const q of order) {
    for (const p of alive) {
      sumF[p] += table.f1[p][q];
      sumC[p] += table.cost[p][q];
    }
    calls += alive.length;
    n++;
    if (n < minQuestions) continue;
    const b = alive.map((p) => ({ p, m: sumF[p] / n, c: sumC[p] / n, ...wilson(sumF[p] / n, n, z) }));
    const next = b.filter((x) => {
      if (x.hi < sla) return false;
      return !b.some((y) => y !== x && y.lo >= sla && (y.c < x.c - 1e-12 || (Math.abs(y.c - x.c) <= 1e-12 && y.m > x.m)));
    });
    if (next.length === 0) {
      return { plan: null, calls, questionsUsed: n, survivors: 0 };
    }
    alive = next.map((x) => x.p);
    if (alive.length === 1) break;
  }
  let best: number | null = null;
  for (const p of alive) {
    if (sumF[p] / n < sla) continue;
    if (best === null || sumC[p] / n < sumC[best] / n - 1e-12) best = p;
  }
  return { plan: best === null ? null : table.plans[best], calls, questionsUsed: n, survivors: alive.length };
}

/** A seeded random permutation of 0..n-1. */
export function shuffled(n: number, seed: number): number[] {
  const rnd = mulberry32(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function percentile(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))))];
}
