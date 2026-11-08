/**
 * Simulated customer-support RAG workload shared by the demo and the benchmarks.
 *
 * Knobs: which model, how many chunks to retrieve, reranker on/off.
 * Easy questions are answered correctly by a small model with little context;
 * hard multi-hop questions need the big model, more context and reranking.
 */
import type { Plan } from "../src/index.js";

export interface Query {
  text: string;
  difficulty: number; // hidden ground truth, the planner never sees it
}

export interface WorkloadOptions {
  /**
   * Extra random difficulty (+/- jitter) on top of the tier range, so the question text
   * predicts difficulty less well. 0 = the clean demo workload.
   */
  jitter?: number;
}

/** Deterministic randomness so results are reproducible. */
export function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const products = ["starter", "team", "business", "enterprise"];
const countries = ["germany", "brazil", "japan", "canada", "pakistan"];

export function createWorkload(seed: number, options: WorkloadOptions = {}) {
  const rand = rng(seed);
  const pick = <T>(xs: readonly T[]) => xs[Math.floor(rand() * xs.length)];
  const between = (a: number, b: number) => a + rand() * (b - a);
  const jitter = options.jitter ?? 0;

  const tiers = {
    easy: {
      difficulty: [0.15, 0.3],
      templates: [
        () => "what are your opening hours",
        () => "how do i reset my password",
        () => `where is my order ${Math.floor(rand() * 90000 + 10000)}`,
        () => `what is the price of the ${pick(products)} plan`,
        () => `do you ship to ${pick(countries)}`,
      ],
    },
    medium: {
      difficulty: [0.42, 0.6],
      templates: [
        () => `how do i migrate my ${pick(products)} workspace to a new region without downtime`,
        () => `why is my invoice higher this month after upgrading to ${pick(products)}`,
        () => `can i combine the ${pick(products)} plan with an annual discount for a team of ${Math.floor(between(5, 200))}`,
      ],
    },
    hard: {
      difficulty: [0.7, 0.9],
      templates: [
        () =>
          `compare the refund policy for enterprise contracts signed before and after the 2025 terms update and explain which applies to a ${pick(countries)} customer who upgraded mid cycle`,
        () =>
          `given our ${pick(products)} usage logs explain why latency spikes when sso and audit logging are both enabled across multiple regions and what the contract says about credits`,
      ],
    },
  } as const;

  const mix: Array<keyof typeof tiers> = ["easy", "easy", "easy", "medium", "medium", "hard"];

  function makeQueries(n: number): Query[] {
    return Array.from({ length: n }, () => {
      const tier = tiers[pick(mix)];
      const text = pick(tier.templates)();
      let difficulty = between(tier.difficulty[0], tier.difficulty[1]);
      if (jitter > 0) difficulty = Math.min(1, Math.max(0, difficulty + between(-jitter, jitter)));
      return { text, difficulty };
    });
  }

  return { makeQueries, rand };
}

// ---------- the simulated pipeline ----------
export const MODEL = {
  small: { capability: 0.35, callUsd: 0.0003, perChunkUsd: 0.00002, latencyMs: 400 },
  medium: { capability: 0.6, callUsd: 0.003, perChunkUsd: 0.0002, latencyMs: 900 },
  large: { capability: 0.85, callUsd: 0.02, perChunkUsd: 0.0012, latencyMs: 2200 },
} as const;
const TOPK_BONUS: Record<number, number> = { 2: 0, 5: 0.05, 10: 0.08 };

export async function ragPipeline(q: Query, plan: Plan) {
  const m = MODEL[plan.model as keyof typeof MODEL];
  const topK = plan.topK as number;
  const rerank = plan.rerank as boolean;
  const skill = m.capability + TOPK_BONUS[topK] + (rerank ? 0.05 : 0);
  return {
    output: { correct: skill >= q.difficulty },
    costUsd: m.callUsd + topK * m.perChunkUsd + (rerank ? 0.001 : 0),
    latencyMs: m.latencyMs + topK * 30 + (rerank ? 250 : 0),
  };
}

export const SPACE = { model: ["small", "medium", "large"], topK: [2, 5, 10], rerank: [false, true] } as const;
