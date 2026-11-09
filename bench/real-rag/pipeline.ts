/** The RAG pipeline Pathwise plans over: BM25 top-k passages -> Gemini model at a thinking level. */
import type { Plan } from "../../src/index.js";
import type { Tier } from "./config.js";
import { retrieve, type Question } from "./dataset.js";
import { generate } from "./gemini.js";
import { f1 } from "./score.js";

export interface Answer {
  answer: string;
}

export function buildPrompt(q: Question, topK: number): { prompt: string; goldRetrieved: number } {
  const passages = retrieve(q, topK);
  const goldRetrieved = passages.filter((p) => q.goldTitles.includes(p.title)).length;
  const prompt =
    "Answer the question using the passages below. Reply with only the answer: a short phrase, a name, " +
    "a number, or yes/no. No explanation and no full sentence.\n\n" +
    passages.map((p, i) => `[${i + 1}] ${p.title}: ${p.text}`).join("\n\n") +
    `\n\nQuestion: ${q.question}\nAnswer:`;
  return { prompt, goldRetrieved };
}

export async function ragPipeline(q: Question, plan: Plan) {
  const { prompt, goldRetrieved } = buildPrompt(q, plan.topK as number);
  const r = await generate(plan.model as Tier, plan.thinking as "low" | "high", prompt, {
    gold: q.answer,
    goldRetrieved,
  });
  return { output: { answer: r.text } satisfies Answer, costUsd: r.costUsd, latencyMs: r.latencyMs };
}

export const scorer = (q: Question, out: Answer) => f1(out.answer, q.answer);

/** Runs tasks with a fixed number in flight. */
export async function pool<T, R>(items: T[], limit: number, fn: (item: T, i: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  let failed = false; // after the first error, workers stop taking new items
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (!failed && next < items.length) {
        const i = next++;
        try {
          out[i] = await fn(items[i], i);
        } catch (err) {
          failed = true;
          throw err;
        }
      }
    }),
  );
  return out;
}

export function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}
