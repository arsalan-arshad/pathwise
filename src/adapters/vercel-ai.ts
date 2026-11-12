/**
 * Vercel AI SDK adapter (targets `ai` 7.x, checked against the npm registry on Nov 13, 2026).
 *
 * Zero runtime dependencies: this file never imports `ai`. You pass in the `generateText` function
 * (and your models) and the adapter turns it into a Pathwise `Pipeline`. Types are structural, so any
 * `ai` major that returns `{ text, usage }` works.
 *
 *   import { generateText } from "ai";
 *   const pipeline = vercelAIPipeline({ generateText, models, prepare });
 *   const planner = createPlanner({ space, pipeline, scorer, sla });
 */
import type { Pipeline, Plan, KnobValue } from "../types.js";

/** Token usage as returned by `generateText`. Accepts the v7 nested shape and flat/older fields. */
export interface AIUsage {
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  outputTokenDetails?: { reasoningTokens?: number };
  promptTokens?: number;
  completionTokens?: number;
}

/** What the adapter needs back from `generateText`. */
export interface AIResult {
  text: string;
  usage?: AIUsage;
}

/** Structural type of `generateText`: `model` stays opaque, the rest is passed through untouched. */
export type GenerateTextFn = (options: {
  model: any;
  prompt?: string;
  system?: string;
  messages?: any[];
  providerOptions?: Record<string, any>;
  maxOutputTokens?: number;
  temperature?: number;
  abortSignal?: AbortSignal;
}) => PromiseLike<AIResult>;

/** USD per 1M tokens. Reasoning tokens are billed as output tokens. */
export interface ModelPrice {
  inputPerMTok: number;
  outputPerMTok: number;
}

export interface ModelEntry {
  /** Anything `generateText` accepts as `model` (a provider model object or a gateway string). */
  model: unknown;
  price: ModelPrice;
}

/** What `prepare` returns for one request under one plan: the prompt plus optional extra costs. */
export interface Prepared {
  prompt?: string;
  system?: string;
  messages?: any[];
  /** Cost of non-LLM steps you did under this plan (retrieval, reranker), in USD. */
  extraCostUsd?: number;
}

export interface VercelAIOptions<I> {
  generateText: GenerateTextFn;
  /** Models by the names used in the `modelKnob` knob values. */
  models: Record<string, ModelEntry>;
  /** Build the prompt for this request under this plan (e.g. retrieve `plan.topK` chunks). */
  prepare: (input: I, plan: Plan) => Prepared | Promise<Prepared>;
  /** Name of the knob holding the model name. Default "model". */
  modelKnob?: string;
  /** Extra generateText options per plan, e.g. `providerOptions` for a thinking level. */
  settings?: (plan: Plan) => {
    providerOptions?: Record<string, any>;
    maxOutputTokens?: number;
    temperature?: number;
  };
  abortSignal?: AbortSignal;
}

/** Dollar cost of one call from its usage. Missing token counts count as 0. */
export function usageCostUsd(usage: AIUsage | undefined, price: ModelPrice): number {
  const input = usage?.inputTokens ?? usage?.promptTokens ?? 0;
  const output = usage?.outputTokens ?? usage?.completionTokens ?? 0;
  const reasoning = usage?.outputTokenDetails?.reasoningTokens ?? usage?.reasoningTokens ?? 0;
  // v7: outputTokens already includes reasoning tokens (textTokens + reasoningTokens).
  // Flat legacy shapes report them separately, so only add them when outputTokenDetails is absent.
  const billedOutput = usage?.outputTokenDetails ? output : output + reasoning;
  return (input * price.inputPerMTok + billedOutput * price.outputPerMTok) / 1e6;
}

/** Turn `generateText` into a Pathwise pipeline: plan -> model + settings, cost from real usage. */
export function vercelAIPipeline<I>(options: VercelAIOptions<I>): Pipeline<I, string> {
  const knob = options.modelKnob ?? "model";
  return async (input, plan) => {
    const name = plan[knob] as KnobValue | undefined;
    const entry = typeof name === "string" ? options.models[name] : undefined;
    if (!entry) throw new Error(`pathwise/vercel-ai: plan has no model "${String(name)}" in options.models (knob "${knob}")`);
    const prepared = await options.prepare(input, plan);
    const { extraCostUsd = 0, ...call } = prepared;
    const result = await options.generateText({
      ...call,
      ...options.settings?.(plan),
      model: entry.model,
      abortSignal: options.abortSignal,
    });
    return { output: result.text, costUsd: usageCostUsd(result.usage, entry.price) + extraCostUsd };
  };
}
