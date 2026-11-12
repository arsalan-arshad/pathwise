/**
 * LangChain.js adapter (targets `@langchain/core` 1.x; npm latest 1.4.0 on 2026-11-13).
 *
 * Zero runtime dependencies: this file never imports `@langchain/core`. Structural types only. You build the
 * chat model for each plan yourself (LangChain sets thinking/reasoning, max tokens and temperature on the model
 * constructor, and the option names differ per provider), so there is nothing to guess:
 *
 *   const pipeline = langchainPipeline({
 *     model: (plan) => new ChatOpenAI({ model: plan.model as string, reasoning: { effort: plan.thinking as string } }),
 *     price: (plan) => PRICES[plan.model as string],
 *     prepare: (q, plan) => ({ input: buildPrompt(q, plan.topK as number) }),
 *   });
 *   const planner = createPlanner({ space, pipeline, scorer, sla });
 */
import type { Pipeline, Plan } from "../types.js";
import type { ModelPrice } from "./vercel-ai.js";

/** `usage_metadata` of a LangChain AIMessage. Reasoning tokens are a detail of `output_tokens`. */
export interface LangChainUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  output_token_details?: { reasoning?: number };
}

/** What the adapter needs back from `invoke`: an AIMessage. */
export interface LangChainMessage {
  /** A string, or content blocks such as `{ type: "text", text }`. */
  content?: unknown;
  /** Present on current `@langchain/core` messages. */
  text?: string;
  usage_metadata?: LangChainUsage;
}

/** Structural chat model: anything with `invoke(input, config?)` (a Runnable). */
export interface LangChainChatModel {
  invoke(input: any, config?: { signal?: AbortSignal; [key: string]: any }): PromiseLike<LangChainMessage>;
}

export interface LangChainOptions<I> {
  /** The chat model for this plan (model name, thinking level, max tokens... set in its constructor). */
  model: (plan: Plan) => LangChainChatModel;
  /** The prompt for this request under this plan (a string, messages, ...), plus extra non-LLM cost (retrieval, reranker). */
  prepare: (input: I, plan: Plan) => { input: unknown; extraCostUsd?: number } | Promise<{ input: unknown; extraCostUsd?: number }>;
  /** USD per 1M tokens for the plan's model. Reasoning tokens are part of output_tokens, so they are billed as output. */
  price?: (plan: Plan) => ModelPrice;
  /** Instead of `price`: your own cost from the usage (use when a model reports no usage_metadata). */
  cost?: (usage: LangChainUsage | undefined, plan: Plan) => number;
  /** Passed to `invoke` as the second argument (callbacks, tags, signal...). */
  config?: Record<string, any>;
}

/** The answer text of a message: `.text`, a string `.content`, or the text blocks of a block array. */
export function messageText(m: LangChainMessage): string {
  if (typeof m.text === "string") return m.text;
  if (typeof m.content === "string") return m.content;
  if (Array.isArray(m.content)) {
    return m.content
      .map((b) => (typeof b === "string" ? b : b && typeof b === "object" && (b as { type?: string }).type === "text" ? String((b as { text?: unknown }).text ?? "") : ""))
      .join("");
  }
  return "";
}

/** Dollar cost from `usage_metadata`. `output_tokens` already includes reasoning tokens. */
export function langchainCostUsd(usage: LangChainUsage, price: ModelPrice): number {
  return ((usage.input_tokens ?? 0) * price.inputPerMTok + (usage.output_tokens ?? 0) * price.outputPerMTok) / 1e6;
}

/** Turn a LangChain chat model per plan into a Pathwise pipeline; cost comes from the real usage_metadata. */
export function langchainPipeline<I>(options: LangChainOptions<I>): Pipeline<I, string> {
  if (!options.price && !options.cost) throw new Error("pathwise/langchain: pass `price` or `cost`");
  return async (input, plan) => {
    const { input: prompt, extraCostUsd = 0 } = await options.prepare(input, plan);
    const message = await options.model(plan).invoke(prompt, options.config);
    const usage = message.usage_metadata;
    let costUsd: number;
    if (options.cost) costUsd = options.cost(usage, plan);
    else if (!usage || usage.input_tokens === undefined || usage.output_tokens === undefined) {
      throw new Error("pathwise/langchain: the model returned no usage_metadata; pass `cost` instead of `price` (costs must not be guessed)");
    } else costUsd = langchainCostUsd(usage, options.price!(plan));
    return { output: messageText(message), costUsd: costUsd + extraCostUsd };
  };
}
