/**
 * OpenAI-compatible Chat Completions adapter: plain global `fetch`, zero dependencies, no SDK.
 * Works with anything that speaks `POST {baseUrl}/chat/completions`: OpenAI, the Gemini OpenAI-compatible endpoint,
 * OpenRouter, Ollama, LM Studio, vLLM ...
 *
 * Base URLs, from the official docs read on 2026-11-13:
 *   OpenAI      https://api.openai.com/v1
 *   Gemini      https://generativelanguage.googleapis.com/v1beta/openai   (Authorization: Bearer <Gemini API key>)
 *   Ollama      http://localhost:11434/v1   (local: any key is ignored, so apiKey can be omitted)
 *   OpenRouter  https://openrouter.ai/api/v1   (inferred from the docs' code samples)
 *
 *   const pipeline = openAICompatiblePipeline({
 *     baseUrl: "http://localhost:11434/v1",
 *     models: { small: { model: "qwen3:4b", price: { inputPerMTok: 0, outputPerMTok: 0 } } },
 *     prepare: (q, plan) => ({ messages: [{ role: "user", content: buildPrompt(q, plan.topK as number) }] }),
 *     settings: (plan) => ({ reasoning_effort: plan.thinking as string }),
 *   });
 *
 * The API key is only ever taken from `apiKey` (never from the environment), is sent only in the Authorization
 * header, and is scrubbed from error messages. Costs come from the response's real `usage`; a response without
 * usage fails loudly unless you pass a `cost` callback. Costs are never guessed.
 */
import type { Pipeline, Plan, KnobValue } from "../types.js";
import type { ModelPrice } from "./vercel-ai.js";

/** `usage` of a chat completion response. */
export interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  completion_tokens_details?: { reasoning_tokens?: number };
}

type FetchFn = (url: string, init: { method: string; headers: Record<string, string>; body: string; signal?: AbortSignal }) => PromiseLike<{
  ok: boolean;
  status: number;
  headers: { get(name: string): string | null };
  text(): PromiseLike<string>;
}>;

export interface ChatMessage {
  role: string;
  content: unknown;
  [key: string]: unknown;
}

export interface OpenAICompatibleOptions<I> {
  /** Without a trailing `/chat/completions`, e.g. "http://localhost:11434/v1". */
  baseUrl: string;
  /** Sent as `Authorization: Bearer`. Never read from the environment, never logged. Omit for local servers. */
  apiKey?: string;
  /** Model names used in the plan's model knob -> the provider's model id and its price (USD per 1M tokens). */
  models: Record<string, { model: string; price: ModelPrice }>;
  /** Messages for this request under this plan (e.g. retrieve `plan.topK` chunks), plus extra non-LLM cost. */
  prepare: (input: I, plan: Plan) => { messages: ChatMessage[]; extraCostUsd?: number } | Promise<{ messages: ChatMessage[]; extraCostUsd?: number }>;
  /** Extra request-body fields per plan: `reasoning_effort`, `max_completion_tokens`, `temperature`... passed through untouched. */
  settings?: (plan: Plan) => Record<string, unknown>;
  /** Name of the knob holding the model name. Default "model". */
  modelKnob?: string;
  /** Instead of the price table: your own cost from the usage (also lets models without usage work). */
  cost?: (usage: ChatUsage | undefined, plan: Plan) => number;
  /**
   * Whether `completion_tokens` already includes reasoning tokens. Default true: OpenAI's docs say rejected-prediction
   * tokens "like reasoning tokens" count in the completion total, which implies it, but providers are not all
   * confirmed. Set false if yours reports `completion_tokens_details.reasoning_tokens` on top of `completion_tokens`.
   */
  reasoningInCompletionTokens?: boolean;
  /** Extra headers (e.g. OpenRouter's HTTP-Referer). Do not put secrets you do not want sent on every call. */
  headers?: Record<string, string>;
  /** Retries on 429, 5xx and network errors or timeouts. Default 2. Not on 4xx, bad JSON or abort. */
  retries?: number;
  /** First backoff in ms, doubled each retry (a numeric Retry-After header up to 30 s wins). Default 500. */
  retryBaseMs?: number;
  /** Per-attempt timeout in ms. Default 60000. */
  timeoutMs?: number;
  /** Aborts the call (and any retry wait). */
  signal?: AbortSignal;
  /** Override `fetch` (tests, proxies). Default: global fetch. */
  fetch?: FetchFn;
}

/** Dollar cost from `usage`. Missing token counts are an error in the pipeline, 0 here. */
export function chatCostUsd(usage: ChatUsage, price: ModelPrice, reasoningInCompletionTokens = true): number {
  const out = (usage.completion_tokens ?? 0) + (reasoningInCompletionTokens ? 0 : (usage.completion_tokens_details?.reasoning_tokens ?? 0));
  return ((usage.prompt_tokens ?? 0) * price.inputPerMTok + out * price.outputPerMTok) / 1e6;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(abortError());
    const t = setTimeout(() => (signal?.removeEventListener("abort", onAbort), resolve()), ms);
    const onAbort = () => (clearTimeout(t), reject(abortError()));
    signal?.addEventListener("abort", onAbort, { once: true });
  });
const abortError = () => Object.assign(new Error("pathwise/openai-compatible: aborted"), { name: "AbortError" });

class Retryable extends Error {
  constructor(msg: string, readonly retryAfterMs?: number) {
    super(msg);
  }
}

/** Turn an OpenAI-compatible endpoint into a Pathwise pipeline: plan -> request, cost from real usage. */
export function openAICompatiblePipeline<I>(o: OpenAICompatibleOptions<I>): Pipeline<I, string> {
  const knob = o.modelKnob ?? "model";
  const retries = o.retries ?? 2;
  const retryBaseMs = o.retryBaseMs ?? 500;
  const timeoutMs = o.timeoutMs ?? 60_000;
  const url = `${o.baseUrl.replace(/\/+$/, "")}/chat/completions`;
  const scrub = (s: string) => (o.apiKey ? s.split(o.apiKey).join("[redacted]") : s).slice(0, 300);

  async function attempt(body: string): Promise<{ text: string; usage?: ChatUsage }> {
    const f = (o.fetch ?? (globalThis as any).fetch) as FetchFn | undefined;
    if (!f) throw new Error("pathwise/openai-compatible: no fetch available (Node 22+ has one)");
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    const onAbort = () => ctl.abort();
    o.signal?.addEventListener("abort", onAbort, { once: true });
    try {
      const headers: Record<string, string> = { "content-type": "application/json", ...o.headers };
      if (o.apiKey) headers.authorization = `Bearer ${o.apiKey}`;
      let res;
      try {
        res = await f(url, { method: "POST", headers, body, signal: ctl.signal });
      } catch (err) {
        if (o.signal?.aborted) throw abortError();
        throw new Retryable(ctl.signal.aborted ? `timed out after ${timeoutMs} ms` : `network error: ${scrub(String((err as Error)?.message ?? err))}`);
      }
      const text = await res.text();
      if (res.status === 429 || res.status >= 500) {
        const ra = Number(res.headers.get("retry-after"));
        throw new Retryable(`HTTP ${res.status}: ${scrub(text)}`, Number.isFinite(ra) && ra > 0 ? Math.min(ra * 1000, 30_000) : undefined);
      }
      if (!res.ok) throw new Error(`pathwise/openai-compatible: HTTP ${res.status}: ${scrub(text)}`);
      let json: any;
      try {
        json = JSON.parse(text);
      } catch {
        throw new Error(`pathwise/openai-compatible: response is not JSON: ${scrub(text)}`);
      }
      const content = json?.choices?.[0]?.message?.content;
      if (typeof content !== "string") throw new Error("pathwise/openai-compatible: response has no choices[0].message.content string");
      return { text: content, usage: json.usage };
    } finally {
      clearTimeout(timer);
      o.signal?.removeEventListener("abort", onAbort);
    }
  }

  return async (input, plan) => {
    const name = plan[knob] as KnobValue | undefined;
    const entry = typeof name === "string" ? o.models[name] : undefined;
    if (!entry) throw new Error(`pathwise/openai-compatible: plan has no model "${String(name)}" in options.models (knob "${knob}")`);
    const { messages, extraCostUsd = 0 } = await o.prepare(input, plan);
    const body = JSON.stringify({ ...o.settings?.(plan), model: entry.model, messages });

    let result: { text: string; usage?: ChatUsage } | undefined;
    for (let n = 0; ; n++) {
      try {
        result = await attempt(body);
        break;
      } catch (err) {
        if (!(err instanceof Retryable)) throw err;
        if (n >= retries) throw new Error(`pathwise/openai-compatible: ${err.message} (after ${n + 1} attempts)`);
        await sleep(err.retryAfterMs ?? retryBaseMs * 2 ** n, o.signal);
      }
    }
    const { usage } = result;
    let costUsd: number;
    if (o.cost) costUsd = o.cost(usage, plan);
    else if (!usage || usage.prompt_tokens === undefined || usage.completion_tokens === undefined) {
      throw new Error("pathwise/openai-compatible: the response has no usage; pass `cost` (costs must not be guessed)");
    } else costUsd = chatCostUsd(usage, entry.price, o.reasoningInCompletionTokens ?? true);
    if (!Number.isFinite(costUsd + extraCostUsd) || extraCostUsd < 0) throw new Error("pathwise/openai-compatible: cost must be finite and non-negative");
    return { output: result.text, costUsd: costUsd + extraCostUsd };
  };
}
