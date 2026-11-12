/**
 * Real-embeddings featurizer (optional adapter, zero dependencies, plain `fetch`).
 *
 * The planner's `featurize` is synchronous and `decide()` takes ~0.07 ms, but embedders are async (network
 * or a local model). So embeddings are computed ahead of time in batches with `await featurizer.warm(inputs)`
 * and cached; the returned function is a normal synchronous `Featurizer` that reads the cache.
 *
 *   const featurize = embeddingFeaturizer({ embed: openAIEmbed({ apiKey, model: "text-embedding-3-small" }), text: (q) => q.question });
 *   await featurize.warm(calibrationSamples);
 *   const planner = createPlanner({ ..., featurize });
 *   await planner.calibrate(calibrationSamples);
 *   await featurize.warm([request]);            // before each new request (a cache hit costs nothing)
 *   await planner.run(request);
 *
 * API shapes checked against the official docs on 2026-11-13 (OpenAI: developers.openai.com/api/reference/resources/embeddings;
 * Voyage: docs.voyageai.com/reference/embeddings-api). Prices then: OpenAI text-embedding-3-small $0.02 / 1M tokens,
 * Voyage voyage-4-lite $0.02 / 1M tokens (200M free tokens). Local OpenAI-compatible servers (Ollama, LM Studio) cost $0.
 */
import { hashedTextFeaturizer } from "../features.js";

/** Embeds a batch of texts. Vectors must come back in input order. `tokens` is what the provider billed, if reported. */
export type EmbedFn = (texts: string[]) => Promise<{ vectors: number[][]; tokens?: number }>;

type FetchFn = (url: string, init: { method: string; headers: Record<string, string>; body: string }) => PromiseLike<{
  ok: boolean;
  status: number;
  text(): PromiseLike<string>;
  json(): PromiseLike<any>;
}>;

interface HttpOptions {
  apiKey?: string;
  /** Override `fetch` (tests, proxies). Default: global fetch. */
  fetch?: FetchFn;
}

async function postEmbeddings(url: string, opts: HttpOptions, body: object, who: string) {
  const f = (opts.fetch ?? (globalThis as any).fetch) as FetchFn | undefined;
  if (!f) throw new Error(`${who}: no fetch available (Node 22+ has one)`);
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (opts.apiKey) headers.authorization = `Bearer ${opts.apiKey}`;
  const res = await f(url, { method: "POST", headers, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${who}: HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const json = await res.json();
  const data = json?.data;
  if (!Array.isArray(data)) throw new Error(`${who}: response has no data[]`);
  const vectors: number[][] = [];
  for (const d of data) vectors[d.index] = d.embedding; // the API tags each vector with its input position
  return { vectors, tokens: (json.usage?.total_tokens ?? json.usage?.prompt_tokens) as number | undefined };
}

export interface OpenAIEmbedOptions extends HttpOptions {
  model: string;
  /** Default https://api.openai.com/v1. Point at a local OpenAI-compatible server (e.g. http://localhost:11434/v1) for $0. */
  baseUrl?: string;
  /** Only text-embedding-3 and later models. */
  dimensions?: number;
}

/** OpenAI-compatible `POST {baseUrl}/embeddings` (OpenAI, Ollama, LM Studio, vLLM...). Float vectors only. */
export function openAIEmbed(o: OpenAIEmbedOptions): EmbedFn {
  const url = `${(o.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "")}/embeddings`;
  return (texts) =>
    postEmbeddings(url, o, { model: o.model, input: texts, encoding_format: "float", ...(o.dimensions ? { dimensions: o.dimensions } : {}) }, "openAIEmbed");
}

export interface VoyageEmbedOptions extends HttpOptions {
  /** e.g. "voyage-4-lite" (cheapest general model on 2026-11-13). */
  model: string;
  /** "query" / "document" make Voyage prepend a retrieval prompt. Default null (plain similarity of questions). */
  inputType?: "query" | "document";
  outputDimension?: 256 | 512 | 1024 | 2048;
}

/** Voyage `POST https://api.voyageai.com/v1/embeddings`. */
export function voyageEmbed(o: VoyageEmbedOptions): EmbedFn {
  return (texts) =>
    postEmbeddings(
      "https://api.voyageai.com/v1/embeddings",
      o,
      { model: o.model, input: texts, ...(o.inputType ? { input_type: o.inputType } : {}), ...(o.outputDimension ? { output_dimension: o.outputDimension } : {}) },
      "voyageEmbed",
    );
}

export interface EmbeddingFeaturizerOptions<I> {
  embed: EmbedFn;
  /** Text to embed for an input. Default: the input itself if it is a string, else JSON. */
  text?: (input: I) => string;
  /** Texts per `embed` call. Default 128 (OpenAI allows 2048 inputs, Voyage 1000). */
  batchSize?: number;
  /** Expected vector length; a mismatch throws. Default: taken from the first vector. */
  dimension?: number;
  /** A text that was never warmed: "throw" (default, a clear error) or "hashed" (hashed bag-of-words, a different vector space!). */
  onMiss?: "throw" | "hashed";
}

export interface EmbeddingStats {
  /** `embed` calls made. */
  calls: number;
  /** Texts sent to the embedder (distinct, uncached). */
  texts: number;
  /** Tokens the provider reported (0 if it reports none). Multiply by the model price to get embedding cost. */
  tokens: number;
  /** Synchronous featurize() reads served from the cache. */
  cacheHits: number;
  /** Reads of a text that was never warmed. */
  misses: number;
}

export type EmbeddingFeaturizer<I> = ((input: I) => number[]) & {
  /** Embed every not-yet-cached input in batches. Call before calibrate() and before each new request. */
  warm(inputs: I[]): Promise<void>;
  readonly stats: EmbeddingStats;
};

export function embeddingFeaturizer<I = string>(options: EmbeddingFeaturizerOptions<I>): EmbeddingFeaturizer<I> {
  const batchSize = options.batchSize ?? 128;
  if (!(Number.isInteger(batchSize) && batchSize >= 1)) throw new Error("batchSize must be a positive integer");
  const textOf = options.text ?? ((x: I) => (typeof x === "string" ? x : JSON.stringify(x)));
  const cache = new Map<string, number[]>();
  const stats: EmbeddingStats = { calls: 0, texts: 0, tokens: 0, cacheHits: 0, misses: 0 };
  let dim = options.dimension;
  const hashed = options.onMiss === "hashed" ? hashedTextFeaturizer() : undefined;

  const unit = (v: number[]) => {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    return v.map((x) => x / n); // cosine() assumes unit vectors
  };

  const fn = ((input: I): number[] => {
    const t = textOf(input);
    const hit = cache.get(t);
    if (hit) {
      stats.cacheHits++;
      return hit;
    }
    stats.misses++;
    if (hashed) return hashed(t);
    throw new Error("embeddingFeaturizer: text was not embedded yet; call `await featurizer.warm([input])` first");
  }) as EmbeddingFeaturizer<I>;

  fn.warm = async (inputs: I[]) => {
    const todo = [...new Set(inputs.map(textOf))].filter((t) => !cache.has(t));
    for (let i = 0; i < todo.length; i += batchSize) {
      const batch = todo.slice(i, i + batchSize);
      const { vectors, tokens } = await options.embed(batch);
      stats.calls++;
      stats.texts += batch.length;
      stats.tokens += tokens ?? 0;
      for (let j = 0; j < batch.length; j++) {
        const v = vectors[j];
        if (!Array.isArray(v) || v.length === 0) throw new Error(`embeddingFeaturizer: no vector returned for input ${i + j}`);
        dim ??= v.length;
        if (v.length !== dim) throw new Error(`embeddingFeaturizer: expected ${dim} dimensions, got ${v.length}`);
        cache.set(batch[j], unit(v));
      }
    }
  };
  Object.defineProperty(fn, "stats", { get: () => stats });
  return fn;
}
