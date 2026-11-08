/**
 * Real-RAG benchmark configuration: models, prices, knobs, budget.
 * Prices: official Gemini API paid tier, checked Nov 8, 2026 (https://aistudio.google.com/docs/pricing),
 * USD per 1M tokens, thinking tokens billed at the output price.
 *
 * Two model sets (PATHWISE_BENCH_MODELS):
 * - "free" (default): models available on the Gemini free tier, so the run costs nothing. Costs are still
 *   reported at paid-tier list prices, which is what the savings comparison needs.
 * - "paid": includes gemini-4-argon (no free tier; needs billing on the key).
 */

const MODEL_SETS = {
  // gemini-3.8-flash was the first choice for "large", but on the free tier it timed out, returned 503s and hit
  // its daily quota within a few calls (Nov 8, 2026). gemini-3.6-flash has the same list price and free tier.
  free: { small: "gemini-3.1-flash-lite", medium: "gemini-3.5-flash-lite", large: "gemini-3.6-flash" },
  paid: { small: "gemini-3.5-flash-lite", medium: "gemini-3.8-flash", large: "gemini-4-argon" },
} as const;

export const MODEL_SET: keyof typeof MODEL_SETS = process.env.PATHWISE_BENCH_MODELS === "paid" ? "paid" : "free";
/** Override any tier with PATHWISE_BENCH_SMALL_MODEL / _MEDIUM_MODEL / _LARGE_MODEL (must have a price below). */
export const MODELS: Record<"small" | "medium" | "large", string> = {
  small: process.env.PATHWISE_BENCH_SMALL_MODEL ?? MODEL_SETS[MODEL_SET].small,
  medium: process.env.PATHWISE_BENCH_MEDIUM_MODEL ?? MODEL_SETS[MODEL_SET].medium,
  large: process.env.PATHWISE_BENCH_LARGE_MODEL ?? MODEL_SETS[MODEL_SET].large,
};
export type Tier = keyof typeof MODELS;

interface Price {
  inputPerM: number;
  outputPerM: number;
}

/** Paid-tier list price of a model on a given date (some have announced increases). */
export function priceOf(model: string, at: Date = new Date()): Price {
  switch (model) {
    case "gemini-3.1-flash-lite":
      return { inputPerM: 0.25, outputPerM: 1.5 };
    case "gemini-3.5-flash-lite":
      return { inputPerM: 0.3, outputPerM: 2.5 };
    case "gemini-3.6-flash":
    case "gemini-3.8-flash":
      return at < new Date("2027-01-01T00:00:00Z")
        ? { inputPerM: 0.75, outputPerM: 3.75 }
        : { inputPerM: 1.5, outputPerM: 7.5 };
    case "gemini-4-argon":
      return at < new Date("2026-12-01T00:00:00Z")
        ? { inputPerM: 2.0, outputPerM: 10.0 }
        : { inputPerM: 4.0, outputPerM: 20.0 };
    default:
      throw new Error(`No price for ${model}`);
  }
}

export function priceFor(tier: Tier, at: Date = new Date()): Price {
  return priceOf(MODELS[tier], at);
}

/** Free tier: requests per minute to stay under (check yours at https://aistudio.google.com/rate-limit). */
export const DEFAULT_RPM = MODEL_SET === "free" ? 10 : 60;

/** The planner's knob space: 3 models x 3 retrieval depths x 2 thinking levels = 18 plans. */
export const SPACE = {
  model: ["small", "medium", "large"],
  topK: [2, 5, 10],
  thinking: ["low", "high"],
};

/** Plans most teams would ship, used as baselines. */
export const BASELINES = {
  max: { model: "large", topK: 10, thinking: "high" },
  typical: { model: "large", topK: 5, thinking: "low" },
  cheap: { model: "small", topK: 2, thinking: "low" },
} as const;

/**
 * Hard cap on API spend across all runs (the ledger persists between runs). On the free tier nothing is
 * billed; the ledger then tracks the list-price equivalent and the cap is just a safety net.
 */
export const BUDGET_USD = Number(process.env.PATHWISE_BENCH_BUDGET ?? 25);

/** Upper bound for one call, used to refuse a call that could cross the budget. */
export const MAX_OUTPUT_TOKENS = 4096;

export const PATHS = {
  data: new URL("./data/", import.meta.url),
  cache: new URL("./cache/", import.meta.url),
  results: new URL("./results/", import.meta.url),
};

/**
 * HotpotQA dev, distractor setting, via the Hugging Face dataset viewer API (100 rows per page).
 * The original host (curtis.ml.cmu.edu) no longer resolves.
 */
export const DATASET_ROWS_URL =
  "https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation";
