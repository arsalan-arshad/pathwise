/**
 * Real-RAG benchmark configuration: models, prices, knobs, budget.
 * Prices: official Gemini API paid tier, checked Nov 8, 2026 (https://aistudio.google.com/docs/pricing),
 * USD per 1M tokens, thinking tokens billed at the output price.
 */

export const MODELS = {
  small: "gemini-3.5-flash-lite",
  medium: "gemini-3.8-flash",
  large: "gemini-4-argon",
} as const;
export type Tier = keyof typeof MODELS;

interface Price {
  inputPerM: number;
  outputPerM: number;
}

/** Price for a model on a given date (two of the three have announced increases). */
export function priceFor(tier: Tier, at: Date = new Date()): Price {
  switch (tier) {
    case "small":
      return { inputPerM: 0.3, outputPerM: 2.5 };
    case "medium":
      return at < new Date("2027-01-01T00:00:00Z")
        ? { inputPerM: 0.75, outputPerM: 3.75 }
        : { inputPerM: 1.5, outputPerM: 7.5 };
    case "large":
      return at < new Date("2026-12-01T00:00:00Z")
        ? { inputPerM: 2.0, outputPerM: 10.0 }
        : { inputPerM: 4.0, outputPerM: 20.0 };
  }
}

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

/** Hard cap on real API spend across all runs (the ledger persists between runs). */
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
