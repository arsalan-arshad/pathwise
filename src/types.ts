/** A single tunable setting in a pipeline, e.g. model name, top_k, rerank on/off. */
export type KnobValue = string | number | boolean;

/** Every knob and the values it is allowed to take. */
export type KnobSpace = Record<string, readonly KnobValue[]>;

/** One concrete choice for every knob: a full execution plan. */
export type Plan = Record<string, KnobValue>;

/** What the developer's pipeline returns for one run. */
export interface RunResult<O> {
  output: O;
  /** Dollar cost of this run (model tokens, reranker calls, etc.). */
  costUsd: number;
  /** Optional: report latency yourself (e.g. simulated). Otherwise it is measured. */
  latencyMs?: number;
}

/** The developer's existing RAG / agent pipeline, parameterised by a plan. */
export type Pipeline<I, O> = (input: I, plan: Plan) => Promise<RunResult<O>>;

/** Scores an output from 0 (wrong) to 1 (perfect): eval-set check, LLM judge, etc. */
export type Scorer<I, O> = (input: I, output: O) => Promise<number> | number;

/** Turns an input into a numeric feature vector used to find similar past queries. */
export type Featurizer<I> = (input: I) => number[];

/** The service level the developer declares. The planner meets it as cheaply as possible. */
export interface SLA {
  /** Minimum predicted quality (0..1), e.g. 0.9 = "90% of answers must be right". */
  minQuality: number;
  maxLatencyMs?: number;
  maxCostUsd?: number;
}

/** One measured run of one plan on one query. */
export interface Observation {
  planKey: string;
  quality: number;
  costUsd: number;
  latencyMs: number;
}

/** Everything we know about one query we have seen. */
export interface QueryRecord {
  features: number[];
  observations: Observation[];
  /**
   * Set by adaptive calibration: the cheapest plan that won this query in a cheapest-first search.
   * Pricier plans were skipped; when the monotonicity check passes they are assumed to score
   * at least as well as this one.
   */
  winner?: string;
}

/** The planner's prediction for one plan on one query. */
export interface PlanEstimate {
  plan: Plan;
  planKey: string;
  quality: number;
  costUsd: number;
  latencyMs: number;
  /** How much evidence backs this estimate (sum of neighbour similarity). */
  support: number;
}

export interface Decision {
  plan: Plan;
  estimate: PlanEstimate;
  /** True when no plan met the SLA and we fell back to the highest-quality plan. */
  fallback: boolean;
}

/** What one calibrate() call did and what it cost. */
export interface CalibrationReport {
  strategy: "adaptive" | "exhaustive";
  samples: number;
  /** Samples that got a full sweep of every plan. */
  seedQueries: number;
  /** Pipeline runs actually made. */
  runs: number;
  /** Runs an exhaustive sweep would have needed (samples x plans). */
  exhaustiveRuns: number;
  costUsd: number;
  /** Queries settled by verifying a winner predicted from near-duplicate neighbours. */
  shortcutQueries: number;
  /** Extra runs of a skipped pricier plan, used to test the "pricier is not worse" assumption. */
  audits: number;
  /** Plans pruned as dominated: a cheaper plan was never worse on shared queries. decide() skips them. */
  prunedPlans: string[];
  /**
   * On seed sweeps and audits: share of measured plans pricier than a query's cheapest winner that
   * still failed it. Early stopping assumes this is near 0.
   */
  monotonicityViolationRate: number;
  /**
   * True when the violation rate is within CalibrationOptions.maxViolationRate, so skipped pricier
   * plans are credited with the winner's score. False = conservative (they get no credit).
   */
  assumeMonotone: boolean;
}
