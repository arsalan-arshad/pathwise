import { cosine, hashedTextFeaturizer } from "./features.js";
import { enumeratePlans, planKey } from "./space.js";
import type {
  CalibrationReport,
  Decision,
  Featurizer,
  KnobSpace,
  Observation,
  Pipeline,
  Plan,
  PlanEstimate,
  QueryRecord,
  RunResult,
  Scorer,
  SLA,
} from "./types.js";

export interface CalibrationOptions {
  /**
   * "adaptive" (default): sweep a few diverse seed queries, then test plans cheapest-first per query,
   * stop at the first plan that wins, and prune plans a cheaper plan never loses to.
   * "exhaustive": run every plan on every sample (v0.1 behaviour).
   */
  strategy?: "adaptive" | "exhaustive";
  /** Diverse samples that get a full sweep, to learn plan costs and check assumptions. Default 2. */
  seedQueries?: number;
  /** A plan "wins" a query when its score reaches this. Default: sla.minQuality. */
  stopQuality?: number;
  /** Shared queries needed before a plan can be pruned as dominated. Default 8. */
  minDominanceEvidence?: number;
  /** Similarity at which calibrated neighbours are trusted to predict a query's winner. Default 0.9 (>1 disables). */
  shortcutSimilarity?: number;
  /** How many such neighbours must agree on the same winner. Default 3. */
  shortcutNeighbours?: number;
  /** Every Nth searched query also runs one skipped pricier plan, to test monotonicity. Default 10 (0 disables). */
  auditEvery?: number;
  /** Highest violation rate at which skipped pricier plans are still credited. Default 0.05. */
  maxViolationRate?: number;
}

export interface PlannerOptions<I, O> {
  space: KnobSpace;
  pipeline: Pipeline<I, O>;
  scorer: Scorer<I, O>;
  sla: SLA;
  featurize?: Featurizer<I>;
  /** How many similar past queries to learn from per decision (per plan). */
  neighbours?: number;
  /** Strength of the global prior per plan (in "virtual neighbours"). */
  priorWeight?: number;
  /** Fraction of production runs to score and learn from (0..1). */
  feedbackRate?: number;
  /** Parallel pipeline runs during calibration. */
  concurrency?: number;
  calibration?: CalibrationOptions;
}

export interface PlannerStats {
  runs: number;
  totalCostUsd: number;
  fallbacks: number;
  calibrationCostUsd: number;
  calibrationRuns: number;
  planUsage: Record<string, number>;
}

interface PlanProfile {
  costUsd: number;
  latencyMs: number;
}

export class Planner<I, O> {
  readonly plans: Plan[];
  private readonly planByKey = new Map<string, Plan>();
  private readonly records: QueryRecord[] = [];
  private readonly featurize: Featurizer<I>;
  private readonly opts: Required<Omit<PlannerOptions<I, O>, "featurize" | "calibration">>;
  private readonly calib: Required<CalibrationOptions>;
  /** Mean cost / latency per plan, refreshed during calibration. */
  private profiles = new Map<string, PlanProfile>();
  /** Plans a cheaper plan was never worse than on enough shared queries. */
  private dominated = new Set<string>();
  /** Credit plans skipped above a query's winner with the winner's score (see QueryRecord.winner). */
  private assumeMonotone = false;
  private violation = { rate: 0, measured: 0 };
  private readonly stats: PlannerStats = {
    runs: 0,
    totalCostUsd: 0,
    fallbacks: 0,
    calibrationCostUsd: 0,
    calibrationRuns: 0,
    planUsage: {},
  };

  constructor(options: PlannerOptions<I, O>) {
    const { featurize, calibration, ...rest } = options;
    this.plans = enumeratePlans(rest.space);
    for (const p of this.plans) this.planByKey.set(planKey(p), p);
    this.featurize = featurize ?? (hashedTextFeaturizer() as Featurizer<I>);
    this.opts = { neighbours: 7, priorWeight: 0.5, feedbackRate: 0.1, concurrency: 8, ...rest };
    this.calib = {
      strategy: "adaptive",
      seedQueries: 2,
      stopQuality: rest.sla.minQuality,
      minDominanceEvidence: 8,
      shortcutSimilarity: 0.9,
      shortcutNeighbours: 3,
      auditEvery: 10,
      maxViolationRate: 0.05,
      ...calibration,
    };
  }

  /**
   * Profiling step: runs plans on a sample of real queries and scores them.
   * This is the one-off "learning" cost before the planner can save money.
   */
  async calibrate(samples: I[]): Promise<CalibrationReport> {
    const costBefore = this.stats.calibrationCostUsd;
    const runsBefore = this.stats.calibrationRuns;
    const items = samples.map((input) => ({
      input,
      record: { features: this.featurize(input), observations: [] } as QueryRecord,
    }));

    let nSwept: number;
    let shortcutQueries = 0;
    let audits = 0;

    if (this.calib.strategy === "exhaustive") {
      nSwept = items.length;
      await this.sweep(items);
    } else {
      // Most diverse queries first: seeds cover the space, near-duplicates come last and can shortcut.
      const ordered = diversityOrder(items.map((i) => i.record.features)).map((i) => items[i]);
      const plansWithoutData = this.plans.some((p) => !this.profiles.has(planKey(p)));
      const nSeeds = Math.min(ordered.length, Math.max(plansWithoutData ? 1 : 0, this.calib.seedQueries));
      const seeds = ordered.slice(0, nSeeds);
      nSwept = seeds.length;
      await this.sweep(seeds);
      this.refresh();

      const completed: QueryRecord[] = seeds.map((i) => i.record);
      let searched = 0;
      const jobs = ordered.slice(nSeeds).map((item) => async () => {
        const nth = ++searched;
        this.records.push(item.record);
        if (await this.search(item.input, item.record, completed)) shortcutQueries++;
        item.record.winner = this.winnerOf(item.record);
        if (this.calib.auditEvery > 0 && nth % this.calib.auditEvery === 0) {
          if (await this.audit(item.input, item.record, nth / this.calib.auditEvery)) audits++;
        }
        completed.push(item.record);
        this.refresh();
      });
      await runWithConcurrency(jobs, this.opts.concurrency);
    }

    this.refresh();
    return {
      strategy: this.calib.strategy,
      samples: samples.length,
      seedQueries: nSwept,
      runs: this.stats.calibrationRuns - runsBefore,
      exhaustiveRuns: samples.length * this.plans.length,
      costUsd: this.stats.calibrationCostUsd - costBefore,
      shortcutQueries,
      audits,
      prunedPlans: [...this.dominated].sort(),
      monotonicityViolationRate: this.violation.rate,
      assumeMonotone: this.assumeMonotone,
    };
  }

  /** Predict quality, cost and latency of every plan for this input. */
  estimate(input: I): PlanEstimate[] {
    if (this.records.length === 0) {
      throw new Error("Planner has no data yet: call calibrate() first");
    }
    const f = this.featurize(input);
    const ranked = this.records
      .map((r) => ({ r, sim: Math.max(0, cosine(f, r.features)) }))
      .sort((a, b) => b.sim - a.sim);

    return this.plans.map((plan) => {
      const key = planKey(plan);
      const prior = this.globalMean(key);
      // k nearest queries *with evidence for this plan*: measured, or skipped above that query's
      // winner (credited only if monotonicity checks out). With a full sweep this is plain k-NN.
      let w = 0, q = 0, c = 0, l = 0, used = 0;
      for (const { r, sim } of ranked) {
        if (used >= this.opts.neighbours) break;
        const weight = sim * sim; // favour the closest matches
        let seen = false;
        for (const o of r.observations) {
          if (o.planKey !== key || Number.isNaN(o.quality)) continue;
          seen = true;
          w += weight;
          q += weight * o.quality;
          c += weight * o.costUsd;
          l += weight * o.latencyMs;
        }
        if (!seen) {
          const imp = this.imputed(r, key);
          if (imp) {
            seen = true;
            w += weight;
            q += weight * imp.quality;
            c += weight * imp.costUsd;
            l += weight * imp.latencyMs;
          }
        }
        if (seen) used++;
      }
      const a = this.opts.priorWeight;
      return {
        plan,
        planKey: key,
        quality: (q + a * prior.quality) / (w + a),
        costUsd: (c + a * prior.costUsd) / (w + a),
        latencyMs: (l + a * prior.latencyMs) / (w + a),
        support: w,
      };
    });
  }

  /** Pick the cheapest plan predicted to meet the SLA. Dominated plans are never chosen. */
  decide(input: I): Decision {
    const { minQuality, maxLatencyMs, maxCostUsd } = this.opts.sla;
    const all = this.estimate(input);
    const active = all.filter((e) => !this.dominated.has(e.planKey));
    const estimates = active.length > 0 ? active : all;
    const feasible = estimates.filter(
      (e) =>
        e.quality >= minQuality &&
        (maxLatencyMs === undefined || e.latencyMs <= maxLatencyMs) &&
        (maxCostUsd === undefined || e.costUsd <= maxCostUsd),
    );
    if (feasible.length > 0) {
      const best = feasible.reduce((a, b) => (b.costUsd < a.costUsd ? b : a));
      return { plan: best.plan, estimate: best, fallback: false };
    }
    // Nothing meets the SLA: protect quality, pick the best predicted plan.
    const best = estimates.reduce((a, b) => (b.quality > a.quality ? b : a));
    return { plan: best.plan, estimate: best, fallback: true };
  }

  /** Production entry point: decide, execute, and occasionally learn. */
  async run(input: I): Promise<RunResult<O> & { plan: Plan; decision: Decision }> {
    const decision = this.decide(input);
    const learn = Math.random() < this.opts.feedbackRate;
    const { result, observation } = await this.execute(input, decision.plan, learn);
    if (learn) {
      this.records.push({ features: this.featurize(input), observations: [observation] });
    }
    this.stats.runs++;
    this.stats.totalCostUsd += result.costUsd;
    if (decision.fallback) this.stats.fallbacks++;
    const key = decision.estimate.planKey;
    this.stats.planUsage[key] = (this.stats.planUsage[key] ?? 0) + 1;
    return { ...result, plan: decision.plan, decision };
  }

  report(): PlannerStats {
    return structuredClone(this.stats);
  }

  /** Export learned data so it survives restarts (save to a DB or file). */
  exportState(): QueryRecord[] {
    return structuredClone(this.records);
  }

  importState(records: QueryRecord[]): void {
    this.records.push(...structuredClone(records));
    this.refresh();
  }

  // ---------------------------------------------------------------- calibration internals

  /** Run every plan on every given query. */
  private async sweep(items: Array<{ input: I; record: QueryRecord }>) {
    const jobs: Array<() => Promise<void>> = [];
    for (const { input, record } of items) {
      this.records.push(record);
      for (const plan of this.plans) {
        jobs.push(async () => {
          await this.calibrationRun(input, plan, record);
        });
      }
    }
    await runWithConcurrency(jobs, this.opts.concurrency);
  }

  /**
   * Cheapest-first search for one query. Returns true if a predicted winner was
   * verified (shortcut) instead of searching from the bottom.
   */
  private async search(input: I, record: QueryRecord, completed: QueryRecord[]): Promise<boolean> {
    const stop = this.calib.stopQuality;
    const measured = new Map<string, number>();
    const run = async (key: string) => {
      const q = await this.calibrationRun(input, this.planByKey.get(key)!, record);
      measured.set(key, q);
      return q;
    };

    // Shortcut: near-duplicate neighbours agree on winner W. Verify the boundary:
    // W must win and the next cheaper plan must lose. Otherwise do the full search.
    const predicted = this.predictWinner(record.features, completed);
    if (predicted !== undefined) {
      const order = this.activeByCost();
      const i = order.indexOf(predicted);
      if (i >= 0 && (await run(order[i])) >= stop) {
        if (i === 0 || (await run(order[i - 1])) < stop) return true;
      }
    }

    // Full search: cheapest untested plan first, stop at the first winner.
    for (;;) {
      let next: string | undefined;
      for (const key of this.activeByCost()) {
        const q = measured.get(key);
        if (q === undefined) {
          next = key;
          break;
        }
        if (q >= stop) return false; // already have the cheapest winner
      }
      if (next === undefined) return false; // nothing wins this query
      if ((await run(next)) >= stop) return false;
    }
  }

  /** Run one skipped pricier plan on this query, to test that pricier plans do not lose. */
  private async audit(input: I, record: QueryRecord, turn: number): Promise<boolean> {
    if (record.winner === undefined) return false;
    const order = this.activeByCost();
    const measured = new Set(record.observations.map((o) => o.planKey));
    const above = order.slice(order.indexOf(record.winner) + 1).filter((k) => !measured.has(k));
    if (above.length === 0) return false;
    await this.calibrationRun(input, this.planByKey.get(above[turn % above.length])!, record);
    return true;
  }

  /** Evidence for a plan that was skipped because a cheaper plan already won this query. */
  private imputed(r: QueryRecord, key: string) {
    if (!this.assumeMonotone || r.winner === undefined || this.dominated.has(key)) return undefined;
    const pw = this.profiles.get(r.winner);
    const pk = this.profiles.get(key);
    if (!pw || !pk || !(pk.costUsd > pw.costUsd)) return undefined;
    let quality = NaN;
    for (const o of r.observations) {
      if (o.planKey === key) return undefined; // measured: real data wins
      if (o.planKey === r.winner) quality = o.quality;
    }
    if (Number.isNaN(quality)) return undefined;
    return { quality, costUsd: pk.costUsd, latencyMs: pk.latencyMs };
  }

  private async calibrationRun(input: I, plan: Plan, record: QueryRecord): Promise<number> {
    const { observation } = await this.execute(input, plan, true);
    record.observations.push(observation);
    this.stats.calibrationCostUsd += observation.costUsd;
    this.stats.calibrationRuns++;
    return observation.quality;
  }

  /** Winner shared by the closest calibrated queries, if they are near-duplicates and agree. */
  private predictWinner(features: number[], completed: QueryRecord[]): string | undefined {
    const m = this.calib.shortcutNeighbours;
    if (m < 1 || completed.length < m) return undefined;
    const top = completed
      .map((r) => ({ r, sim: cosine(features, r.features) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, m);
    let winner: string | undefined;
    for (const { r, sim } of top) {
      if (sim < this.calib.shortcutSimilarity) return undefined;
      const w = this.winnerOf(r);
      if (w === undefined || (winner !== undefined && w !== winner)) return undefined;
      winner = w;
    }
    return winner;
  }

  /** Cheapest non-dominated plan that won this query. */
  private winnerOf(r: QueryRecord): string | undefined {
    let best: string | undefined;
    let bestCost = Infinity;
    for (const o of r.observations) {
      if (!(o.quality >= this.calib.stopQuality) || this.dominated.has(o.planKey)) continue;
      const cost = this.profiles.get(o.planKey)?.costUsd ?? Infinity;
      if (cost < bestCost) {
        bestCost = cost;
        best = o.planKey;
      }
    }
    return best;
  }

  /** Non-dominated plan keys, cheapest first. */
  private activeByCost(): string[] {
    const cost = (k: string) => this.profiles.get(k)?.costUsd ?? Infinity;
    return [...this.planByKey.keys()]
      .filter((k) => !this.dominated.has(k))
      .sort((a, b) => cost(a) - cost(b) || (a < b ? -1 : 1));
  }

  /** Recompute plan cost profiles and the dominated set from all records. */
  private refresh() {
    const sums = new Map<string, { n: number; c: number; l: number }>();
    for (const r of this.records) {
      for (const o of r.observations) {
        const s = sums.get(o.planKey) ?? { n: 0, c: 0, l: 0 };
        s.n++;
        s.c += o.costUsd;
        s.l += o.latencyMs;
        sums.set(o.planKey, s);
      }
    }
    this.profiles = new Map([...sums].map(([k, s]) => [k, { costUsd: s.c / s.n, latencyMs: s.l / s.n }]));

    // A is dominated by B if B is cheaper (and no slower, when latency is constrained) and
    // B scored >= A on every query where both ran, with enough such queries.
    const shared = this.records
      .filter((r) => r.observations.length > 1)
      .map((r) => qualityMap(r));
    const useLatency = this.opts.sla.maxLatencyMs !== undefined;
    const dominated = new Set<string>();
    const keys = [...this.planByKey.keys()];
    for (const a of keys) {
      const pa = this.profiles.get(a);
      if (!pa) continue;
      for (const b of keys) {
        const pb = this.profiles.get(b);
        if (b === a || !pb) continue;
        const cheaper = pb.costUsd < pa.costUsd || (pb.costUsd === pa.costUsd && b < a);
        if (!cheaper || (useLatency && pb.latencyMs > pa.latencyMs)) continue;
        let n = 0;
        let neverWorse = true;
        for (const m of shared) {
          const qa = m.get(a);
          const qb = m.get(b);
          if (qa === undefined || qb === undefined) continue;
          n++;
          if (qb < qa) {
            neverWorse = false;
            break;
          }
        }
        if (neverWorse && n >= this.calib.minDominanceEvidence) {
          dominated.add(a);
          break;
        }
      }
    }
    this.dominated = dominated;
    this.violation = this.violationRate();
    this.assumeMonotone =
      this.violation.measured > 0 && this.violation.rate <= this.calib.maxViolationRate;
  }

  /** Across all records: share of measured plans pricier than the query's cheapest winner that still fail. */
  private violationRate(): { rate: number; measured: number } {
    const order = this.activeByCost();
    let after = 0, fails = 0;
    for (const r of this.records) {
      const m = qualityMap(r);
      let won = false;
      for (const key of order) {
        const q = m.get(key);
        if (q === undefined) continue;
        if (won) {
          after++;
          if (q < this.calib.stopQuality) fails++;
        } else if (q >= this.calib.stopQuality) {
          won = true;
        }
      }
    }
    return { rate: after === 0 ? 0 : fails / after, measured: after };
  }

  // ---------------------------------------------------------------- shared helpers

  private async execute(input: I, plan: Plan, score: boolean) {
    const start = performance.now();
    const result = await this.opts.pipeline(input, plan);
    const latencyMs = result.latencyMs ?? performance.now() - start;
    const quality = score ? await this.opts.scorer(input, result.output) : NaN;
    const observation: Observation = {
      planKey: planKey(plan),
      quality,
      costUsd: result.costUsd,
      latencyMs,
    };
    return { result: { ...result, latencyMs }, observation };
  }

  private globalMean(key: string) {
    let n = 0, quality = 0, costUsd = 0, latencyMs = 0;
    for (const r of this.records) {
      for (const o of r.observations) {
        if (o.planKey !== key || Number.isNaN(o.quality)) continue;
        n++;
        quality += o.quality;
        costUsd += o.costUsd;
        latencyMs += o.latencyMs;
      }
      const imp = this.imputed(r, key);
      if (imp) {
        n++;
        quality += imp.quality;
        costUsd += imp.costUsd;
        latencyMs += imp.latencyMs;
      }
    }
    if (n === 0) return { quality: 0, costUsd: Infinity, latencyMs: Infinity };
    return { quality: quality / n, costUsd: costUsd / n, latencyMs: latencyMs / n };
  }
}

export function createPlanner<I, O>(options: PlannerOptions<I, O>): Planner<I, O> {
  return new Planner(options);
}

function qualityMap(r: QueryRecord): Map<string, number> {
  const m = new Map<string, number>();
  for (const o of r.observations) {
    if (Number.isNaN(o.quality)) continue;
    m.set(o.planKey, Math.min(m.get(o.planKey) ?? Infinity, o.quality));
  }
  return m;
}

/** Farthest-point ordering: each next query is the one least similar to everything picked so far. */
function diversityOrder(features: number[][]): number[] {
  const n = features.length;
  if (n === 0) return [];
  const order = [0];
  const picked = new Array<boolean>(n).fill(false);
  picked[0] = true;
  const minDist = features.map((f) => 1 - cosine(f, features[0]));
  while (order.length < n) {
    let best = -1;
    for (let j = 0; j < n; j++) {
      if (!picked[j] && (best < 0 || minDist[j] > minDist[best])) best = j;
    }
    picked[best] = true;
    order.push(best);
    for (let j = 0; j < n; j++) {
      if (!picked[j]) minDist[j] = Math.min(minDist[j], 1 - cosine(features[j], features[best]));
    }
  }
  return order;
}

async function runWithConcurrency(jobs: Array<() => Promise<void>>, limit: number) {
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, jobs.length) }, async () => {
    while (next < jobs.length) {
      const job = jobs[next++];
      await job();
    }
  });
  await Promise.all(workers);
}
