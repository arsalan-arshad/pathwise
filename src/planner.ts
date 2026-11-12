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
  /** Shared queries, where at least one of the two plans cleared stopQuality, needed before a plan can be pruned as dominated. Default 8. */
  minDominanceEvidence?: number;
  /** Similarity at which calibrated neighbours are trusted to predict a query's winner. Default 0.9 (>1 disables). */
  shortcutSimilarity?: number;
  /** How many such neighbours must agree on the same winner. Default 3. */
  shortcutNeighbours?: number;
  /** Every Nth searched query also runs one skipped pricier plan, to test monotonicity. Default 10 (0 disables). */
  auditEvery?: number;
  /** Highest violation rate at which skipped pricier plans are still credited. Default 0.05. */
  maxViolationRate?: number;
  /**
   * Times each calibration output is scored; the median (majority for pass/fail) is kept.
   * Default 3 when scorerErrorRate > 0, else 1. Must be odd.
   */
  scoreVotes?: number;
  /**
   * A calibration verdict that passes stopQuality is re-scored up to this many votes in total before it
   * can end a query's search. Stops the "winner's curse": with a noisy judge, trying cheap plans one by one
   * otherwise ends on false passes. Default 7 when scorerErrorRate > 0, else scoreVotes. Must be odd.
   */
  confirmVotes?: number;
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
  /**
   * How often the scorer gives the wrong verdict (0..0.5), e.g. how often your LLM judge disagrees
   * with human labels on a small labelled set. Quality estimates are corrected for it; without this,
   * a judge that is wrong 10% of the time makes even perfect plans look only 90% correct. Stored
   * observation qualities are bias-corrected, so they can fall slightly outside 0..1. Default 0.
   */
  scorerErrorRate?: number;
  calibration?: CalibrationOptions;
}

export interface PlannerStats {
  runs: number;
  totalCostUsd: number;
  fallbacks: number;
  calibrationCostUsd: number;
  calibrationRuns: number;
  /** Scorer (e.g. LLM judge) calls made, calibration and production feedback together. */
  scorerCalls: number;
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
  /** Predicted quality needed to choose a plan; raised above minQuality by sla.confidence. */
  private threshold: number;
  private certification = { quality: NaN, certified: false };
  /** sla.confidence set, SLA not certifiable, onUncertified "safe": every request gets the safe plan. */
  private safeMode = false;
  private safeKey: string | undefined;
  /** Effective scorer error rate of a calibration observation after majority voting. */
  private calibErrorRate = 0;
  private readonly stats: PlannerStats = {
    runs: 0,
    totalCostUsd: 0,
    fallbacks: 0,
    calibrationCostUsd: 0,
    calibrationRuns: 0,
    scorerCalls: 0,
    planUsage: {},
  };

  constructor(options: PlannerOptions<I, O>) {
    const { featurize, calibration, ...rest } = options;
    this.plans = enumeratePlans(rest.space);
    for (const p of this.plans) this.planByKey.set(planKey(p), p);
    this.featurize = featurize ?? (hashedTextFeaturizer() as Featurizer<I>);
    this.opts = { neighbours: 7, priorWeight: 0.5, feedbackRate: 0.1, concurrency: 8, scorerErrorRate: 0, ...rest };
    if (!(this.opts.scorerErrorRate >= 0 && this.opts.scorerErrorRate < 0.5)) {
      throw new Error("scorerErrorRate must be in [0, 0.5)");
    }
    this.calib = {
      strategy: "adaptive",
      seedQueries: 2,
      stopQuality: rest.sla.minQuality,
      minDominanceEvidence: 8,
      shortcutSimilarity: 0.9,
      shortcutNeighbours: 3,
      auditEvery: 10,
      maxViolationRate: 0.05,
      scoreVotes: this.opts.scorerErrorRate > 0 ? 3 : 1,
      confirmVotes: this.opts.scorerErrorRate > 0 ? 7 : (calibration?.scoreVotes ?? 1),
      ...calibration,
    };
    for (const [name, v] of [["scoreVotes", this.calib.scoreVotes], ["confirmVotes", this.calib.confirmVotes]] as const) {
      if (!(Number.isInteger(v) && v % 2 === 1)) throw new Error(`calibration.${name} must be an odd integer`);
    }
    const fb = rest.sla.fallbackPlan;
    if (fb) {
      if (rest.sla.confidence === undefined) throw new Error("sla.fallbackPlan needs sla.confidence");
      if ((rest.sla.onUncertified ?? "safe") !== "safe") throw new Error('sla.fallbackPlan needs onUncertified "safe"');
      if (!this.planByKey.has(planKey(fb))) throw new Error("sla.fallbackPlan must be one of the plans in the space");
    }
    this.calibErrorRate = majorityErrorRate(this.opts.scorerErrorRate, this.calib.scoreVotes);
    this.threshold = rest.sla.minQuality;
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
    this.tuneThreshold();
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
      qualityThreshold: this.threshold,
      certifiedQuality: this.certification.quality,
      slaCertified: this.certification.certified,
      policy: this.safeMode ? "safe-plan" : "routing",
      ...(this.safeMode && this.safePlanKey() !== undefined ? { fixedPlan: this.safePlanKey() } : {}),
    };
  }

  /** Predict quality, cost and latency of every plan for this input. */
  estimate(input: I): PlanEstimate[] {
    if (this.records.length === 0) {
      throw new Error("Planner has no data yet: call calibrate() first");
    }
    return this.estimateFeatures(this.featurize(input));
  }

  /** Estimates from a feature vector, optionally pretending one record does not exist (leave-one-out). */
  private estimateFeatures(f: number[], exclude?: QueryRecord): PlanEstimate[] {
    const ranked = this.records
      .filter((r) => r !== exclude)
      .map((r) => ({ r, sim: Math.max(0, cosine(f, r.features)) }))
      .sort((a, b) => b.sim - a.sim);

    return this.plans.map((plan) => {
      const key = planKey(plan);
      const prior = this.globalMean(key, exclude);
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
        quality: clamp01((q + a * prior.quality) / (w + a)),
        costUsd: (c + a * prior.costUsd) / (w + a),
        latencyMs: (l + a * prior.latencyMs) / (w + a),
        support: w,
      };
    });
  }

  /**
   * Pick the cheapest plan predicted to meet the SLA. Dominated plans are never chosen.
   * With sla.confidence, "meets" uses the calibrated threshold instead of minQuality.
   */
  decide(input: I): Decision {
    const estimates = this.estimate(input);
    if (this.safeMode) {
      // SLA could not be certified: run the declared sla.fallbackPlan, else the plan with the best measured quality.
      const safe = estimates.find((e) => e.planKey === this.safePlanKey());
      if (safe) return { plan: safe.plan, estimate: safe, fallback: true };
    }
    return this.choose(estimates, this.threshold);
  }

  private choose(all: PlanEstimate[], minQuality: number): Decision {
    const { maxLatencyMs, maxCostUsd } = this.opts.sla;
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
    // Nothing meets the SLA: protect quality, pick the best predicted plan. Estimates are capped at 1,
    // so on a tie prefer the pricier plan (usually the more capable one), not the first in enumeration order.
    const best = estimates.reduce((a, b) =>
      b.quality > a.quality || (b.quality === a.quality && b.costUsd > a.costUsd) ? b : a,
    );
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
    this.tuneThreshold();
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
    const { observation } = await this.execute(input, plan, true, this.calib.scoreVotes, {
      stopQuality: this.calib.stopQuality,
      confirmVotes: this.calib.confirmVotes,
    });
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
    // B scored >= A on every query where both ran, with enough such queries where either plan
    // cleared stopQuality.
    const shared = this.records
      .filter((r) => r.observations.length > 1)
      .map((r) => qualityMap(r));
    const useLatency = this.opts.sla.maxLatencyMs !== undefined;
    const stop = this.calib.stopQuality;
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
          if (qb < qa) {
            neverWorse = false;
            break;
          }
          // A query that both plans fail says nothing about which is better, so it is no evidence.
          // Counting it pruned the best plans on real data: hard queries tie at 0 and come first.
          if (qa >= stop || qb >= stop) n++;
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
    // Some "violations" are just scorer mistakes: correct the observed rate for the calibration error rate.
    const raw = after === 0 ? 0 : fails / after;
    return { rate: clamp01(unbias(raw, this.calibErrorRate)), measured: after };
  }

  // ---------------------------------------------------------------- SLA guarantee

  /**
   * With sla.confidence: replay every calibration query leave-one-out (decide as if it were unseen,
   * then look up how the chosen plan actually scored on it) for a range of thresholds, and keep the
   * lowest (cheapest) threshold whose replayed average quality clears minQuality at that confidence
   * (Wilson lower bound). If none does, keep the threshold with the best bound and report uncertified.
   */
  private tuneThreshold() {
    const { minQuality, confidence } = this.opts.sla;
    this.threshold = minQuality;
    this.certification = { quality: NaN, certified: false };
    this.safeMode = false;
    this.safeKey = undefined;
    if (confidence === undefined) return;
    if (!(confidence > 0 && confidence < 1)) throw new Error("sla.confidence must be between 0 and 1");

    // Calibration records only: they know every plan's outcome (measured or implied by the search).
    const replay = this.records.filter((r) => r.winner !== undefined || r.observations.length > 1);
    if (replay.length === 0) return;
    const loo = replay.map((r) => ({ r, estimates: this.estimateFeatures(r.features, r) }));

    const candidates = new Set<number>([minQuality]);
    for (let t = Math.ceil(minQuality * 100) / 100; t <= 1.0001; t += 0.01) candidates.add(Math.min(1, t));
    const z = normalQuantile(confidence);
    let best = { threshold: minQuality, lower: -Infinity };
    for (const t of [...candidates].sort((a, b) => a - b)) {
      let sum = 0;
      for (const { r, estimates } of loo) sum += this.realized(r, this.choose(estimates, t).estimate.planKey);
      // Replayed scores are bias-corrected scorer verdicts. Bound the mean on the scorer's own scale
      // (where Wilson applies), then map the bound back. The mapping is monotone, so confidence carries over.
      const e = this.calibErrorRate;
      const observed = clamp01(clamp01(sum / loo.length) * (1 - 2 * e) + e);
      const lower = unbias(wilsonLower(observed, loo.length, z), e);
      if (lower > best.lower) best = { threshold: t, lower };
      if (lower >= minQuality) {
        this.threshold = t;
        this.certification = { quality: lower, certified: true };
        return;
      }
    }
    // Not certifiable with this data. "safe": every request gets the plan with the best measured
    // quality overall (see safePlanKey). "best-effort": the threshold that came closest.
    this.safeMode = (this.opts.sla.onUncertified ?? "safe") === "safe";
    this.threshold = best.threshold;
    this.certification = { quality: best.lower, certified: false };
  }

  /**
   * Plan for "safe" mode: the non-dominated plan with the best quality replayed over every calibration
   * query, ties to the pricier plan. Raw per-plan averages would be biased (cheap plans are tested on easy
   * and hard queries, pricier ones only where cheaper plans failed), so plans skipped above a query's
   * winner are credited with the winner's score here, even if the monotonicity check failed.
   */
  private safePlanKey(): string | undefined {
    const declared = this.opts.sla.fallbackPlan;
    if (declared) return planKey(declared);
    if (this.safeKey !== undefined) return this.safeKey;
    const replay = this.records.filter((r) => r.winner !== undefined || r.observations.length > 1);
    let best: { key: string; q: number; c: number } | undefined;
    for (const key of this.activeByCost()) {
      const c = this.profiles.get(key)?.costUsd ?? Infinity;
      if (!Number.isFinite(c)) continue;
      let q = 0;
      for (const r of replay) q += this.realized(r, key, true);
      if (!best || q > best.q || (q === best.q && c > best.c)) best = { key, q, c };
    }
    this.safeKey = best?.key;
    return this.safeKey;
  }

  /** How a plan scored on a calibration query: measured, implied by cheapest-first search, or 0 if unknown. */
  private realized(r: QueryRecord, key: string, forceCredit = false): number {
    let sum = 0, n = 0, winnerQuality = NaN;
    for (const o of r.observations) {
      if (Number.isNaN(o.quality)) continue;
      if (o.planKey === key) {
        sum += o.quality;
        n++;
      }
      if (o.planKey === r.winner) winnerQuality = o.quality;
    }
    if (n > 0) return sum / n;
    // Unmeasured. Pricier than the winner: credited with the winner's score if monotonicity holds.
    // Cheaper than the winner (or no winner): the search found cheaper plans failing, so assume it fails.
    const pk = this.profiles.get(key)?.costUsd ?? Infinity;
    const pw = r.winner === undefined ? undefined : this.profiles.get(r.winner)?.costUsd;
    if (pw !== undefined && pk > pw && (this.assumeMonotone || forceCredit) && !Number.isNaN(winnerQuality)) {
      return winnerQuality;
    }
    return 0;
  }

  // ---------------------------------------------------------------- shared helpers

  private async execute(
    input: I,
    plan: Plan,
    score: boolean,
    votes = 1,
    confirm?: { stopQuality: number; confirmVotes: number },
  ) {
    const start = performance.now();
    const result = await this.opts.pipeline(input, plan);
    const latencyMs = result.latencyMs ?? performance.now() - start;
    let quality = NaN;
    if (score) {
      const scores: number[] = [];
      const median = () => [...scores].sort((a, b) => a - b)[(scores.length - 1) / 2];
      const vote = async (n: number) => {
        for (let i = 0; i < n; i++) scores.push(await this.opts.scorer(input, result.output));
        this.stats.scorerCalls += n;
      };
      await vote(votes);
      // A passing verdict can end a calibration search, so confirm it with extra votes first.
      if (confirm && confirm.confirmVotes > votes && median() >= confirm.stopQuality) {
        await vote(confirm.confirmVotes - votes);
      }
      // Median verdict, then correct for the scorer's (voted) error rate so the stored value is unbiased.
      quality = unbias(median(), majorityErrorRate(this.opts.scorerErrorRate, scores.length));
    }
    const observation: Observation = {
      planKey: planKey(plan),
      quality,
      costUsd: result.costUsd,
      latencyMs,
    };
    return { result: { ...result, latencyMs }, observation };
  }

  private globalMean(key: string, exclude?: QueryRecord) {
    let n = 0, quality = 0, costUsd = 0, latencyMs = 0;
    for (const r of this.records) {
      if (r === exclude) continue;
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

/** Wilson score lower bound for a mean in [0, 1] observed over n trials (conservative for graded scores). */
function wilsonLower(mean: number, n: number, z: number): number {
  if (n === 0) return 0;
  const z2 = z * z;
  const centre = mean + z2 / (2 * n);
  const margin = z * Math.sqrt((mean * (1 - mean)) / n + z2 / (4 * n * n));
  return (centre - margin) / (1 + z2 / n);
}

/** One-sided standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9). */
function normalQuantile(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

/** Observed score -> unbiased true-quality estimate, for a scorer that flips verdicts with probability e. */
function unbias(observed: number, e: number): number {
  return e === 0 ? observed : (observed - e) / (1 - 2 * e);
}

/** Probability that the majority of `votes` independent verdicts is wrong, each wrong with probability e. */
function majorityErrorRate(e: number, votes: number): number {
  if (e === 0 || votes === 1) return e;
  let p = 0;
  for (let k = Math.ceil(votes / 2); k <= votes; k++) p += binom(votes, k) * e ** k * (1 - e) ** (votes - k);
  return p;
}

function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}
