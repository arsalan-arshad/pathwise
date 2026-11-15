/**
 * Offline experiment ($0, cache only, no network, no key): how many calls does it take to find the cheapest single plan
 * whose mean F1 meets an SLA, and how often is it the right one? REAL data (HotpotQA dev distractor, Gemini free-tier models
 * at list prices), the complete 12 x 100 calibration table, ONE seed, in-sample: the shuffles measure the method's
 * variability over question order, not generalization. The test set is never read.
 *
 *   npm run bench:rag:single-plan [-- --shuffles 300 --z 1.96 --min 10 --calibration 100 --test 200 --seed 7]
 *
 * Methods: (a) full table (baseline, plans x questions calls); (b) racing / successive elimination (Wilson bounds);
 * (c) Pathwise's adaptive calibration, reading the single plan it reports (if it reports one).
 * NOT a novelty claim: configuration search and best-arm identification for LLM/RAG pipelines are published work.
 */
process.env.PATHWISE_BENCH_OFFLINE = "1";
const { createPlanner, enumeratePlans, hashedTextFeaturizer, planKey } = await import("../../src/index.js");
const { BASELINES, SPACE } = await import("./config.js");
const { loadHotpot, sample } = await import("./dataset.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");
const core = await import("./analyze-core.js");
const sp = await import("./single-plan-core.js");
type Question = import("./dataset.js").Question;
type Answer = import("./pipeline.js").Answer;

const { existsSync } = await import("node:fs");
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.error("The dataset is not downloaded (bench/real-rag/data/). This tool never uses the network.");
  process.exit(1);
}
const args = process.argv.slice(2);
const num = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const SHUFFLES = num("shuffles", 300);
const Z = num("z", 1.96);
const MIN_Q = num("min", 10);
const CAL = num("calibration", 100);
const TEST = num("test", 200);
const SEED = num("seed", 7);
if (SHUFFLES < 200) {
  console.error("--shuffles must be at least 200 (T-020).");
  process.exit(2);
}
const { calibration } = sample(await loadHotpot(), CAL, TEST, SEED); // the test split is drawn but never read
const plans = enumeratePlans(SPACE);
const ids = plans.map((p) => planKey(p));
const maxPlan = planKey(BASELINES.max);

const table: import("./analyze-core.js").Table = { plans: ids, f1: [], cost: [] };
let missing = 0;
for (const plan of plans) {
  const f1: number[] = [];
  const cost: number[] = [];
  for (const q of calibration) {
    try {
      const r = await ragPipeline(q, plan);
      f1.push(scorer(q, r.output));
      cost.push(r.costUsd);
    } catch (err) {
      if (!(err instanceof OfflineMiss)) throw err;
      missing++;
    }
  }
  table.f1.push(f1);
  table.cost.push(cost);
}
if (process.env.PATHWISE_BENCH_MOCK === "1") console.log("SYNTHETIC: PATHWISE_BENCH_MOCK=1, the model answers below are mock answers, not real data.");
console.log(`== single-plan: ${calibration.length} calibration questions, ${plans.length} plans, ${SHUFFLES} shuffles, z ${Z}, min ${MIN_Q} questions before eliminating (HotpotQA dev distractor, Gemini free-tier models at list prices; REAL data, one dataset, 100 questions, one seed, in-sample)`);
if (missing > 0) {
  console.log(`\nincomplete table: ${missing} calibration pairs missing, nothing measured`);
  process.exit(0);
}
const fmt = (x: number) => x.toFixed(3);
const usd = (x: number) => `$${x.toFixed(5)}`;
const meanOf = core.meanOf;
const fullCalls = plans.length * calibration.length;
const idx = (id: string) => ids.indexOf(id);
const calF1 = (id: string) => meanOf(table.f1[idx(id)]);
const calCost = (id: string) => meanOf(table.cost[idx(id)]);

for (const fraction of [0.95, 0.9]) {
  const choice = core.chooseFixed(table, maxPlan, fraction);
  console.log(`\n== SLA ${Math.round(fraction * 100)}% of always-max: ${choice.sla}`);
  console.log(`(a) full table: ${fullCalls} calls; chooses ${choice.plan} (calibration F1 ${fmt(choice.calF1)}, ${usd(choice.calCost)}/query)`);

  // (b) racing
  const calls: number[] = [];
  const used: number[] = [];
  let same = 0;
  let none = 0;
  let below = 0;
  const f1s: number[] = [];
  const costs: number[] = [];
  const picked = new Map<string, number>();
  for (let s = 0; s < SHUFFLES; s++) {
    const r = sp.race(table, sp.shuffled(calibration.length, 1000 + s), choice.sla, Z, MIN_Q);
    calls.push(r.calls);
    used.push(r.questionsUsed);
    if (r.plan === null) {
      none++;
      continue;
    }
    picked.set(r.plan, (picked.get(r.plan) ?? 0) + 1);
    if (r.plan === choice.plan) same++;
    const f = calF1(r.plan);
    if (f < choice.sla) below++;
    f1s.push(f);
    costs.push(calCost(r.plan));
  }
  const q = (xs: number[]) => `mean ${meanOf(xs).toFixed(0)} (5th to 95th percentile ${sp.percentile(xs, 5)} to ${sp.percentile(xs, 95)})`;
  console.log(`(b) racing (Wilson z ${Z}, after >= ${MIN_Q} questions): calls ${q(calls)} of ${fullCalls}; questions used ${q(used)} of ${calibration.length}`);
  console.log(
    `    same plan as the full table: ${same}/${SHUFFLES} (${((100 * same) / SHUFFLES).toFixed(0)}%); no feasible plan found: ${none}; chose a plan below the SLA (full-table F1): ${below}` +
      (f1s.length > 0 ? `; chosen plan mean calibration F1 ${fmt(meanOf(f1s))}, ${usd(meanOf(costs))}/query` : ""),
  );
  console.log(`    plans chosen: ${[...picked].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} x${v}`).join(", ") || "none"}`);
  console.log(`    savings vs the full table: ${(100 * (1 - meanOf(calls) / fullCalls)).toFixed(0)}% fewer calls (mean)`);

  // (c) Pathwise adaptive calibration
  for (const confidence of [undefined, 0.9]) {
    const n = Math.min(SHUFFLES, 30); // a calibration replay is heavier than a race
    const runs: number[] = [];
    const policies = new Map<string, number>();
    const fixedPlans = new Map<string, number>();
    let certified = 0;
    for (let s = 0; s < n; s++) {
      const order = sp.shuffled(calibration.length, 5000 + s).map((i) => calibration[i]);
      const featurize = hashedTextFeaturizer();
      const planner = createPlanner<Question, Answer>({
        space: SPACE,
        pipeline: ragPipeline,
        scorer,
        sla: confidence === undefined ? { minQuality: choice.sla } : { minQuality: choice.sla, confidence },
        featurize: (qq) => featurize(qq.question),
        feedbackRate: 0,
        concurrency: 1,
      });
      const report = await planner.calibrate(order);
      runs.push(report.runs);
      policies.set(report.policy, (policies.get(report.policy) ?? 0) + 1);
      if (report.fixedPlan) fixedPlans.set(report.fixedPlan, (fixedPlans.get(report.fixedPlan) ?? 0) + 1);
      if (report.slaCertified) certified++;
    }
    const label = confidence === undefined ? "Pathwise adaptive calibration (average-based)" : "Pathwise adaptive calibration (guaranteed, confidence 0.9)";
    console.log(
      `(c) ${label}, ${n} shuffles of the question order: calibration runs ${q(runs)} of ${fullCalls}; policy ${[...policies].map(([k, v]) => `${k} x${v}`).join(", ")}` +
        (confidence === undefined ? "" : `; certified ${certified}/${n}`),
    );
    console.log(
      fixedPlans.size > 0
        ? `    single plan it reports (fixedPlan): ${[...fixedPlans].map(([k, v]) => `${k} x${v} (calibration F1 ${fmt(calF1(k))}${calF1(k) < choice.sla ? ", BELOW the SLA" : ""})`).join(", ")}; the full-table choice is ${choice.plan}`
        : "    it reports no single plan (policy is routing, there is no fixedPlan): nothing is invented here",
    );
  }
}
console.log("\n(offline: no API calls were made; the test set was not read. One dataset, 100 questions, one seed, in-sample; the shuffles measure variability over question order, not generalization. Configuration search for LLM/RAG pipelines is published work: this is a measurement, not a novelty claim.)");
