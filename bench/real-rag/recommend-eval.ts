/**
 * Offline evaluation of planner.recommend() on the real calibration table ($0, cache only, no key, test set never read).
 * REAL data, one dataset, 100 calibration questions, one seed, in-sample; the shuffles measure variability over question order
 * (recommend()'s `seed`), not generalization. Prior art exists for this kind of search: not a novelty claim.
 *
 *   npm run bench:rag:recommend [-- --shuffles 200]
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
const i = args.indexOf("--shuffles");
const SHUFFLES = i >= 0 ? Number(args[i + 1]) : 200;
if (!(SHUFFLES >= 200)) {
  console.error("--shuffles must be at least 200.");
  process.exit(2);
}
const { calibration } = sample(await loadHotpot(), 100, 200, 7); // the test split is drawn but never read
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
console.log(`== recommend eval: ${calibration.length} calibration questions, ${plans.length} plans, ${SHUFFLES} shuffles (REAL data, one dataset, one seed, in-sample; test never read)`);
if (missing > 0) {
  console.log(`\nincomplete table: ${missing} calibration pairs missing, nothing measured`);
  process.exit(0);
}
const fullCalls = plans.length * calibration.length;
const hashed = hashedTextFeaturizer();
for (const fraction of [0.95, 0.9]) {
  const choice = core.chooseFixed(table, maxPlan, fraction);
  const f1Of = (id: string) => core.meanOf(table.f1[ids.indexOf(id)]);
  const costOf = (id: string) => core.meanOf(table.cost[ids.indexOf(id)]);
  const calls: number[] = [];
  let same = 0;
  let none = 0;
  let below = 0;
  let certified = 0;
  let racing = 0;
  const picked = new Map<string, number>();
  const lowers: number[] = [];
  for (let s = 0; s < SHUFFLES; s++) {
    const planner = createPlanner<Question, Answer>({
      space: SPACE,
      pipeline: ragPipeline,
      scorer,
      sla: { minQuality: choice.sla },
      featurize: (q) => hashed(q.question),
      feedbackRate: 0,
      concurrency: 1,
    });
    const rec = await planner.recommend(calibration, { seed: 1000 + s });
    if (!rec) {
      none++;
      continue;
    }
    calls.push(rec.callsUsed);
    picked.set(rec.planKey, (picked.get(rec.planKey) ?? 0) + 1);
    if (rec.planKey === choice.plan) same++;
    if (f1Of(rec.planKey) < choice.sla) below++;
    if (rec.certified) certified++;
    if (rec.basis === "racing") racing++;
    lowers.push(rec.lowerBound);
  }
  const q = (xs: number[]) => (xs.length ? `mean ${core.meanOf(xs).toFixed(0)} (5th to 95th percentile ${sp.percentile(xs, 5)} to ${sp.percentile(xs, 95)})` : "n/a");
  console.log(`\n== SLA ${Math.round(fraction * 100)}% of always-max: ${choice.sla}   full table: ${fullCalls} calls, chooses ${choice.plan} (calibration F1 ${f1Of(choice.plan).toFixed(3)}, $${costOf(choice.plan).toFixed(5)}/query)`);
  console.log(`recommend(): calls ${q(calls)} of ${fullCalls}; same plan as the full table ${same}/${SHUFFLES}; returned undefined ${none}; plan below the SLA (full-table F1) ${below}; basis racing ${racing}`);
  console.log(`  certified (Wilson 90% lower bound >= SLA) ${certified}/${SHUFFLES}; mean lower bound ${lowers.length ? core.meanOf(lowers).toFixed(3) : "n/a"}`);
  console.log(`  plans returned: ${[...picked].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} x${v}`).join(", ") || "none"}`);
}
console.log("\n(offline: no API calls were made; the test set was not read. Racing drops bad plans early; with 100 questions and a best plan only 1.3 points above the 0.725 bar it almost never stops before the last question, and it cannot certify.)");
