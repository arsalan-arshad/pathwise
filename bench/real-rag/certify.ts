/**
 * Split-based certification evaluation (offline, $0: response cache only, no key, no network, test set never read).
 * REAL data: HotpotQA dev distractor, Gemini free-tier models at list prices, the 100 calibration questions that have a FULL
 * 12-plan table, one seed. A / B / C are splits of those 100 questions (default 40 / 30 / 30), repeated over seeded random splits.
 *
 *   npm run bench:rag:certify [-- --splits 200 --a 40 --b 30 --c 30 --confidence 0.9]
 *
 * For each SLA (0.725 and 0.687 from the earlier runs, plus a LOOSE SLA 0.60 that is a harness check, not a product claim):
 * calibrate a fresh planner (sla.confidence) on A; "in-sample certified" is what the planner itself says; freeze the policy and
 * compute ONE Wilson lower bound of its realized quality on B ("split-certified"); report mean quality and SLA miss on C, which
 * is never used for any choice. Compared with the best FIXED plan chosen on A. See certify-core.ts. Does not change the planner.
 * Expect certification near 0 at the real SLAs: at n_B = 30 even a plan at 0.738 cannot certify 0.725. That is reported as is.
 */
process.env.PATHWISE_BENCH_OFFLINE = "1";
const { enumeratePlans, hashedTextFeaturizer, planKey } = await import("../../src/index.js");
const { BASELINES, SPACE } = await import("./config.js");
const { loadHotpot, sample } = await import("./dataset.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");
const core = await import("./analyze-core.js");
const cc = await import("./certify-core.js");

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
const SPLITS = num("splits", 200);
const SIZES: [number, number, number] = [num("a", 40), num("b", 30), num("c", 30)];
const CONF = num("confidence", 0.9);
if (!(SPLITS >= 200)) {
  console.error("--splits must be at least 200.");
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
console.log(`== certify: ${calibration.length} questions with a full ${plans.length}-plan table split ${SIZES.join("/")} (A/B/C), ${SPLITS} seeded splits, confidence ${CONF}`);
console.log("REAL data, one dataset, one seed, in-sample over the same 100 questions: the splits overlap, so they are not independent draws. Test set never read.");
if (missing > 0) {
  console.log(`\nincomplete table: ${missing} calibration pairs missing, nothing measured`);
  process.exit(0);
}
const hashed = hashedTextFeaturizer();
const features = calibration.map((q) => hashed(q.question));
const pct = (x: number) => (Number.isNaN(x) ? "n/a (none certified)" : `${(100 * x).toFixed(1)}%`);
const fmt = (x: number) => x.toFixed(3);
const usd = (x: number) => `$${x.toFixed(5)}`;
const maxF1 = core.meanOf(table.f1[ids.indexOf(maxPlan)]);
console.log(`always-max ${maxPlan}: mean F1 ${fmt(maxF1)} over the 100 questions`);

for (const [label, sla] of [["SLA 0.725 (95% of always-max, earlier runs)", 0.725], ["SLA 0.687 (90% of always-max, earlier runs)", 0.687], ["LOOSE SLA 0.60: harness check, not a product claim", 0.6]] as const) {
  const rs: import("./certify-core.js").SplitResult[] = [];
  for (let s = 0; s < SPLITS; s++) {
    const sp = cc.split3(calibration.length, SIZES, 7000 + s);
    rs.push(await cc.runSplit({ table, features, space: SPACE, ...sp, sla, confidence: CONF, maxPlan }));
  }
  const g = cc.aggregate(rs);
  console.log(`\n== ${label}`);
  console.log(`certification rate: in-sample (the planner's own claim) ${pct(g.inSampleCertRate)} (${g.nInSample}/${g.n}); split (one bound on B, policy frozen) ${pct(g.splitCertRate)} (${g.nSplit}/${g.n})`);
  console.log(`miss on C among certified runs: in-sample ${pct(g.missAmongInSample)} of ${g.nInSample}; split ${pct(g.missAmongSplit)} of ${g.nSplit}   (nominal allowance ${((1 - CONF) * 100).toFixed(0)}%)`);
  console.log(`certified AND missed, as a share of all splits: in-sample ${pct(g.falseCertInSample)}; split ${pct(g.falseCertSplit)}`);
  console.log(`policy: always-max fallback in ${pct(g.fallbackShare)} of splits; certified runs that were not a fallback and cost >1% less than always-max: ${g.certifiedSaving}; certified only because it fell back to always-max: ${g.certifiedOnlyFallback}`);
  console.log(`routing (frozen policy) on C: mean F1 ${fmt(g.meanF1C)}, ${usd(g.costC)}/query (+ ${usd(g.costWithCalibration - g.costC)} calibration on A, spread over only the ${SIZES[2]} C questions, per C query = ${usd(g.costWithCalibration)}); SLA miss in ${pct(g.routingMissRate)} of splits`);
  console.log(`best fixed plan chosen on A, on C: mean F1 ${fmt(g.fixedMeanF1)}, ${usd(g.fixedCost)}/query; SLA miss in ${pct(g.fixedMissRate)} of splits; always-max on C: ${fmt(g.maxMeanF1)}, ${usd(g.maxCost)}/query`);
  console.log(`noise floor of the 'mean on C below the SLA' definition: always-max (true mean ${fmt(maxF1)}) itself "misses" in ${pct(g.maxMissRate)} of splits, because C has only ${SIZES[2]} questions. Read miss rates against that, not against ${((1 - CONF) * 100).toFixed(0)}%.`);
  const diff = core.pairedBootstrap({ f1: rs.map((r) => r.meanC), cost: rs.map((r) => r.costC + r.calibCostPerC) }, { f1: rs.map((r) => r.fixed.meanC), cost: rs.map((r) => r.fixed.costC) });
  console.log(`routing minus fixed plan (per split, calibration cost included for routing; bootstrap over ${diff.n} overlapping splits, so the interval is optimistic): F1 ${diff.f1.mean.toFixed(3)} [${diff.f1.lo.toFixed(3)}, ${diff.f1.hi.toFixed(3)}]; cost ${diff.cost.mean.toFixed(5)} [${diff.cost.lo.toFixed(5)}, ${diff.cost.hi.toFixed(5)}] per query`);
}
console.log("\n(offline: no API calls were made; the test set was not read. Certification needs evidence: with 30 questions in B a plan only a little above the SLA cannot be certified. Nothing here was tuned until it certified.)");
