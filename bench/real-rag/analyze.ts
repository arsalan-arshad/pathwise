/**
 * Fixed plan vs routing, offline ($0, cache only, no network, no key).
 *
 *   npm run bench:rag:analyze [-- --calibration 100 --test 200 --seed 7]
 *
 * 1. If any calibration (question, plan) pair is missing: prints "incomplete table: N pairs missing, no choice made" and
 *    exits 0. Nothing is chosen from a partial table.
 * 2. With the full table: for SLA = 95% and 90% of always-max's F1 on the CALIBRATION questions, picks the best fixed plan
 *    (cheapest plan meeting the SLA on calibration; ties: higher F1) using calibration only, then reports that plan on the
 *    test questions. If its test pairs are not cached it says what `bench:rag:fill --phase test --plans ...` must fill.
 * 3. Replays the Pathwise strategies (average-based and guaranteed) offline from the cache at the 95% SLA, exactly as run.ts
 *    ran them (same questions, same featurizer, feedbackRate 0), and compares them with the fixed plan with a paired
 *    bootstrap. Pathwise at 90% was never run: "not measured".
 */
process.env.PATHWISE_BENCH_OFFLINE = "1"; // before the imports below read it: a cache miss throws, nothing is requested
const { createPlanner, enumeratePlans, hashedTextFeaturizer, planKey } = await import("../../src/index.js");
const { BASELINES, SPACE } = await import("./config.js");
const { loadHotpot, sample } = await import("./dataset.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");
const core = await import("./analyze-core.js");
type Question = import("./dataset.js").Question;
type Answer = import("./pipeline.js").Answer;

const { existsSync, readdirSync, readFileSync } = await import("node:fs");
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.error("The dataset is not downloaded (bench/real-rag/data/). This tool never uses the network: run `npm run bench:rag:pilot` once to download it.");
  process.exit(1);
}
const args = process.argv.slice(2);
const num = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const CAL = num("calibration", 100);
const TEST = num("test", 200);
const SEED = num("seed", 7);
const CONCURRENCY = num("concurrency", 2); // run.ts default; the order of calls can change early stops, so it is a flag
const { calibration, test } = sample(await loadHotpot(), CAL, TEST, SEED);
const plans = enumeratePlans(SPACE);
const ids = plans.map((p) => planKey(p));
const maxPlan = planKey(BASELINES.max);

/** Reads one set of questions x all plans from the cache. Returns the table and the missing pairs per plan. */
async function load(questions: Question[], only?: string[]) {
  const wanted = only ?? ids;
  const t: import("./analyze-core.js").Table = { plans: [], f1: [], cost: [] };
  const missing = new Map<string, number>();
  for (const plan of plans) {
    const id = planKey(plan);
    if (!wanted.includes(id)) continue;
    const f1: number[] = [];
    const cost: number[] = [];
    let miss = 0;
    for (const q of questions) {
      try {
        const r = await ragPipeline(q, plan);
        f1.push(scorer(q, r.output));
        cost.push(r.costUsd);
      } catch (err) {
        if (!(err instanceof OfflineMiss)) throw err;
        miss++;
      }
    }
    missing.set(id, miss);
    if (miss === 0) {
      t.plans.push(id);
      t.f1.push(f1);
      t.cost.push(cost);
    }
  }
  return { table: t, missing };
}

const cal = await load(calibration);
const calMissing = [...cal.missing.values()].reduce((a, b) => a + b, 0);
if (process.env.PATHWISE_BENCH_MOCK === "1") console.log("SYNTHETIC: PATHWISE_BENCH_MOCK=1, the model answers below are mock answers, not real data.");
console.log(`== analyze: ${calibration.length} calibration / ${test.length} test questions, ${plans.length} plans, seed ${SEED} (HotpotQA dev distractor, Gemini free-tier models at list prices; one dataset, one provider, one seed)`);
if (calMissing > 0) {
  console.log(`\nincomplete table: ${calMissing} pairs missing, no choice made`);
  for (const [id, n] of cal.missing) if (n > 0) console.log(`  ${id}: ${n} missing`);
  console.log("\nFill it with: npm run bench:rag:fill -- --phase calibration --yes-live   (see bench/real-rag/README.md)");
  process.exit(0);
}

const fmt = (x: number) => x.toFixed(3);
const usd = (x: number) => `$${x.toFixed(5)}`;
console.log(`always-max plan: ${maxPlan}`);
console.log("\ncalibration table (all plans, all calibration questions)");
console.log("plan                                   F1     cost/query");
for (let p = 0; p < cal.table.plans.length; p++) {
  console.log(`${cal.table.plans[p].padEnd(38)} ${fmt(core.meanOf(cal.table.f1[p]))}  ${usd(core.meanOf(cal.table.cost[p]))}`);
}
const tableUsd = cal.table.cost.flat().reduce((a, b) => a + b, 0);
console.log(`filling the whole calibration table costs ${plans.length} x ${calibration.length} = ${plans.length * calibration.length} calls, $${tableUsd.toFixed(2)} at list price (free tier bills $0)`);

const fixedByFraction = [0.95, 0.9].map((f) => core.chooseFixed(cal.table, maxPlan, f));
const chosenIds = [...new Set(fixedByFraction.map((c) => c.plan))];

// Pathwise strategies, replayed offline at the 95% SLA (the only SLA they were run at).
const sla95 = fixedByFraction[0].sla;
interface Replay {
  name: string;
  f1: number[];
  cost: number[];
  calRuns: number;
  calCost: number;
  certified?: boolean;
}
interface NotReplayed {
  name: string;
  reason: string;
}
async function replay(name: string, confidence: number | undefined): Promise<Replay | NotReplayed> {
  const featurize = hashedTextFeaturizer();
  const planner = createPlanner<Question, Answer>({
    space: SPACE,
    pipeline: ragPipeline,
    scorer,
    sla: confidence === undefined ? { minQuality: sla95 } : { minQuality: sla95, confidence },
    featurize: (q) => featurize(q.question),
    feedbackRate: 0,
    concurrency: CONCURRENCY,
  });
  try {
    const report = await planner.calibrate(calibration);
    const f1: number[] = [];
    const cost: number[] = [];
    for (const q of test) {
      const r = await planner.run(q);
      f1.push(scorer(q, r.output));
      cost.push(r.costUsd);
    }
    return { name, f1, cost, calRuns: report.runs, calCost: report.costUsd, certified: confidence === undefined ? undefined : report.slaCertified };
  } catch (err) {
    if (!(err instanceof OfflineMiss)) throw err;
    return { name, reason: `not measured: the replay needs a response that is not cached (${err.message})` };
  }
}

/** Aggregates of a recorded real run (bench/real-rag/results/*.json) whose SLA and sizes match this analysis. Never replayed. */
interface Recorded {
  file: string;
  quality: number;
  costPerQueryUsd: number;
  slaMet: boolean;
  calibrationRuns?: number;
  certified?: boolean;
  perQuestion?: { f1: number[]; cost: number[] };
}
function findRecorded(strategy: string, sla: number): Recorded | undefined {
  const dir = process.env.PATHWISE_BENCH_RESULTS_DIR ?? new URL("./results/", import.meta.url).pathname;
  if (!existsSync(dir)) return undefined;
  const files = readdirSync(dir).filter((f) => /^run-.*\.json$/.test(f)).sort().reverse(); // names carry the timestamp: newest first
  for (const f of files) {
    try {
      const d = JSON.parse(readFileSync(`${dir}/${f}`, "utf8")) as { sla?: number; calibration?: number; test?: number; results?: Array<Record<string, unknown>> };
      if (d.sla === undefined || Math.abs(d.sla - sla) > 1e-9 || d.calibration !== calibration.length || d.test !== test.length) continue;
      const r = d.results?.find((x) => x.name === strategy);
      if (!r || typeof r.quality !== "number" || typeof r.costPerQueryUsd !== "number") continue;
      const rep = r.report as { slaCertified?: boolean } | undefined;
      return {
        file: f,
        quality: r.quality,
        costPerQueryUsd: r.costPerQueryUsd,
        slaMet: r.quality >= sla,
        calibrationRuns: typeof r.calibrationRuns === "number" ? r.calibrationRuns : undefined,
        certified: rep?.slaCertified,
        perQuestion: (() => {
          const pq = r.perQuestion as { f1?: unknown; costUsd?: unknown } | undefined;
          const ok = (x: unknown): x is number[] => Array.isArray(x) && x.length === test.length && x.every((v) => typeof v === "number");
          return pq && ok(pq.f1) && ok(pq.costUsd) ? { f1: pq.f1, cost: pq.costUsd } : undefined;
        })(),
      };
    } catch {
      continue; // an unreadable result file is skipped, not fatal
    }
  }
  return undefined;
}
// Replayed only when a chosen plan's test pairs are cached, so a missing test table costs no compute.
let replayed: (Replay | NotReplayed)[] | undefined;
const getReplays = async () => (replayed ??= [await replay("Pathwise (average-based)", undefined), await replay("Pathwise (guaranteed, confidence 0.9)", 0.9)]);

let needFill = false;
for (const c of fixedByFraction) {
  console.log(`\n== SLA ${Math.round(c.fraction * 100)}% of always-max: ${c.sla} (always-max calibration F1 ${fmt(c.maxF1)})`);
  console.log(`best fixed plan (chosen on calibration only): ${c.plan}   calibration F1 ${fmt(c.calF1)}, ${usd(c.calCost)}/query`);
  const need = core.questionsNeeded(c.calF1, c.sla);
  const bound = core.wilsonLower(c.calF1, calibration.length, core.Z_90);
  console.log(
    `can it be certified at n=${calibration.length}? Wilson 90% lower bound ${fmt(bound)} vs SLA ${c.sla}: ${bound >= c.sla ? "yes" : "no"}` +
      (need === null ? " (its observed F1 is not above the SLA)" : `; needs about ${need} questions at this observed F1 (computed, not measured; half the time more)`),
  );
  if (c.fraction !== 0.95) console.log("Pathwise at 90%: not measured (it was run at the 95% SLA only; nothing is inferred from that run).");
  const t = await load(test, [c.plan]);
  const miss = t.missing.get(c.plan) ?? 0;
  if (miss > 0) {
    needFill = true;
    console.log(`test not measured: fill needs ${miss} calls for plan ${c.plan}`);
    continue;
  }
  const fixedF1 = t.table.f1[0];
  const fixedCost = t.table.cost[0];
  const fixedMean = core.meanOf(fixedF1);
  console.log(`test (${test.length} q): F1 ${fmt(fixedMean)}, ${usd(core.meanOf(fixedCost))}/query, SLA ${fixedMean >= c.sla ? "met" : "MISSED"}`);
  if (c.fraction !== 0.95) continue;
  for (const r of await getReplays()) {
    if ("reason" in r) {
      console.log(r.reason);
      const rec = findRecorded(r.name, c.sla);
      if (!rec) {
        console.log(`${r.name}: no recorded real run with SLA ${c.sla}, ${calibration.length} calibration and ${test.length} test questions was found in bench/real-rag/results/.`);
        continue;
      }
      const recF1 = rec.quality;
      console.log(
        `${r.name}: recorded real run, not replayed${rec.perQuestion ? "" : ", no paired interval"} (${rec.file}): test F1 ${fmt(recF1)}, ${usd(rec.costPerQueryUsd)}/query, SLA ${rec.slaMet ? "met" : "MISSED"}` +
          (rec.calibrationRuns === undefined ? "" : `; calibration ${rec.calibrationRuns} runs`) +
          (rec.certified === undefined ? "" : `, certified ${rec.certified}`) +
          (rec.perQuestion ? ". This file stores per-question results, so the interval below is paired." : ". A paired bootstrap needs per-question data, which this file does not store (runs of run.ts before T-017 do not); point estimates only."),
      );
      if (rec.perQuestion) {
        const diff = core.pairedBootstrap(rec.perQuestion, { f1: fixedF1, cost: fixedCost });
        console.log(
          core.verdict(
            { name: `${r.name} (recorded)`, f1: recF1, cost: rec.costPerQueryUsd, meetsSla: rec.slaMet },
            { name: c.plan, f1: fixedMean, cost: core.meanOf(fixedCost), meetsSla: fixedMean >= c.sla },
            diff,
            { sla: c.sla, calibration: calibration.length },
          ),
        );
        continue;
      }
      const dearer = rec.costPerQueryUsd - core.meanOf(fixedCost);
      console.log(
        `point estimates, recorded ${r.name} vs fixed plan ${c.plan}: SLA ${rec.slaMet ? "met" : "MISSED"} vs ${fixedMean >= c.sla ? "met" : "MISSED"}; ` +
          `cost ${usd(rec.costPerQueryUsd)} vs ${usd(core.meanOf(fixedCost))}/query (${dearer > 0 ? `the fixed plan is ${usd(dearer)} cheaper` : `routing is ${usd(-dearer)} cheaper`}); ` +
          `F1 ${fmt(recF1)} vs ${fmt(fixedMean)}. No interval, so no winner is declared (one dataset, one provider, one seed).`,
      );
      continue;
    }
    const rf1 = core.meanOf(r.f1);
    console.log(
      `${r.name}: test F1 ${fmt(rf1)}, ${usd(core.meanOf(r.cost))}/query, SLA ${rf1 >= c.sla ? "met" : "MISSED"}; calibration ${r.calRuns} of ${plans.length * calibration.length} runs, $${r.calCost.toFixed(2)} list` +
        (r.certified === undefined ? "" : `, certified ${r.certified}`),
    );
    const extra = tableUsd - r.calCost;
    const perQuery = core.meanOf(fixedCost) - core.meanOf(r.cost);
    console.log(
      `calibration spend: the fixed plan needed the full table ($${tableUsd.toFixed(2)}, ${plans.length * calibration.length} calls); ${r.name} needed ${r.calRuns} runs ($${r.calCost.toFixed(2)} list), $${extra.toFixed(2)} less. ` +
        (perQuery > 0
          ? `Per query routing is $${perQuery.toFixed(5)} cheaper than the fixed plan, so the smaller calibration is repaid after about ${Math.ceil(Math.max(0, extra) / perQuery)} queries (point estimates, only if routing's quality is acceptable).`
          : "Routing is not cheaper per query than the fixed plan, so routing saves nothing over it."),
    );
    const diff = core.pairedBootstrap({ f1: r.f1, cost: r.cost }, { f1: fixedF1, cost: fixedCost });
    console.log(
      core.verdict(
        { name: r.name, f1: rf1, cost: core.meanOf(r.cost), meetsSla: rf1 >= c.sla },
        { name: c.plan, f1: fixedMean, cost: core.meanOf(fixedCost), meetsSla: fixedMean >= c.sla },
        diff,
        { sla: c.sla, calibration: calibration.length },
      ),
    );
  }
}
if (needFill) {
  console.log(
    `\nnext: fill the test pairs of the chosen plans, then rerun this tool:\n  caffeinate -i npm run bench:rag:fill -- --phase test --plans "${chosenIds.join(",")}" --yes-live`,
  );
}
console.log("\n(offline: no API calls were made)");
