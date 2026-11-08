/**
 * Step 1 and 2 of the real-RAG benchmark.
 *
 *   npm run bench:rag:smoke     one tiny call per model; prints the raw response so we can check fields and cost
 *   npm run bench:rag:pilot     every plan on the first N calibration questions (default 20): per-plan quality,
 *                               cost, latency, a suggested SLA, and a cost projection for the main run
 *
 * Pilot calls are cached and reused by the main run (same questions, same prompts).
 */
import { MODELS, SPACE, BUDGET_USD, type Tier } from "./config.js";
import { loadHotpot, sample } from "./dataset.js";
import { BudgetExceeded, DailyQuotaError, QuotaError, rawSmokeCall, setRpm, spendSoFar } from "./gemini.js";
import { percentile, pool, ragPipeline, scorer } from "./pipeline.js";
import { enumeratePlans, planKey } from "../../src/index.js";

const args = process.argv.slice(2);
const arg = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};

if (args.includes("--smoke")) {
  for (const tier of Object.keys(MODELS) as Tier[]) {
    const r = await rawSmokeCall(tier, "low", "Reply with exactly one word: ok");
    console.log(`\n=== ${MODELS[tier]} ===\n${JSON.stringify(r, null, 2).slice(0, 3000)}`);
  }
  console.log("\nCheck: status 200, a model_output step with text, and usage token counts.");
  process.exit(0);
}

const N = arg("n", 20);
const CALIBRATION = arg("calibration", 100);
const TEST = arg("test", 200);
setRpm(arg("rpm", 0));
const all = await loadHotpot();
const { calibration } = sample(all, CALIBRATION, TEST);
const questions = calibration.slice(0, N);
const plans = enumeratePlans(SPACE);
console.log(`Pilot: ${plans.length} plans x ${questions.length} questions = ${plans.length * questions.length} calls`);

type Cell = { f1: number; cost: number; latency: number };
const grid = new Map<string, Cell[]>(); // planKey -> per-question results
try {
  for (const plan of plans) {
    const key = planKey(plan);
    const cells = await pool(questions, arg("concurrency", 2), async (q) => {
      const r = await ragPipeline(q, plan);
      return { f1: scorer(q, r.output), cost: r.costUsd, latency: r.latencyMs } satisfies Cell;
    });
    grid.set(key, cells);
    const s = spendSoFar();
    console.log(`  ${key.padEnd(40)} done   spent $${s.spentUsd.toFixed(2)} / $${BUDGET_USD}`);
  }
} catch (err) {
  if (err instanceof BudgetExceeded || err instanceof QuotaError || err instanceof DailyQuotaError) console.error(`\n${err.message}`);
  else throw err;
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const rows = [...grid].map(([key, cells]) => ({
  key,
  f1: mean(cells.map((c) => c.f1)),
  cost: mean(cells.map((c) => c.cost)),
  p50: percentile(cells.map((c) => c.latency), 50),
}));
rows.sort((a, b) => a.cost - b.cost);
console.log(`\nPer plan (${spendSoFar().mock ? "MOCK model" : "real API"}, ${questions.length} questions), cheapest first:\n`);
console.log("Plan".padEnd(40), "Mean F1".padEnd(9), "$/query".padEnd(10), "p50 latency");
for (const r of rows) {
  console.log(r.key.padEnd(40), r.f1.toFixed(3).padEnd(9), `$${r.cost.toFixed(5)}`.padEnd(10), `${(r.p50 / 1000).toFixed(1)} s`);
}

if (rows.length === plans.length) {
  const best = rows.reduce((a, b) => (b.f1 > a.f1 ? b : a));
  const sla = Math.floor(best.f1 * 0.95 * 100) / 100;
  // Cheapest-first calibration cost estimate: per question, cost of plans tried until one reaches the SLA.
  const order = rows.map((r) => r.key);
  let calibPerQ = 0;
  let callsPerQ = 0;
  for (let i = 0; i < questions.length; i++) {
    for (const k of order) {
      const c = grid.get(k)![i];
      calibPerQ += c.cost;
      callsPerQ++;
      if (c.f1 >= sla) break;
    }
  }
  calibPerQ /= questions.length;
  callsPerQ /= questions.length;
  const maxRow = rows.find((r) => r.key === planKey({ model: "large", topK: 10, thinking: "high" }))!;
  const typRow = rows.find((r) => r.key === planKey({ model: "large", topK: 5, thinking: "low" }))!;
  const cheapest = rows[0];
  const planner = (cheapest.cost + best.cost) / 2; // rough upper guess until the real run
  const proj =
    CALIBRATION * calibPerQ + // both planners make the same calibration calls; the cache pays once
    TEST * (maxRow.cost + typRow.cost + cheapest.cost + 2 * planner);
  console.log(`\nBest plan: ${best.key}  mean F1 ${best.f1.toFixed(3)}`);
  console.log(`Suggested SLA (95% of best): --sla ${sla}`);
  const free = spendSoFar().modelSet === "free";
  console.log(
    `Projected main-run cost for --calibration ${CALIBRATION} --test ${TEST}: about $${proj.toFixed(2)} (pessimistic)` +
      (free ? ", list-price equivalent; billed $0 on the free tier" : ""),
  );
  // Calls: calibration search + 2 planner modes + 3 baselines on the test set (cache removes repeats).
  const calls = Math.round(CALIBRATION * callsPerQ + TEST * 5);
  const rpmNow = Number(process.env.PATHWISE_BENCH_RPM ?? arg("rpm", 0)) || (free ? 10 : 60);
  console.log(`About ${calls} API calls (at most): ~${Math.ceil(calls / rpmNow)} minutes at ${rpmNow} requests/minute.` +
    (free ? " If the daily free quota runs out, run the same command the next day; it resumes from the cache." : ""));
}
const s = spendSoFar();
console.log(
  s.mock
    ? `\nMOCK MODE: no real calls.`
    : s.modelSet === "free"
      ? `\nFree tier: you were billed $0. List-price equivalent of all calls so far: $${s.spentUsd.toFixed(2)}.`
      : `\nTotal real spend so far: $${s.spentUsd.toFixed(2)} of $${BUDGET_USD}.`,
);
