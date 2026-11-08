/**
 * Step 3: the main real-RAG benchmark.
 *
 *   npm run bench:rag -- --sla 0.7 [--calibration 100] [--test 200] [--confidence 0.9] [--concurrency 2] [--rpm 10]
 *
 * Calibrates Pathwise on the calibration questions, then answers the test questions with:
 *   - Pathwise, average-based decisions (v0.1 behaviour)
 *   - Pathwise with the SLA guarantee (sla.confidence)
 *   - fixed baselines: always max, a typical default, always cheap
 * Quality = mean token F1 against HotpotQA gold answers. Cost = real API cost (cached calls report the
 * cost they originally had, so every strategy is charged as if it ran alone). Results go to results/.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { createPlanner, hashedTextFeaturizer, planKey, type Plan } from "../../src/index.js";
import { BASELINES, BUDGET_USD, MODELS, PATHS, SPACE } from "./config.js";
import { loadHotpot, sample, type Question } from "./dataset.js";
import { BudgetExceeded, DailyQuotaError, ModelUnavailableError, QuotaError, stopAll, setRpm, spendSoFar } from "./gemini.js";
import { percentile, pool, ragPipeline, scorer, type Answer } from "./pipeline.js";

const args = process.argv.slice(2);
const arg = (name: string, def?: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const SLA = arg("sla");
if (SLA === undefined || Number.isNaN(SLA)) {
  console.error("Pass --sla (run the pilot first; it suggests one).");
  process.exit(1);
}
const CALIBRATION = arg("calibration", 100)!;
const TEST = arg("test", 200)!;
const CONFIDENCE = arg("confidence", 0.9)!;
const CONCURRENCY = arg("concurrency", 2)!;
setRpm(arg("rpm", 0)!);

const all = await loadHotpot();
const { calibration, test } = sample(all, CALIBRATION, TEST);
const featurize = hashedTextFeaturizer();

interface StrategyResult {
  name: string;
  quality: number;
  slaMet: boolean;
  costUsd: number;
  costPerQueryUsd: number;
  latencyP50Ms: number;
  latencyP95Ms: number;
  calibrationCostUsd?: number;
  calibrationRuns?: number;
  planMix?: Record<string, number>;
  report?: unknown;
}

const results: StrategyResult[] = [];
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

async function runPlanner(name: string, confidence: number | undefined): Promise<StrategyResult> {
  const planner = createPlanner<Question, Answer>({
    space: SPACE,
    pipeline: ragPipeline,
    scorer,
    sla: confidence === undefined ? { minQuality: SLA! } : { minQuality: SLA!, confidence },
    featurize: (q) => featurize(q.question),
    feedbackRate: 0,
    concurrency: CONCURRENCY,
  });
  console.log(`\n[${name}] calibrating on ${calibration.length} questions ...`);
  const report = await planner.calibrate(calibration);
  console.log(`[${name}] calibration: ${report.runs} runs, $${report.costUsd.toFixed(2)}, threshold ${report.qualityThreshold.toFixed(2)}${confidence ? `, certified ${report.slaCertified}` : ""}`);
  const rows = await pool(test, CONCURRENCY, async (q) => {
    const r = await planner.run(q);
    return { f1: scorer(q, r.output), cost: r.costUsd, latency: r.latencyMs ?? 0 };
  });
  const stats = planner.report();
  const quality = mean(rows.map((r) => r.f1));
  return {
    name,
    quality,
    slaMet: quality >= SLA!,
    costUsd: stats.totalCostUsd,
    costPerQueryUsd: stats.totalCostUsd / test.length,
    latencyP50Ms: percentile(rows.map((r) => r.latency), 50),
    latencyP95Ms: percentile(rows.map((r) => r.latency), 95),
    calibrationCostUsd: stats.calibrationCostUsd,
    calibrationRuns: stats.calibrationRuns,
    planMix: stats.planUsage,
    report,
  };
}

async function runFixed(name: string, plan: Plan): Promise<StrategyResult> {
  console.log(`\n[${name}] ${planKey(plan)} on ${test.length} questions ...`);
  const rows = await pool(test, CONCURRENCY, async (q) => {
    const r = await ragPipeline(q, plan);
    return { f1: scorer(q, r.output), cost: r.costUsd, latency: r.latencyMs };
  });
  const quality = mean(rows.map((r) => r.f1));
  const cost = rows.reduce((s, r) => s + r.cost, 0);
  return {
    name,
    quality,
    slaMet: quality >= SLA!,
    costUsd: cost,
    costPerQueryUsd: cost / test.length,
    latencyP50Ms: percentile(rows.map((r) => r.latency), 50),
    latencyP95Ms: percentile(rows.map((r) => r.latency), 95),
  };
}

let stopped: string | undefined;
try {
  // Cheapest first, so a budget stop still leaves the most useful results.
  results.push(await runFixed("Always cheap", { ...BASELINES.cheap }));
  results.push(await runPlanner("Pathwise (average-based)", undefined));
  results.push(await runPlanner(`Pathwise (guaranteed, confidence ${CONFIDENCE})`, CONFIDENCE));
  results.push(await runFixed("Typical default", { ...BASELINES.typical }));
  results.push(await runFixed("Always max", { ...BASELINES.max }));
} catch (err) {
  stopAll();
  if (!(err instanceof BudgetExceeded || err instanceof QuotaError || err instanceof DailyQuotaError || err instanceof ModelUnavailableError)) throw err;
  stopped = err.message;
  if (err instanceof DailyQuotaError || err instanceof ModelUnavailableError) stopped += " Results below are partial.";
  console.error(`\n${stopped}`);
}

// ---------------------------------------------------------------- report

const spend = spendSoFar();
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const max = results.find((r) => r.name === "Always max");
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const usd = (x: number) => `$${x.toFixed(2)}`;
const lines = [
  `# Real-RAG benchmark: ${new Date().toISOString().slice(0, 10)}`,
  "",
  `Real Gemini API calls on HotpotQA dev (distractor), ${calibration.length} calibration + ${test.length} test questions.`,
  `Models: ${Object.values(MODELS).join(", ")}. SLA: mean F1 >= ${SLA}.${spend.mock ? " **MOCK MODE: fake model, not real results.**" : ""}`,
  ...(spend.modelSet === "free" && !spend.mock
    ? ["Run on the Gemini free tier (billed $0). All costs below are what the same calls cost at paid-tier list prices."]
    : []),
  "",
  "| Strategy | Mean F1 | SLA met | Cost (test) | $/query | p50 latency | p95 latency | Calibration |",
  "|---|---|---|---|---|---|---|---|",
  ...results.map(
    (r) =>
      `| ${r.name} | ${r.quality.toFixed(3)} | ${r.slaMet ? "yes" : "**no**"} | ${usd(r.costUsd)} | $${r.costPerQueryUsd.toFixed(5)} | ` +
      `${(r.latencyP50Ms / 1000).toFixed(1)} s | ${(r.latencyP95Ms / 1000).toFixed(1)} s | ` +
      `${r.calibrationCostUsd === undefined ? "-" : `${usd(r.calibrationCostUsd)} (${r.calibrationRuns} runs)`} |`,
  ),
  "",
  ...(max
    ? results
        .filter((r) => r.name.startsWith("Pathwise"))
        .map((r) => `- ${r.name}: ${pct(1 - r.costUsd / max.costUsd)} cheaper than always-max, F1 ${r.quality.toFixed(3)} vs ${max.quality.toFixed(3)}.`)
    : []),
  spend.modelSet === "free"
    ? `- List-price equivalent of all API calls so far (each counted once): ${usd(spend.spentUsd)}; billed $0 (free tier).`
    : `- Real API spend so far (all runs, cache included once): ${usd(spend.spentUsd)} of ${usd(BUDGET_USD)}.`,
  ...(stopped ? [`- **Stopped early:** ${stopped}`] : []),
];
await mkdir(PATHS.results, { recursive: true });
const base = new URL(`${spend.mock ? "mock-" : ""}run-${stamp}`, PATHS.results);
await writeFile(new URL(`${base.pathname.split("/").pop()}.md`, PATHS.results), lines.join("\n") + "\n");
await writeFile(
  new URL(`${base.pathname.split("/").pop()}.json`, PATHS.results),
  JSON.stringify({ sla: SLA, confidence: CONFIDENCE, calibration: calibration.length, test: test.length, models: MODELS, results, spend }, null, 2),
);
console.log("\n" + lines.join("\n"));
console.log(`\nSaved to bench/real-rag/results/${base.pathname.split("/").pop()}.md`);
// Exit even if a cancelled call is still sleeping between retries.
process.exit(0);
