/**
 * Plan-table fill: the LIVE counterpart of table.ts. Fills the missing (question, plan) pairs in the response cache
 * so the whole table can be compared (see analyze.ts). Free Gemini models only.
 *
 *   npm run bench:rag:fill -- --phase calibration                       dry run: what is missing, no call
 *   npm run bench:rag:fill -- --phase calibration --yes-live            fill every missing calibration pair
 *   npm run bench:rag:fill -- --phase test --plans <id>[,<id>] [--yes-live]   fill ONLY the named plans on the test set
 *
 * Plan ids are the ones table.ts prints (e.g. model=small|thinking=low|topK=5). Test pairs are never filled for a
 * plan you did not name, so nothing is chosen using test results. At most 4 plans per invocation.
 *
 * Options: --calibration 100 --test 200 --seed 7 (same sampling as run.ts / replay.ts / table.ts), --rpm N (total
 * requests per minute across both models; default 10), --progress N (print every N calls; default 25).
 *
 * Safety: without --yes-live nothing is sent. --yes-live is refused with PATHWISE_BENCH_OFFLINE=1, with the paid model
 * set, or with any model outside gemini-3.1-flash-lite / gemini-3.5-flash-lite. Every response is cached as it arrives, so a
 * rerun (or Ctrl-C and rerun) skips finished pairs. When a model's daily quota is used up it stops that model, goes on
 * with the other, prints what remains and exits 0. The key is read only by gemini.ts (GEMINI_API_KEY); this file never reads,
 * prints or logs it and never touches .env.
 */
import { enumeratePlans, planKey, type Plan } from "../../src/index.js";
import { BUDGET_USD, DEFAULT_RPM, MODEL_SET, MODELS, SPACE, type Tier } from "./config.js";
import { loadHotpot, sample, type Question } from "./dataset.js";
import { BudgetExceeded, DailyQuotaError, ModelUnavailableError, QuotaError, peekCache, setRpm, spendSoFar, stopAll } from "./gemini.js";
import { buildPrompt, ragPipeline } from "./pipeline.js";

/** The only models this tool may call (the free set, no billing). */
const FREE_MODELS = new Set(["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]);
const MAX_PLANS_PER_TEST_FILL = 4;

function fail(msg: string): never {
  console.error(`refused: ${msg}`);
  process.exit(2);
}

const args = process.argv.slice(2);
const flag = (name: string) => args.includes(`--${name}`);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const num = (name: string, def: number) => {
  const v = opt(name);
  if (v === undefined) return def;
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) fail(`--${name} must be a positive number (got "${v}")`);
  return n;
};

const phase = opt("phase");
if (phase !== "calibration" && phase !== "test") fail("--phase calibration|test is required");
const live = flag("yes-live");
const progressEvery = Math.round(num("progress", 25));
const planIds = opt("plans")
  ?.split(",")
  .map((s) => s.trim())
  .filter(Boolean);

if (phase === "test" && (!planIds || planIds.length === 0)) {
  fail("--phase test needs --plans <id>[,<id>]: test pairs are filled only for plans chosen on the calibration set (see bench:rag:analyze).");
}
if (phase === "calibration" && planIds) fail("--plans is for --phase test only; calibration fills every missing pair.");
if (planIds && planIds.length > MAX_PLANS_PER_TEST_FILL) fail(`at most ${MAX_PLANS_PER_TEST_FILL} plans per invocation (got ${planIds.length}).`);
if (live && process.env.PATHWISE_BENCH_OFFLINE === "1") fail("--yes-live cannot be combined with PATHWISE_BENCH_OFFLINE=1.");
if (MODEL_SET !== "free") fail("only the free model set is allowed (unset PATHWISE_BENCH_MODELS).");

const allPlans = enumeratePlans(SPACE);
for (const p of allPlans) {
  const model = MODELS[p.model as Tier];
  if (!FREE_MODELS.has(model)) fail(`plan ${planKey(p)} uses ${model}, which is not in the free set (${[...FREE_MODELS].join(", ")}).`);
}
let plans: Plan[] = allPlans;
if (planIds) {
  const byId = new Map(allPlans.map((p) => [planKey(p), p]));
  plans = planIds.map((id) => {
    const p = byId.get(id);
    if (!p) fail(`unknown plan id "${id}". Known ids:\n  ${[...byId.keys()].join("\n  ")}`);
    return p;
  });
  if (new Set(planIds).size !== planIds.length) fail("duplicate plan id in --plans.");
}

const { existsSync } = await import("node:fs");
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  fail("the dataset is not downloaded (bench/real-rag/data/). Run `npm run bench:rag:pilot` once to download it; this tool does not.");
}
const { calibration, test } = sample(await loadHotpot(), num("calibration", 100), num("test", 200), num("seed", 7));
const questions = phase === "calibration" ? calibration : test;

interface Pair {
  q: Question;
  plan: Plan;
  tier: Tier;
}
const todo: Pair[] = [];
const estimate = new Map<string, { sum: number; n: number }>(); // per plan: mean list-price cost of its cached pairs
let cached = 0;
for (const plan of plans) {
  const tier = plan.model as Tier;
  const thinking = plan.thinking as "low" | "high";
  for (const q of questions) {
    const hit = await peekCache(tier, thinking, buildPrompt(q, plan.topK as number).prompt);
    if (hit) {
      cached++;
      const e = estimate.get(planKey(plan)) ?? { sum: 0, n: 0 };
      e.sum += hit.costUsd;
      e.n++;
      estimate.set(planKey(plan), e);
    } else todo.push({ q, plan, tier });
  }
}
const total = plans.length * questions.length;
const byModel = new Map<Tier, Pair[]>();
for (const t of todo) byModel.set(t.tier, [...(byModel.get(t.tier) ?? []), t]);

const rpm = num("rpm", Number(process.env.PATHWISE_BENCH_RPM ?? DEFAULT_RPM));
console.log(`== fill ${phase}: ${questions.length} questions x ${plans.length} plan(s), ${cached}/${total} cached, ${todo.length} missing`);
for (const [tier, pairs] of byModel) console.log(`  missing ${MODELS[tier]} (${tier}): ${pairs.length}`);
if (todo.length === 0) {
  console.log("nothing to fill: remaining 0.");
  process.exit(0);
}
let listUsd = 0;
let unestimated = 0;
for (const t of todo) {
  const e = estimate.get(planKey(t.plan));
  if (e && e.n > 0) listUsd += e.sum / e.n;
  else unestimated++;
}
console.log(
  `  list-price cost of filling (what a paid key would be billed; the free tier bills $0): about $${listUsd.toFixed(2)}, from each plan's own cached mean cost` +
    (unestimated ? `; ${unestimated} pairs belong to plans with no cached pair, so this is a LOWER bound` : ""),
);
console.log(`  wall time at ${rpm} requests/minute in total: about ${Math.ceil(todo.length / rpm)} min (daily quota permitting; the free tier may stop a model earlier).`);

if (!live) {
  console.log("\ndry run: no call was made. Add --yes-live to fill (needs GEMINI_API_KEY in your environment).");
  process.exit(0);
}

setRpm(rpm);
const remaining = new Map<Tier, number>([...byModel].map(([t, p]) => [t, p.length]));
const stopped = new Map<Tier, string>();
let interrupted = false;
let failed = false; // an unexpected error: exit 1 after the summary
let halt = false; // the budget ledger ceiling stops every model
let done = 0;
process.on("SIGINT", () => {
  if (interrupted) process.exit(130);
  interrupted = true;
  console.log("\nCtrl-C: finishing the calls in flight, then stopping. Everything finished is cached. (Ctrl-C again to quit now.)");
  stopAll();
});

async function worker(tier: Tier, pairs: Pair[]) {
  for (const { q, plan } of pairs) {
    if (interrupted || halt) return;
    try {
      await ragPipeline(q, plan);
    } catch (err) {
      if (err instanceof DailyQuotaError) stopped.set(tier, "daily quota used up");
      else if (err instanceof QuotaError) stopped.set(tier, "no quota for this key");
      else if (err instanceof ModelUnavailableError) stopped.set(tier, interrupted ? "interrupted" : "model kept failing or limiting (treated as the daily limit)");
      else if (err instanceof BudgetExceeded) {
        stopped.set(tier, "budget ledger ceiling reached");
        halt = true;
      }
      else {
        // Anything else (no key, HTTP 400, ...): stop every model, keep the cache, report, exit 1.
        stopped.set(tier, `error: ${(err as Error).message.slice(0, 200)}`);
        halt = true;
        failed = true;
      }
      return;
    }
    remaining.set(tier, remaining.get(tier)! - 1);
    done++;
    if (done % progressEvery === 0) {
      const s = spendSoFar();
      console.log(
        `  ${done}/${todo.length} done; remaining ${[...remaining].map(([t, n]) => `${MODELS[t]} ${n}`).join(", ")}; ledger $${s.spentUsd.toFixed(3)} of $${s.budgetUsd}`,
      );
    }
  }
}

console.log(`\nfilling (free tier, ${rpm} requests/minute in total, Ctrl-C is safe)...`);
await Promise.all([...byModel].map(([tier, pairs]) => worker(tier, pairs)));

const s = spendSoFar();
console.log(`\n== stopped after ${done} new calls this run (ledger list-price total $${s.spentUsd.toFixed(3)} of $${BUDGET_USD}; free tier bills $0)`);
for (const [tier, n] of remaining) {
  console.log(`  remaining ${MODELS[tier]}: ${n}${stopped.has(tier) ? `  (stopped: ${stopped.get(tier)})` : ""}`);
}
const left = [...remaining.values()].reduce((a, b) => a + b, 0);
if (left === 0) console.log("remaining 0: the table for this phase is complete.");
else if (halt && !failed) console.log("the ledger ceiling stopped the run: rerunning will stop again. Raise PATHWISE_BENCH_BUDGET only if you mean to.");
else if (!failed) console.log("rerun the SAME command after the daily reset (about 13:15 PKT per Arsalan). It resumes from the cache.");
process.exit(failed || halt ? 1 : 0);
