/**
 * Test for sla.fallbackPlan (SYNTHETIC, labeled as such): when the SLA cannot be certified, every request
 * must get exactly the declared plan, so accuracy and cost equal running that plan directly.
 *
 * Usage: npx tsx examples/fallback-bench.ts [seeds=5]
 */
import { createPlanner, hashedTextFeaturizer, planKey, type Plan } from "../src/index.js";
import { createWorkload, ragPipeline, SPACE } from "./workload.js";

const SEEDS = Number(process.argv[2] ?? 5);
const fallbackPlan: Plan = { model: "medium", topK: 5, rerank: false };
const feats = hashedTextFeaturizer();
let failures = 0;
const check = (ok: boolean, msg: string) => {
  if (!ok) {
    failures++;
    console.log("FAIL:", msg);
  }
};

for (let seed = 1; seed <= SEEDS; seed++) {
  const { makeQueries } = createWorkload(seed, { jitter: 0.25 });
  const mk = (sla: object) =>
    createPlanner<{ text: string; difficulty: number }, { correct: boolean }>({
      space: SPACE,
      pipeline: ragPipeline,
      scorer: (_q, o) => (o.correct ? 1 : 0),
      sla: sla as never,
      featurize: (q) => feats(q.text),
      feedbackRate: 0,
    });
  // 30 samples and a 99% bar: far too little evidence to certify.
  const planner = mk({ minQuality: 0.99, confidence: 0.9, fallbackPlan });
  const report = await planner.calibrate(makeQueries(30));
  check(!report.slaCertified, `seed ${seed}: expected uncertified`);
  check(report.policy === "safe-plan" && report.fixedPlan === planKey(fallbackPlan), `seed ${seed}: report ${report.policy} ${report.fixedPlan}`);

  const prod = makeQueries(300);
  let ok = 0, cost = 0, refOk = 0, refCost = 0;
  for (const q of prod) {
    const r = await planner.run(q);
    check(planKey(r.plan) === planKey(fallbackPlan), `seed ${seed}: ran ${planKey(r.plan)}`);
    ok += r.output.correct ? 1 : 0;
    cost += r.costUsd;
    const ref = await ragPipeline(q, fallbackPlan);
    refOk += ref.output.correct ? 1 : 0;
    refCost += ref.costUsd;
  }
  check(ok === refOk && Math.abs(cost - refCost) < 1e-9, `seed ${seed}: accuracy/cost differ from the fallback plan`);
  console.log(`seed ${seed}: uncertified -> fallback ${planKey(fallbackPlan)}: accuracy ${(ok / prod.length * 100).toFixed(1)}% = ${(refOk / prod.length * 100).toFixed(1)}%, cost $${cost.toFixed(2)} = $${refCost.toFixed(2)} (synthetic)`);
}

// Certified case: policy is "routing" and there is no fixedPlan.
{
  const { makeQueries } = createWorkload(1);
  const planner = createPlanner<{ text: string; difficulty: number }, { correct: boolean }>({
    space: SPACE, pipeline: ragPipeline, scorer: (_q, o) => (o.correct ? 1 : 0),
    sla: { minQuality: 0.7, confidence: 0.9, fallbackPlan }, featurize: (q) => feats(q.text), feedbackRate: 0,
  });
  const r = await planner.calibrate(makeQueries(300));
  check(r.slaCertified && r.policy === "routing" && r.fixedPlan === undefined, `certified case: ${r.slaCertified} ${r.policy} ${r.fixedPlan}`);
}

// Validation.
const bad = (sla: object) => { try { createPlanner({ space: SPACE, pipeline: ragPipeline, scorer: () => 1, sla: sla as never }); return false; } catch { return true; } };
check(bad({ minQuality: 0.9, confidence: 0.9, fallbackPlan: { model: "huge", topK: 5, rerank: false } }), "plan outside the space must throw");
check(bad({ minQuality: 0.9, fallbackPlan }), "fallbackPlan without confidence must throw");
check(bad({ minQuality: 0.9, confidence: 0.9, onUncertified: "best-effort", fallbackPlan }), "fallbackPlan with best-effort must throw");

console.log(failures === 0 ? "\nAll fallbackPlan checks passed." : `\n${failures} check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
