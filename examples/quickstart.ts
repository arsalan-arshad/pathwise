/**
 * Quick start (docs/quickstart.md embeds this file). A simulated pipeline (SYNTHETIC): no model, no network, no key.
 * With the published package the import is: import { createPlanner } from "@arsalandev/pathwise";
 */
import { createPlanner } from "../src/index.js";

// 1. Your questions. Here: short ones are easy, long ones are hard.
const questions = Array.from({ length: 300 }, (_, i) => "word ".repeat(2 + ((i * 7) % 30)).trim());
const difficulty = (q: string) => Math.min(1, q.split(" ").length / 32);

const planner = createPlanner<string, { correct: boolean }>({
  // 2. The knobs: everything that trades cost for quality in your pipeline.
  space: { model: ["small", "large"], topK: [2, 5] },

  // 3. Your existing pipeline, parameterised by a plan. Return the output and what the call cost.
  pipeline: async (q, plan) => {
    const skill = (plan.model === "large" ? 0.9 : 0.5) + (plan.topK === 5 ? 0.08 : 0);
    const costUsd = (plan.model === "large" ? 0.01 : 0.001) + (plan.topK as number) * 0.0002;
    return { output: { correct: skill >= difficulty(q) }, costUsd };
  },

  // 4. Score an output from 0 to 1: an eval-set check or an LLM judge.
  scorer: (_q, out) => (out.correct ? 1 : 0),

  // 5. The service level. `confidence` turns "on average" into a bound you can check, and `fallbackPlan` is what runs
  //    if calibration cannot certify it (use the plan you trust today).
  sla: { minQuality: 0.8, confidence: 0.9, fallbackPlan: { model: "large", topK: 5 } },
  feedbackRate: 0, // 0.1 in production: score 10% of live runs and keep learning
});

// 6. Calibrate on a sample of real questions, then read the report.
const report = await planner.calibrate(questions.slice(0, 120));
console.log(`certified: ${report.slaCertified} (lower bound ${report.certifiedQuality.toFixed(3)} vs SLA 0.8)`);
console.log(`policy: ${report.policy}${report.fixedPlan ? `, plan for every request: ${report.fixedPlan}` : ""}`);
console.log(`calibration: ${report.runs} runs, $${report.costUsd.toFixed(3)}`);

// 7. Serve traffic: each request gets the cheapest plan predicted to meet the SLA.
let correct = 0;
let cost = 0;
for (const q of questions.slice(120)) {
  const r = await planner.run(q);
  if (r.output.correct) correct++;
  cost += r.costUsd;
}
const n = questions.length - 120;
console.log(`served ${n}: accuracy ${((correct / n) * 100).toFixed(1)}%, $${cost.toFixed(3)} (always-large/topK5 would be $${(n * 0.011).toFixed(3)})`);
console.log("plan mix:", planner.report().planUsage);
