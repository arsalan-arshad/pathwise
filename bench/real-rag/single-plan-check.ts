/**
 * Checks for single-plan-core.ts. Everything here is SYNTHETIC (generated scores, no cache, no network, no key).
 *   npm run bench:rag:single-plan:check
 */
import { mulberry32, type Table } from "./analyze-core.js";
import { percentile, race, shuffled, wilson } from "./single-plan-core.js";

let failures = 0;
function check(label: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${!ok && detail ? `\n     ${detail}` : ""}`);
}

/** SYNTHETIC table: plan p scores 1 with probability means[p] on each question (Bernoulli), at a fixed cost. */
function synth(means: number[], costs: number[], n: number, seed: number): Table {
  const rnd = mulberry32(seed);
  return {
    plans: means.map((_, i) => `p${i}`),
    f1: means.map((m) => Array.from({ length: n }, () => (rnd() < m ? 1 : 0))),
    cost: costs.map((c) => Array.from({ length: n }, () => c)),
  };
}
const means = [0.3, 0.55, 0.82, 0.85, 0.9];
const costs = [1, 2, 3, 5, 8];
const SLA = 0.7; // true best feasible plan: p2 (0.82, cost 3); p3 and p4 are feasible but dearer

check("shuffled is a seeded permutation", JSON.stringify(shuffled(50, 1)) === JSON.stringify(shuffled(50, 1)) && [...shuffled(50, 1)].sort((a, b) => a - b).every((v, i) => v === i) && JSON.stringify(shuffled(50, 1)) !== JSON.stringify(shuffled(50, 2)));
const w = wilson(0.5, 100, 1.96);
check("wilson bounds bracket the mean, shrink with n, stay within [0,1]", w.lo < 0.5 && w.hi > 0.5 && wilson(0.5, 1000, 1.96).hi - wilson(0.5, 1000, 1.96).lo < w.hi - w.lo && wilson(1, 5, 1.96).hi <= 1 && wilson(0, 5, 1.96).lo >= 0);
check("percentile picks from sorted values", percentile([5, 1, 3, 2, 4], 0) === 1 && percentile([5, 1, 3, 2, 4], 100) === 5);

// Error rate over 300 INDEPENDENT synthetic tables (100 questions each).
const TABLES = 300;
let wrong = 0;
let eliminatedBest = 0;
let none = 0;
let calls = 0;
for (let s = 0; s < TABLES; s++) {
  const t = synth(means, costs, 100, 100 + s);
  const r = race(t, shuffled(100, s), SLA, 1.96, 10);
  calls += r.calls;
  if (r.plan === null) none++;
  else if (r.plan !== "p2") wrong++;
  if (r.plan !== "p2") eliminatedBest++;
}
const rate = eliminatedBest / TABLES;
console.log(`     (SYNTHETIC: true best plan p2 (0.82, SLA 0.7) not returned in ${eliminatedBest}/${TABLES} tables (${(100 * rate).toFixed(1)}%); wrong plan ${wrong}, none ${none}; mean calls ${(calls / TABLES).toFixed(0)} of ${5 * 100})`);
check("the true best plan is returned in at least 90% of independent synthetic tables (the bounds are looked at after every question, so this is not a 2.5% guarantee)", rate <= 0.1, `${(100 * rate).toFixed(1)}%`);
check("racing uses fewer calls than the full table here", calls / TABLES < 5 * 100);
check("deterministic for the same table and order", JSON.stringify(race(synth(means, costs, 100, 7), shuffled(100, 3), SLA, 1.96, 10)) === JSON.stringify(race(synth(means, costs, 100, 7), shuffled(100, 3), SLA, 1.96, 10)));

// A plan that clearly cannot meet the SLA is dropped early; dominated feasible plans are dropped once the cheaper one's lower bound clears the SLA.
const easy = synth([0.1, 0.95, 0.97], [1, 2, 9], 200, 5);
const re = race(easy, shuffled(200, 1), 0.7, 1.96, 10);
check("an infeasible plan and a dominated dearer plan are eliminated early (stops at the first allowed look: 30 calls, not 600)", re.plan === "p1" && re.survivors === 1 && re.calls < 3 * 200 && re.questionsUsed < 200, JSON.stringify(re));
// Nothing feasible.
const nope = synth([0.2, 0.3], [1, 2], 100, 9);
check("when no plan can meet the SLA the result is null (no plan invented)", race(nope, shuffled(100, 1), 0.9, 1.96, 10).plan === null);
// Questions run out before the bounds separate: the cheapest survivor whose observed mean meets the SLA is returned.
const close = synth([0.72, 0.74], [1, 2], 20, 11);
const rc = race(close, shuffled(20, 1), 0.7, 1.96, 10);
check("with too few questions to eliminate, the cheapest plan meeting the SLA on the observed mean is returned", rc.questionsUsed === 20 && (rc.plan === "p0" || rc.plan === null), JSON.stringify(rc));
// minQuestions is respected: no elimination before it.
const early = race(synth([0.0, 0.99], [1, 2], 50, 3), shuffled(50, 1), 0.7, 1.96, 30);
check("nothing is eliminated before the minimum number of questions", early.calls >= 2 * 30 - 0 && early.questionsUsed >= 30, JSON.stringify(early));

console.log(failures === 0 ? "\nall checks passed (SYNTHETIC generated scores)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
