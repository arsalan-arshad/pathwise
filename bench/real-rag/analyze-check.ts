/**
 * Checks for analyze.ts. Everything here is SYNTHETIC: a hand-made fixture and mock model answers (no network, no key,
 * temp cache). The end-to-end part needs the dataset downloaded and is skipped without it.
 *   npm run bench:rag:analyze:check
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chooseFixed, pairedBootstrap, questionsNeeded, verdict, type Table } from "./analyze-core.js";

let failures = 0;
function check(label: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${!ok && detail ? `\n     ${detail}` : ""}`);
}

// ---- SYNTHETIC fixture: 4 plans x 50 questions, plan "max" is best and dearest.
const plans = ["cheap", "mid", "good", "max"];
const make = (f1s: number[], costs: number[], n: number, shift = 0): Table => ({
  plans,
  f1: f1s.map((m, p) => Array.from({ length: n }, (_, i) => Math.min(1, Math.max(0, m + (((i * (p + 3)) % 7) - 3) * 0.01 + shift)))),
  cost: costs.map((c) => Array.from({ length: n }, () => c)),
});
const cal = make([0.5, 0.6, 0.77, 0.8], [1, 2, 3, 6], 50);
const test1 = make([0.5, 0.6, 0.7, 0.8], [1, 2, 3, 6], 50);
const test2 = make([0.9, 0.1, 0.2, 0.3], [9, 8, 7, 1], 50, 0.05); // wildly different test values
const c95 = chooseFixed(cal, "max", 0.95);
const c90 = chooseFixed(cal, "max", 0.9);
check("SLA is the floored fraction of always-max calibration F1", Math.abs(c95.sla - Math.floor(0.95 * c95.maxF1 * 1000) / 1000) < 1e-12);
check("95% picks the cheapest plan meeting the SLA ('good')", c95.plan === "good", JSON.stringify(c95));
check("90% picks 'good' too (mid 0.6 is below 0.72)", c90.plan === "good", JSON.stringify(c90));
// calibration-only: chooseFixed has no test input; prove the pipeline of choose-then-report ignores test values.
const report = (testTable: Table) => {
  const choice = chooseFixed(cal, "max", 0.95);
  const i = testTable.plans.indexOf(choice.plan);
  return { plan: choice.plan, testF1: testTable.f1[i].reduce((a, b) => a + b, 0) / 50 };
};
const r1 = report(test1);
const r2 = report(test2);
check("the choice is identical when the test values change (calibration only)", r1.plan === r2.plan);
check("...while the reported test F1 does change", Math.abs(r1.testF1 - r2.testF1) > 0.1);
const tie: Table = { plans: ["a", "b", "max"], f1: [[0.8, 0.8], [0.9, 0.9], [0.9, 0.9]], cost: [[1, 1], [1, 1], [5, 5]] };
check("a cost tie goes to the higher F1", chooseFixed(tie, "max", 0.9).plan === "b");

const a = { f1: Array.from({ length: 50 }, (_, i) => (i % 5) / 5), cost: Array.from({ length: 50 }, () => 1) };
const b = { f1: Array.from({ length: 50 }, (_, i) => ((i + 1) % 5) / 5), cost: Array.from({ length: 50 }, () => 2) };
const d1 = pairedBootstrap(a, b);
const d2 = pairedBootstrap(a, b);
check("the bootstrap is deterministic", JSON.stringify(d1) === JSON.stringify(d2));
check("a different seed gives a different interval", JSON.stringify(pairedBootstrap(a, b, 10000, 1).f1) !== JSON.stringify(d1.f1));
check("the cost interval is exactly -1 when every paired cost difference is -1", d1.cost.lo === -1 && d1.cost.hi === -1 && d1.n === 50);

const side = (name: string, meetsSla: boolean): { name: string; f1: number; cost: number; meetsSla: boolean } => ({ name, f1: 0.7, cost: 1, meetsSla });
const ctx = { sla: 0.7, calibration: 100 };
const iv = (lo: number, hi: number) => ({ mean: (lo + hi) / 2, lo, hi });
check("verdict: routing wins when both meet and cost interval is below 0", /ROUTING WINS/.test(verdict(side("R", true), side("F", true), { f1: iv(-0.01, 0.01), cost: iv(-0.5, -0.1), n: 200 }, ctx)));
check("verdict: fixed wins when both meet and cost interval is above 0", /FIXED PLAN WINS/.test(verdict(side("R", true), side("F", true), { f1: iv(-0.01, 0.01), cost: iv(0.1, 0.5), n: 200 }, ctx)));
check("verdict: no winner when intervals overlap", /NO WINNER/.test(verdict(side("R", true), side("F", true), { f1: iv(-0.01, 0.01), cost: iv(-0.1, 0.5), n: 200 }, ctx)));
check("verdict: unequal SLA outcomes are not called a win", /not comparable/.test(verdict(side("R", false), side("F", true), { f1: iv(-0.1, -0.01), cost: iv(-0.5, -0.1), n: 200 }, ctx)));
check("questionsNeeded matches the quick-start table (0.80 SLA, plan 5 points above: about 106)", Math.abs((questionsNeeded(0.85, 0.8) ?? 0) - 106) <= 1, String(questionsNeeded(0.85, 0.8)));
check("questionsNeeded is null when the plan is not above the SLA", questionsNeeded(0.8, 0.8) === null);

// ---- end to end on mock answers (SYNTHETIC)
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.log("skipped end-to-end checks: dataset not downloaded.");
} else {
  const dir = mkdtempSync(join(tmpdir(), "pathwise-analyze-"));
  const env = { ...process.env, PATHWISE_BENCH_MOCK: "1", PATHWISE_BENCH_CACHE_DIR: dir, GEMINI_API_KEY: "" };
  const run = (script: string, args: string[]) => {
    const r = spawnSync("npx", ["tsx", `bench/real-rag/${script}`, ...args], { encoding: "utf8", env });
    return { code: r.status, out: `${r.stdout}${r.stderr}` };
  };
  try {
    const empty = run("analyze.ts", []);
    check("empty cache: prints the incomplete-table message, no choice, exit 0", empty.code === 0 && /incomplete table: 1200 pairs missing, no choice made/.test(empty.out) && /SYNTHETIC/.test(empty.out), empty.out.slice(-300));
    check("empty cache: no plan is chosen", !/best fixed plan/.test(empty.out));
    run("fill.ts", ["--phase", "calibration", "--yes-live"]);
    const calOnly = run("analyze.ts", []);
    check("full calibration, no test pairs: chooses on calibration and says test not measured", calOnly.code === 0 && /best fixed plan \(chosen on calibration only\)/.test(calOnly.out) && /test not measured: fill needs 200 calls for plan/.test(calOnly.out), calOnly.out.slice(-600));
    check("...and prints the fill command", /bench:rag:fill -- --phase test --plans/.test(calOnly.out));
    check("...and never prints a verdict or test F1 without test pairs", !/NO WINNER|ROUTING WINS|FIXED PLAN WINS/.test(calOnly.out));
    const planIds = [...calOnly.out.matchAll(/^(model=\S+)\s+\d\.\d{3}\s+\$/gm)].map((m) => m[1]);
    for (let i = 0; i < planIds.length; i += 4) run("fill.ts", ["--phase", "test", "--plans", planIds.slice(i, i + 4).join(","), "--yes-live"]);
    const full = run("analyze.ts", []);
    const again = run("analyze.ts", []);
    check("full tables: reports test F1 for the chosen plan at both SLAs", (full.out.match(/test \(200 q\): F1/g) ?? []).length === 2, full.out.slice(-800));
    check("full tables: Pathwise at 90% is 'not measured'", /Pathwise at 90%: not measured/.test(full.out));
    check("full tables: a verdict is printed for both Pathwise strategies", (full.out.match(/ROUTING WINS|FIXED PLAN WINS|NO WINNER|not comparable/g) ?? []).length === 2, full.out.slice(-1200));
    check("full tables: the output is deterministic across runs", full.out === again.out);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC fixture and mock answers)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
