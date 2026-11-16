/**
 * Checks for the `pathwise` command line (SYNTHETIC logs written to a temp dir; no network, no key, no environment variable read
 * by the CLI). Runs the TypeScript source through tsx; the compiled bin is checked separately by the packaging test.
 *   npm run bench:cli
 */
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createPlanner } from "../src/index.js";

let failures = 0;
const check = (ok: boolean, msg: string, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}${!ok && detail ? `\n     ${detail.slice(0, 400)}` : ""}`);
  if (!ok) failures++;
};
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const dir = mkdtempSync(join(tmpdir(), "pathwise-cli-"));
const run = (args: string[]) => {
  const r = spawnSync("npx", ["tsx", "src/cli.ts", ...args], { encoding: "utf8", env: { PATH: process.env.PATH ?? "" } }); // no other environment is passed on
  return { code: r.status, out: r.stdout, err: r.stderr };
};
try {
  // SYNTHETIC full table: 3 plans, 120 questions, 0/1 quality with true means 0.6 / 0.85 / 0.95 and costs 1 / 2 / 5 (per 1000 runs)
  const r = rng(5);
  const means: Record<string, number> = { "model=small": 0.6, "model=mid": 0.85, "model=max": 0.95 };
  const cost: Record<string, number> = { "model=small": 0.001, "model=mid": 0.002, "model=max": 0.005 };
  const rows: string[] = [];
  const table: Record<string, Record<string, number>> = {};
  for (let q = 0; q < 120; q++)
    for (const p of Object.keys(means)) {
      const quality = r() < means[p] ? 1 : 0;
      (table[`q${q}`] ??= {})[p] = quality;
      rows.push(JSON.stringify({ question: `q${q}`, plan: p, quality, costUsd: cost[p] }));
    }
  const log = join(dir, "runs.jsonl");
  writeFileSync(log, rows.join("\n") + "\n\n"); // trailing blank lines are ignored

  const rec = run(["recommend", "--log", log, "--min-quality", "0.8", "--json"]);
  check(rec.code === 0, "recommend --json exits 0", rec.err);
  const j = JSON.parse(rec.out || "{}");
  // the library call on the same data
  const lib = createPlanner<string, number>({
    space: { plan: Object.keys(means) },
    pipeline: async (q, plan) => ({ output: table[q][String(plan.plan)], costUsd: cost[String(plan.plan)] }),
    scorer: (_q, o) => o,
    sla: { minQuality: 0.8 },
    concurrency: 1,
  });
  const libRec = await lib.recommend(Object.keys(table), {});
  check(!!libRec && j.plan === String(libRec.plan.plan) && Math.abs(j.recommendation.meanQuality - libRec.meanQuality) < 1e-12 && j.recommendation.lowerBound === libRec.lowerBound && j.recommendation.callsUsed === libRec.callsUsed && j.recommendation.certified === libRec.certified, "recommend result equals planner.recommend() on the same data");
  const txt = run(["recommend", "--log", log, "--min-quality", "0.8"]);
  check(txt.code === 0 && /Recommended plan: model=/.test(txt.out) && /not a certificate/.test(txt.out), "recommend text output names the plan and says it is not a certificate", txt.out);
  const none = run(["recommend", "--log", log, "--min-quality", "0.99"]);
  check(none.code === 0 && /No plan meets the bar/.test(none.out), "an unreachable bar prints 'No plan meets the bar' and exits 0", none.out + none.err);

  // certify: a log of fresh questions for one plan
  const fr = rng(99);
  const fresh: string[] = [];
  const freshQ: string[] = [];
  for (let q = 0; q < 150; q++) {
    freshQ.push(`f${q}`);
    fresh.push(JSON.stringify({ question: `f${q}`, plan: { model: "max" }, quality: fr() < 0.95 ? 1 : 0, costUsd: 0.005 }));
  }
  const flog = join(dir, "fresh.jsonl");
  writeFileSync(flog, fresh.join("\n"));
  const cert = run(["certify", "--log", flog, "--plan", "model=max", "--min-quality", "0.9", "--json"]);
  check(cert.code === 0, "certify --json exits 0 (a plan given as an object is keyed like planKey)", cert.err);
  const cj = JSON.parse(cert.out || "{}");
  const fl = createPlanner<string, number>({ space: { model: ["max"] }, pipeline: async () => ({ output: 1, costUsd: 0 }), scorer: (_q, o) => o, sla: { minQuality: 0.9 }, featurize: () => [1], feedbackRate: 0 });
  await fl.calibrate(["a"]);
  const fq = new Map(fresh.map((l) => [JSON.parse(l).question as string, JSON.parse(l).quality as number]));
  const fl2 = createPlanner<string, number>({ space: { model: ["max"] }, pipeline: async (q) => ({ output: fq.get(q)!, costUsd: 0.005 }), scorer: (_q, o) => o, sla: { minQuality: 0.9 }, featurize: () => [1], feedbackRate: 0, concurrency: 1 });
  fl2.importState(fl.exportState());
  const libCert = await fl2.certify(freshQ);
  check(cj.n === 150 && Math.abs(cj.meanQuality - libCert.meanQuality) < 1e-12 && cj.lowerBound === libCert.lowerBound && cj.certified === libCert.certified, "certify result equals planner.certify() on the same data");
  check(cj.warnings.some((w: string) => /ONCE per frozen policy/.test(w)) && cj.warnings.some((w: string) => /drift is not covered/.test(w)), "certify prints the use-once and drift warnings");
  const ctxt = run(["certify", "--log", flog, "--plan", "model=max", "--min-quality", "0.9"]);
  check(ctxt.code === 0 && /certified (yes|no)/.test(ctxt.out), "certify text output", ctxt.out);

  // errors: exit 2, a line number in every parse error, nothing evaluated
  const bad = (name: string, content: string, args: string[], re: RegExp) => {
    const f = join(dir, name);
    writeFileSync(f, content);
    const x = run(["recommend", "--log", f, "--min-quality", "0.5", ...args]);
    check(x.code === 2 && re.test(x.err), `bad input: ${name}`, `${x.code} ${x.err}`);
  };
  bad("badjson.jsonl", '{"question":"a","plan":"p","quality":1,"costUsd":0}\n{nope\n', [], /line 2: not valid JSON/);
  bad("quality.jsonl", '{"question":"a","plan":"p","quality":1.5,"costUsd":0}\n', [], /line 1: "quality" must be a number in \[0, 1\]/);
  bad("nan.jsonl", '{"question":"a","plan":"p","quality":"NaN","costUsd":0}\n', [], /line 1: "quality"/);
  bad("cost.jsonl", '{"question":"a","plan":"p","quality":1,"costUsd":-1}\n', [], /line 1: "costUsd" must be a finite number >= 0/);
  bad("noplan.jsonl", '{"question":"a","quality":1,"costUsd":0}\n', [], /line 1: "plan" must be/);
  bad("dup.jsonl", '{"question":"a","plan":"p","quality":1,"costUsd":0}\n{"question":"a","plan":"p","quality":0,"costUsd":0}\n', [], /line 2: question "a" and plan "p" already appear on line 1/);
  bad("partial.jsonl", '{"question":"a","plan":"p","quality":1,"costUsd":0}\n{"question":"a","plan":"q","quality":1,"costUsd":0}\n{"question":"b","plan":"p","quality":1,"costUsd":0}\n', [], /every plan on every question.*1 of 4 .*pairs are missing/);
  bad("empty.jsonl", "\n\n", [], /the log has no rows/);
  const missingFile = run(["recommend", "--log", join(dir, "nope.jsonl"), "--min-quality", "0.5"]);
  check(missingFile.code === 2 && /cannot read the log file/.test(missingFile.err), "a missing file exits 2");
  const args1 = run(["recommend", "--min-quality", "0.5"]);
  const args2 = run(["recommend", "--log", log, "--min-quality", "1.5"]);
  const args3 = run(["recommend", "--log", log, "--min-quality", "0.5", "--confidence", "1"]);
  const args4 = run(["certify", "--log", flog, "--min-quality", "0.5"]);
  const args5 = run(["certify", "--log", flog, "--plan", "nope", "--min-quality", "0.5"]);
  const args6 = run(["frobnicate"]);
  const args7 = run(["recommend", "--log", log, "--min-quality", "0.5", "--bogus"]);
  check([args1, args2, args3, args4, args5, args6, args7].every((x) => x.code === 2) && /--log/.test(args1.err) && /between 0 and 1/.test(args2.err) && /strictly between/.test(args3.err) && /--plan/.test(args4.err) && /no rows for plan/.test(args5.err) && /unknown command/.test(args6.err), "argument errors exit 2 with a clear message", [args1, args2, args3, args4, args5, args6, args7].map((x) => `${x.code} ${x.err}`).join(" | "));
  const help = run(["--help"]);
  check(help.code === 0 && /pathwise recommend/.test(help.out), "--help prints usage and exits 0");
} finally {
  rmSync(dir, { recursive: true, force: true });
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC logs)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
