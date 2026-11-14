/**
 * Checks for fill.ts, mock mode only (no network, no key, temp cache). Needs the dataset downloaded; skips otherwise.
 *   npm run bench:rag:fill:check
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.log("skipped: dataset not downloaded.");
  process.exit(0);
}
const dir = mkdtempSync(join(tmpdir(), "pathwise-fill-"));
let failures = 0;
function run(label: string, args: string[], env: Record<string, string> = {}) {
  const r = spawnSync("npx", ["tsx", "bench/real-rag/fill.ts", ...args], {
    encoding: "utf8",
    env: { ...process.env, PATHWISE_BENCH_MOCK: "1", PATHWISE_BENCH_CACHE_DIR: dir, GEMINI_API_KEY: "", ...env },
  });
  return { label, code: r.status, out: `${r.stdout}${r.stderr}` };
}
function expect(r: { label: string; code: number | null; out: string }, code: number, ...patterns: RegExp[]) {
  const ok = r.code === code && patterns.every((p) => p.test(r.out));
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${r.label}`);
  if (!ok) console.log(`     exit ${r.code} (wanted ${code}); output:\n${r.out.split("\n").map((l) => "     " + l).join("\n")}`);
}
try {
  const t = "model=small|thinking=low|topK=2";
  expect(run("test phase without --plans is refused", ["--phase", "test"]), 2, /--phase test needs --plans/);
  expect(run("more than 4 plans is refused", ["--phase", "test", "--plans", "a,b,c,d,e"]), 2, /at most 4 plans/);
  expect(run("unknown plan id is refused", ["--phase", "test", "--plans", "model=huge|thinking=low|topK=2"]), 2, /unknown plan id/);
  expect(run("--plans with calibration is refused", ["--phase", "calibration", "--plans", t]), 2, /test phase only|--phase test only/);
  expect(run("--yes-live with offline is refused", ["--phase", "calibration", "--yes-live"], { PATHWISE_BENCH_OFFLINE: "1" }), 2, /cannot be combined with PATHWISE_BENCH_OFFLINE/);
  expect(run("paid model set is refused", ["--phase", "calibration"], { PATHWISE_BENCH_MODELS: "paid" }), 2, /only the free model set/);
  expect(run("a model outside the free set is refused", ["--phase", "calibration"], { PATHWISE_BENCH_SMALL_MODEL: "gemini-3.8-flash" }), 2, /not in the free set/);
  expect(run("dry run reports 1200 missing and makes no call", ["--phase", "calibration"]), 0, /0\/1200 cached, 1200 missing/, /dry run: no call was made/);
  expect(
    run("daily quota stops each model cleanly and exits 0", ["--phase", "calibration", "--yes-live", "--progress", "100"], { PATHWISE_BENCH_MOCK_DAILY_QUOTA: "50" }),
    0,
    /remaining gemini-3.1-flash-lite: 550\s+\(stopped: daily quota used up\)/,
    /remaining gemini-3.5-flash-lite: 550\s+\(stopped: daily quota used up\)/,
    /rerun the SAME command/,
  );
  expect(run("rerun resumes: 100 cached, 1100 missing", ["--phase", "calibration"]), 0, /100\/1200 cached, 1100 missing/);
  expect(run("rerun without the quota completes", ["--phase", "calibration", "--yes-live"]), 0, /remaining 0: the table for this phase is complete/);
  expect(run("a finished table has nothing to fill", ["--phase", "calibration", "--yes-live"]), 0, /1200\/1200 cached, 0 missing/, /nothing to fill: remaining 0/);
  expect(run("test phase fills only the named plan", ["--phase", "test", "--plans", t, "--yes-live"]), 0, /200 questions x 1 plan/, /remaining 0/);
  expect(run("a second test plan is a separate fill", ["--phase", "test", "--plans", `${t},model=medium|thinking=low|topK=2`]), 0, /2 plan\(s\), 200\/400 cached, 200 missing/);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
console.log(failures === 0 ? "\nall checks passed" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
