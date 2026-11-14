/**
 * Checks for embed-compare.ts. SYNTHETIC: mock model answers, fake embedders, and a fake OpenAI-compatible embeddings server on
 * localhost (no Ollama, no network, no key, temp cache). Needs the dataset downloaded; skips otherwise.
 *   npm run bench:rag:embed-compare:check
 */
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.log("skipped: dataset not downloaded.");
  process.exit(0);
}
let failures = 0;
function check(label: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${!ok && detail ? `\n${detail.split("\n").map((l) => "     " + l).join("\n")}` : ""}`);
}
const dir = mkdtempSync(join(tmpdir(), "pathwise-embed-"));
const env = { ...process.env, PATHWISE_BENCH_MOCK: "1", PATHWISE_BENCH_CACHE_DIR: dir, GEMINI_API_KEY: "" };
const run = (script: string, args: string[]) => {
  const r = spawnSync("npx", ["tsx", `bench/real-rag/${script}`, ...args], { encoding: "utf8", env });
  return { code: r.status, out: `${r.stdout}${r.stderr}` };
};
// The fake server lives in this process, so calls that talk to it must not block the event loop (spawnSync would).
const runAsync = (script: string, args: string[]) =>
  new Promise<{ code: number | null; out: string }>((resolve) => {
    const { PATHWISE_BENCH_MOCK: _mock, ...realEnv } = env; // real mode refuses MOCK; the temp cache dir already holds the mock answers
    const child = spawn("npx", ["tsx", `bench/real-rag/${script}`, ...args], { env: realEnv });
    let out = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (out += d));
    child.on("close", (code) => resolve({ code, out }));
  });
// A fake Ollama-like server: deterministic vectors from the text, `usage` reported or not, and a model that "is not pulled".
const server = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const { model, input } = JSON.parse(body || "{}") as { model: string; input: string[] };
    if (model !== "fake-embed") {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: `model "${model}" not found, try pulling it first` } }));
      return;
    }
    const data = input.map((t, index) => {
      let h = 2166136261;
      for (let i = 0; i < t.length; i++) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
      return { object: "embedding", index, embedding: Array.from({ length: 8 }, (_, k) => Math.sin(h * (k + 1))) };
    });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ object: "list", data, model, usage: { prompt_tokens: input.length * 7, total_tokens: input.length * 7 } }));
  });
});
await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
const port = (server.address() as { port: number }).port;
const url = `http://127.0.0.1:${port}/v1`;
try {
  const empty = run("embed-compare.ts", ["--fake"]);
  check("incomplete calibration table: prints the missing count, exit 0, nothing measured", empty.code === 0 && /incomplete table: 1200 calibration pairs missing, nothing measured/.test(empty.out) && /SYNTHETIC/.test(empty.out), empty.out.slice(-400));
  check("...and shows no results table", !/cross-validation \(/.test(empty.out));

  run("fill.ts", ["--phase", "calibration", "--yes-live"]);
  const fake = run("embed-compare.ts", ["--fake"]);
  check("--fake: completes, every number labelled SYNTHETIC", fake.code === 0 && /SYNTHETIC: --fake/.test(fake.out) && /cross-validation[^\n]*SYNTHETIC/.test(fake.out), fake.out.slice(-1500));
  check("--fake: three featurizers are compared (hashed, oracle, noise)", /hashed text/.test(fake.out) && /fake oracle/.test(fake.out) && /fake noise/.test(fake.out));
  check("--fake: reports the noise control against hashed", /control: the noise embedder/.test(fake.out), fake.out.slice(-1500));
  console.log(`     (noise control line: ${(fake.out.match(/control: the noise embedder[^\n]*/) ?? ["-"])[0]})`);
  check("--fake: test is 'not measured' while the test table is incomplete, no partial numbers", /test: not measured\. 2400 of 2400 test pairs are missing/.test(fake.out));
  check("--fake: embedding calls, tokens, dimension and wall time are printed", /embed calls, \d+ texts, \d+ tokens/.test(fake.out) && /dimension \d+/.test(fake.out) && /ms to embed 300 question texts/.test(fake.out));
  check("--fake: no Gemini call", /no Gemini call was made/.test(fake.out));
  const again = run("embed-compare.ts", ["--fake"]);
  check("--fake: output is deterministic apart from timings", fake.out.replace(/\d+ ms/g, "") === again.out.replace(/\d+ ms/g, ""));

  const real = await runAsync("embed-compare.ts", ["--model", "fake-embed", "--base-url", url]);
  check("real path against the fake local server: completes, reports tokens from usage, no SYNTHETIC banner", real.code === 0 && /fake-embed \(Ollama\): \d+ embed calls, 300 texts, 2100 tokens reported by the server, dimension 8/.test(real.out) && !/SYNTHETIC/.test(real.out), real.out.slice(-1200));
  const refuse = run("embed-compare.ts", ["--model", "fake-embed", "--base-url", url]);
  check("a real run refuses PATHWISE_BENCH_MOCK=1", refuse.code === 2 && /refused: PATHWISE_BENCH_MOCK=1/.test(refuse.out), refuse.out.slice(-300));
  const missing = await runAsync("embed-compare.ts", ["--model", "not-pulled", "--base-url", url]);
  check("model not pulled: exit 1, prints the exact ollama pull command, pulls nothing", missing.code === 1 && /ollama pull not-pulled/.test(missing.out), missing.out.slice(-500));
  const down = await runAsync("embed-compare.ts", ["--model", "fake-embed", "--base-url", "http://127.0.0.1:1/v1"]);
  check("server not running: exit 1 with a start-Ollama hint and the pull command", down.code === 1 && /ollama pull fake-embed/.test(down.out) && /Start Ollama/.test(down.out), down.out.slice(-500));

  // Full test table (all 12 plans) -> test numbers appear.
  const ids = ["small", "medium"].flatMap((m) => ["low", "high"].flatMap((t) => [2, 5, 10].map((k) => `model=${m}|thinking=${t}|topK=${k}`)));
  for (let i = 0; i < ids.length; i += 4) run("fill.ts", ["--phase", "test", "--plans", ids.slice(i, i + 4).join(","), "--yes-live"]);
  const full = run("embed-compare.ts", ["--fake"]);
  check("--fake with a complete test table: test rows for every featurizer", !/test: not measured/.test(full.out) && ((full.out.split("-- test set")[1] ?? "").split("-- embedding cost")[0]).match(/hashed text|fake oracle|fake noise/g)?.length === 3, full.out.slice(-1500));
} finally {
  server.close();
  rmSync(dir, { recursive: true, force: true });
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC: mock answers, fake embedders, fake server)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
