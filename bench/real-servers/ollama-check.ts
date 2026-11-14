/**
 * Checks for ollama.ts against FAKE local servers (SYNTHETIC: no Ollama, no model, no network beyond localhost).
 * They prove the script's own logic: SKIP path, table, classification of server behaviours. They say nothing about real Ollama.
 *   npm run smoke:ollama:check
 */
import { spawn } from "node:child_process";
import { createServer } from "node:http";

let failures = 0;
function check(label: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${!ok && detail ? `\n${detail.split("\n").map((l) => "     " + l).join("\n")}` : ""}`);
}
const cfg = { honorDimensions: true, usage: true, rejectReasoning: false, embedUsage: true };
const vec = (t: string, n: number) => {
  let h = 2166136261;
  for (let i = 0; i < t.length; i++) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
  return Array.from({ length: n }, (_, k) => Math.sin(h * (k + 1)));
};
const server = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const send = (code: number, o: unknown) => (res.writeHead(code, { "content-type": "application/json" }), res.end(JSON.stringify(o)));
    if (req.method === "GET" && req.url === "/v1/models") return send(200, { data: [] });
    const b = JSON.parse(body || "{}") as { input?: string[]; dimensions?: number; reasoning_effort?: string; messages?: { content: string }[] };
    if (req.url === "/v1/embeddings") {
      const n = cfg.honorDimensions && b.dimensions ? b.dimensions : 8;
      return send(200, { data: (b.input ?? []).map((t, index) => ({ index, embedding: vec(t, n) })), ...(cfg.embedUsage ? { usage: { prompt_tokens: 5, total_tokens: 5 } } : {}) });
    }
    if (req.url === "/v1/chat/completions") {
      if (cfg.rejectReasoning && b.reasoning_effort) return send(400, { error: { message: "reasoning_effort is not supported by this model" } });
      const slow = /long story/.test(b.messages?.[0]?.content ?? "");
      return setTimeout(() => send(200, { choices: [{ message: { content: "pong" } }], ...(cfg.usage ? { usage: { prompt_tokens: 9, completion_tokens: 2, total_tokens: 11 } } : {}) }), slow ? 800 : 0);
    }
    send(404, { error: { message: "not found" } });
  });
});
await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`;
const run = (env: Record<string, string>) =>
  new Promise<{ code: number | null; out: string }>((resolve) => {
    const child = spawn("npx", ["tsx", "bench/real-servers/ollama.ts"], { env: { ...process.env, ...env } });
    let out = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (out += d));
    child.on("close", (code) => resolve({ code, out }));
  });
try {
  const skip = await run({ OLLAMA_BASE_URL: "http://127.0.0.1:1/v1" });
  check("no server: prints SKIP with the reason and exits 0", skip.code === 0 && /^SKIP: no OpenAI-compatible server reachable/m.test(skip.out), skip.out.slice(-300));

  const emb = await run({ OLLAMA_BASE_URL: url, OLLAMA_EMBED_MODEL: "fake-embed", OLLAMA_CHAT_MODEL: "" });
  check("embeddings only: five embeddings rows, dimension 8, order and dimension check pass, exit 0", emb.code === 0 && /5 vectors, dimension 8/.test(emb.out) && /each batched vector equals its single-text vector/.test(emb.out) && /accepts 8, rejects 9/.test(emb.out), emb.out.slice(-1500));
  check("embeddings only: `dimensions` honored and usage reported are classified", /honored: asked 4, got 4/.test(emb.out) && /tokens 5/.test(emb.out), emb.out.slice(-1500));
  check("embeddings only: chat checks are reported as SKIPPED, not run", /SKIPPED: set OLLAMA_CHAT_MODEL/.test(emb.out) && !/chat: normal completion/.test(emb.out));
  check("a table with a 'difference' column is printed", /difference from the docs \/ README/.test(emb.out) && /No README claim was edited/.test(emb.out));

  cfg.honorDimensions = false;
  cfg.embedUsage = false;
  const odd = await run({ OLLAMA_BASE_URL: url, OLLAMA_EMBED_MODEL: "fake-embed" });
  check("a server that ignores `dimensions` is flagged IGNORED", /IGNORED: asked 4, got 8/.test(odd.out), odd.out.slice(-1200));
  check("a server without embedding usage is reported as such", /no usage in the response/.test(odd.out));

  cfg.honorDimensions = true;
  cfg.embedUsage = true;
  const chat = await run({ OLLAMA_BASE_URL: url, OLLAMA_EMBED_MODEL: "fake-embed", OLLAMA_CHAT_MODEL: "fake-chat" });
  check("with a chat model: completion, usage, cost callback, reasoning_effort accepted, abort, timeout", chat.code === 0 && /answer "pong"/.test(chat.out) && /usage present/.test(chat.out) && /cost callback used \(0\.5\)/.test(chat.out) && /accepted \(no error\)/.test(chat.out) && /AbortError after/.test(chat.out) && /timeout error/.test(chat.out), chat.out.slice(-2500));

  cfg.rejectReasoning = true;
  cfg.usage = false;
  const strict = await run({ OLLAMA_BASE_URL: url, OLLAMA_EMBED_MODEL: "fake-embed", OLLAMA_CHAT_MODEL: "fake-chat" });
  check("reasoning_effort rejection is reported as REJECTED (not an unexpected error)", /REJECTED: .*reasoning_effort/.test(strict.out), strict.out.slice(-2500));
  check("missing chat usage: the pipeline fails loudly (an ERROR row) and the exit code says so", /usage absent/.test(strict.out) && /ERROR: .*usage/i.test(strict.out) && strict.code === 1, strict.out.slice(-2500));
} finally {
  server.close();
}
console.log(failures === 0 ? "\nall checks passed (fake servers only; nothing here describes real Ollama)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
