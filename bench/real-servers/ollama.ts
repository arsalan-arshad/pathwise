/**
 * Real-server smoke test: pathwise/embeddings and pathwise/openai-compatible against a REAL local Ollama server.
 *
 *   npm run smoke:ollama
 *
 * Environment (all optional):
 *   OLLAMA_BASE_URL     default http://localhost:11434/v1
 *   OLLAMA_EMBED_MODEL  default nomic-embed-text (the one model approved for download; this script never pulls anything)
 *   OLLAMA_CHAT_MODEL   no default. Without it the chat checks are skipped: a chat model is a second download.
 *
 * If the server is not reachable it prints SKIP with the reason and exits 0. It prints a table: check, result, and how that
 * differs from what the adapter docs / README say ("matches" or the difference). It edits no docs; read the output first.
 * Round-trip times describe the machine it ran on only. Written without ever running against a real server: the
 * script's own logic is exercised by `npm run smoke:ollama:check` against a fake local server.
 */
import { embeddingFeaturizer, openAIEmbed } from "../../src/adapters/embeddings.js";
import { openAICompatiblePipeline } from "../../src/adapters/openai-compatible.js";

const BASE = (process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1").replace(/\/+$/, "");
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text";
const CHAT_MODEL = process.env.OLLAMA_CHAT_MODEL || undefined;

interface Row {
  check: string;
  result: string;
  diff: string;
}
const rows: Row[] = [];
const ms = (t0: number) => `${Math.round(performance.now() - t0)} ms`;
const msg = (e: unknown) => (e instanceof Error ? e.message : String(e)).slice(0, 160).replace(/\s+/g, " ");
async function check(name: string, fn: () => Promise<{ result: string; diff?: string }>) {
  const t0 = performance.now();
  try {
    const r = await fn();
    rows.push({ check: name, result: `${r.result} (${ms(t0)})`, diff: r.diff ?? "matches" });
  } catch (e) {
    rows.push({ check: name, result: `ERROR: ${msg(e)} (${ms(t0)})`, diff: "unexpected: investigate" });
  }
}

// ---- reachability: SKIP, never fail, when there is no server
try {
  const res = await fetch(`${BASE}/models`, { signal: AbortSignal.timeout(3000) });
  if (!res.ok) throw new Error(`GET ${BASE}/models answered HTTP ${res.status}`);
} catch (e) {
  console.log(`SKIP: no OpenAI-compatible server reachable at ${BASE} (${msg(e)}). Start Ollama and run this again. Nothing was tested.`);
  process.exit(0);
}
console.log(`server: ${BASE}\nembeddings model: ${EMBED_MODEL}\nchat model: ${CHAT_MODEL ?? "(none: chat checks skipped)"}\n`);

// ---- embeddings
const texts = ["What is the capital of France?", "Who wrote Hamlet?", "How many legs does a spider have?", "Name a prime number above 100.", "Where is the Eiffel Tower?"];
const embed = openAIEmbed({ model: EMBED_MODEL, baseUrl: BASE });
const cosine = (a: number[], b: number[]) => {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) (d += a[i] * b[i]), (na += a[i] * a[i]), (nb += b[i] * b[i]);
  return d / (Math.sqrt(na) * Math.sqrt(nb));
};
let batch: number[][] = [];
let dim = 0;

await check("embeddings: a batch of 5 texts returns 5 vectors", async () => {
  const r = await embed(texts);
  batch = r.vectors;
  dim = batch[0]?.length ?? 0;
  if (batch.length !== texts.length) throw new Error(`got ${batch.length} vectors`);
  return { result: `5 vectors, dimension ${dim}` };
});
await check("embeddings: vectors are in input order", async () => {
  if (batch.length !== texts.length) throw new Error("no batch to compare (see the check above)");
  const worst: number[] = [];
  for (let i = 0; i < texts.length; i++) worst.push(cosine(batch[i], (await embed([texts[i]])).vectors[0]));
  const min = Math.min(...worst);
  if (min < 0.99) throw new Error(`a batched vector differs from the same text embedded alone (min cosine ${min.toFixed(4)})`);
  return { result: `each batched vector equals its single-text vector (min cosine ${min.toFixed(4)})` };
});
await check("embeddings: dimension check in embeddingFeaturizer", async () => {
  if (!dim) throw new Error("no dimension known");
  const ok = embeddingFeaturizer<string>({ embed, dimension: dim });
  await ok.warm(["dimension probe"]);
  const bad = embeddingFeaturizer<string>({ embed, dimension: dim + 1 });
  const rejected = await bad.warm(["dimension probe"]).then(() => false, (e: Error) => /dimension/i.test(e.message));
  if (!rejected) throw new Error("a wrong expected dimension was NOT rejected");
  return { result: `accepts ${dim}, rejects ${dim + 1}` };
});
await check("embeddings: `dimensions` option", async () => {
  const want = Math.max(2, Math.floor(dim / 2) || 64);
  try {
    const r = await openAIEmbed({ model: EMBED_MODEL, baseUrl: BASE, dimensions: want })(["dimensions probe"]);
    const got = r.vectors[0].length;
    return got === want
      ? { result: `honored: asked ${want}, got ${got}` }
      : { result: `IGNORED: asked ${want}, got ${got}`, diff: "the adapter docs say `dimensions` is only for text-embedding-3+ models; here the server ignores it (silently): do not rely on it with Ollama" };
  } catch (e) {
    return { result: `REJECTED: ${msg(e)}`, diff: "the server rejects `dimensions`; the adapter passes it through unchanged and surfaces the error" };
  }
});
await check("embeddings: `usage` in the response", async () => {
  const r = await embed(["usage probe"]);
  const raw = await fetch(`${BASE}/embeddings`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: EMBED_MODEL, input: ["usage probe"] }) });
  const usage = ((await raw.json()) as { usage?: unknown }).usage;
  return r.tokens === undefined
    ? { result: `no usage in the response (raw usage: ${JSON.stringify(usage) ?? "absent"}); adapter tokens undefined`, diff: "Ollama does not document `usage` for embeddings: the adapter's token counter stays 0 (docs already say 'if reported')" }
    : { result: `tokens ${r.tokens} (raw usage: ${JSON.stringify(usage)})`, diff: "usage IS present although Ollama does not document it: the token counter works" };
});
await check("embeddings: abort / timeout", async () => ({
  result: "not applicable: openAIEmbed has no abort or timeout option",
  diff: "matches the docs (they say: no retry, backoff or timeout; a hung server hangs warm())",
}));

// ---- chat
if (!CHAT_MODEL) {
  rows.push({ check: "chat checks (completion, usage, reasoning_effort, abort, timeout)", result: "SKIPPED: set OLLAMA_CHAT_MODEL (a second model; needs the owner's separate yes to download)", diff: "-" });
} else {
  const plan = { model: "m" };
  const build = (settings?: Record<string, unknown>, extra: Record<string, unknown> = {}) =>
    openAICompatiblePipeline<string>({
      baseUrl: BASE,
      models: { m: { model: CHAT_MODEL, price: { inputPerMTok: 0, outputPerMTok: 0 } } },
      prepare: (q) => ({ messages: [{ role: "user", content: q }] }),
      ...(settings ? { settings: () => settings } : {}),
      retries: 0,
      timeoutMs: 120_000,
      ...extra,
    });
  await check("chat: normal completion with usage-based cost", async () => {
    const r = await build()("Reply with the single word: pong", plan);
    return { result: `answer "${String(r.output).slice(0, 40).replace(/\s+/g, " ")}", cost $${r.costUsd} (price 0, so cost 0 is expected)` };
  });
  await check("chat: `usage` present in the response", async () => {
    const raw = await fetch(`${BASE}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: CHAT_MODEL, messages: [{ role: "user", content: "Say hi" }], max_tokens: 16 }),
    });
    const usage = ((await raw.json()) as { usage?: Record<string, unknown> }).usage;
    return usage && typeof usage.prompt_tokens === "number"
      ? { result: `usage present: ${JSON.stringify(usage)}` }
      : { result: `usage absent or incomplete: ${JSON.stringify(usage) ?? "absent"}`, diff: "the pipeline fails loudly without usage unless a `cost` callback is passed (by design)" };
  });
  await check("chat: a `cost` callback works", async () => {
    const r = await build(undefined, { cost: () => 0.5 })("Reply with the single word: pong", plan);
    if (r.costUsd !== 0.5) throw new Error(`cost was ${r.costUsd}, wanted 0.5`);
    return { result: "cost callback used (0.5)" };
  });
  await check("chat: `reasoning_effort` setting", async () => {
    try {
      await build({ reasoning_effort: "low" })("Reply with the single word: pong", plan);
      return { result: "accepted (no error); whether this model honors it cannot be told from here", diff: "the adapter passes `settings` through untouched; Ollama accepted it" };
    } catch (e) {
      return { result: `REJECTED: ${msg(e)}`, diff: "this model/server rejects `reasoning_effort`: the plan's thinking knob must not set it for this model" };
    }
  });
  await check("chat: abort via signal", async () => {
    const ac = new AbortController();
    setTimeout(() => ac.abort(), 30);
    const t0 = performance.now();
    const err = await build(undefined, { signal: ac.signal })("Write a long story about a dragon.", plan).then(() => undefined, (e: Error) => e);
    if (!err) return { result: "the call finished before the 30 ms abort (fast model); abort not exercised", diff: "inconclusive: use a bigger prompt or a slower model" };
    if (err.name !== "AbortError") throw new Error(`wrong error: ${err.name}: ${err.message}`);
    return { result: `AbortError after ${ms(t0)}` };
  });
  await check("chat: timeout", async () => {
    const err = await build(undefined, { timeoutMs: 1 })("Write a long story about a dragon.", plan).then(() => undefined, (e: Error) => e);
    if (!err) return { result: "finished within 1 ms (impossible)", diff: "unexpected" };
    return /timed? ?out|timeout/i.test(err.message) ? { result: `timeout error: ${err.message.slice(0, 80)}` } : { result: `error without the word timeout: ${err.message.slice(0, 100)}`, diff: "the error does not say it was a timeout" };
  });
  rows.push({ check: "chat: 429 / 5xx retries", result: "not testable against a healthy local server", diff: "-" });
}

// ---- table
const w = (k: keyof Row) => Math.min(70, Math.max(...rows.map((r) => r[k].length), k.length));
const cut = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + "…" : s.padEnd(n));
console.log(`${cut("check", w("check"))} | ${cut("result", w("result"))} | difference from the docs / README`);
for (const r of rows) console.log(`${cut(r.check, w("check"))} | ${cut(r.result, w("result"))} | ${r.diff}`);
const errors = rows.filter((r) => r.result.startsWith("ERROR")).length;
console.log(`\n${rows.length} rows, ${errors} unexpected error(s). Timings describe this machine only. No README claim was edited.`);
process.exit(errors ? 1 : 0);
