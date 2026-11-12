/**
 * Real-embeddings adapter example and benchmark. No network, no key: FAKE embedders only (SYNTHETIC).
 *
 * 1. HTTP shapes: openAIEmbed / voyageEmbed against a fake fetch (request body, order by index, token counter).
 * 2. Quality: "messy" workload (question text predicts difficulty poorly), SLA 0.9 guaranteed at 90% confidence,
 *    hashed text features vs a difficulty-ENCODING fake embedder (an oracle, i.e. an upper bound) vs a pure-NOISE
 *    fake embedder (the control, which must not beat hashed features).
 * 3. Latency: warm() cost and decide() after warm.
 *
 * Usage: npx tsx examples/embeddings.ts [seeds=8] [calibrationSamples=90]
 */
import { performance } from "node:perf_hooks";
import { createPlanner, hashedTextFeaturizer } from "../src/index.js";
import { embeddingFeaturizer, openAIEmbed, voyageEmbed, type EmbedFn } from "../src/adapters/embeddings.js";
import { createWorkload, ragPipeline, rng, SPACE, type Query } from "./workload.js";

const SEEDS = Number(process.argv[2] ?? 8);
const CALIBRATION = Number(process.argv[3] ?? 90);
let failures = 0;
const check = (ok: boolean, msg: string) => {
  if (!ok) {
    failures++;
    console.log("FAIL:", msg);
  }
};

// ---------------------------------------------------------------- 1. HTTP shapes (fake fetch)
{
  const seen: { url: string; headers: Record<string, string>; body: any }[] = [];
  const fakeFetch = async (url: string, init: { method: string; headers: Record<string, string>; body: string }) => {
    const body = JSON.parse(init.body);
    seen.push({ url, headers: init.headers, body });
    const inputs: string[] = body.input;
    // Return out of order on purpose: the adapter must use `index`.
    const data = inputs.map((t, index) => ({ object: "embedding", index, embedding: [t.length, 1, 0] })).reverse();
    return { ok: true, status: 200, text: async () => "", json: async () => ({ data, usage: { total_tokens: inputs.length * 5 } }) };
  };
  const oa = await openAIEmbed({ apiKey: "k", model: "m", dimensions: 3, fetch: fakeFetch })(["a", "bbb"]);
  check(seen[0].url === "https://api.openai.com/v1/embeddings" && seen[0].headers.authorization === "Bearer k", "openai url/auth");
  check(seen[0].body.model === "m" && seen[0].body.dimensions === 3 && seen[0].body.encoding_format === "float", "openai body");
  check(oa.vectors[0][0] === 1 && oa.vectors[1][0] === 3 && oa.tokens === 10, "openai order by index + tokens");
  await openAIEmbed({ model: "nomic", baseUrl: "http://localhost:11434/v1/", fetch: fakeFetch })(["x"]);
  check(seen[1].url === "http://localhost:11434/v1/embeddings" && seen[1].headers.authorization === undefined, "local server url, no auth header");
  const vy = await voyageEmbed({ apiKey: "v", model: "voyage-4-lite", inputType: "query", outputDimension: 256, fetch: fakeFetch })(["zz", "y"]);
  check(seen[2].url === "https://api.voyageai.com/v1/embeddings" && seen[2].body.input_type === "query" && seen[2].body.output_dimension === 256, "voyage body");
  check(vy.vectors[0][0] === 2 && vy.tokens === 10, "voyage order + tokens");
  const failing = openAIEmbed({ model: "m", fetch: async () => ({ ok: false, status: 429, text: async () => "slow down", json: async () => ({}) }) });
  check(await failing(["a"]).then(() => false, (e: Error) => /HTTP 429/.test(e.message)), "HTTP error surfaces");

  // Featurizer behaviour: batching, cache, dimension check, miss handling.
  let embedCalls = 0;
  const f = embeddingFeaturizer<string>({ embed: async (t) => (embedCalls++, { vectors: t.map((x) => [x.length, 1]), tokens: t.length }), batchSize: 2 });
  await f.warm(["a", "bb", "ccc", "a"]);
  check(embedCalls === 2 && f.stats.texts === 3 && f.stats.tokens === 3, `batching: ${embedCalls} calls, ${f.stats.texts} texts`);
  await f.warm(["a", "bb"]);
  check(embedCalls === 2, "warm of cached inputs makes no call");
  check(Math.abs(Math.hypot(...f("ccc")) - 1) < 1e-9 && f.stats.cacheHits === 1, "unit vectors, cache hit counted");
  check((() => { try { f("never"); return false; } catch (e) { return /warm/.test((e as Error).message); } })(), "miss throws a clear error");
  // Concurrent warm() of the same new text embeds it once.
  let concurrentCalls = 0;
  const c = embeddingFeaturizer<string>({ embed: async (t) => (concurrentCalls++, await new Promise((r) => setTimeout(r, 5)), { vectors: t.map(() => [1, 0]) }) });
  await Promise.all([c.warm(["same"]), c.warm(["same"]), c.warm(["same"])]);
  check(concurrentCalls === 1, `concurrent warm: ${concurrentCalls} calls`);
  // Servers that omit `index`: positional order is used; a short response is an error.
  const noIndex = async (_u: string, init: { body: string }) => {
    const input: string[] = JSON.parse(init.body).input;
    return { ok: true, status: 200, text: async () => "", json: async () => ({ data: input.map((t) => ({ embedding: [t.length] })) }) };
  };
  check((await openAIEmbed({ model: "m", fetch: noIndex })(["a", "bbb"])).vectors[1][0] === 3, "no index: positional");
  check(await openAIEmbed({ model: "m", fetch: async () => ({ ok: true, status: 200, text: async () => "", json: async () => ({ data: [{ embedding: [1] }] }) }) })(["a", "b"]).then(() => false, (e: Error) => /got 1 vectors/.test(e.message)), "short response throws");
  // A bad vector does not fix the dimension or poison the cache.
  const nan = embeddingFeaturizer<string>({ embed: async (t) => ({ vectors: t.map(() => [Number.NaN, 1]), tokens: 7 }) });
  check(await nan.warm(["x"]).then(() => false, (e: Error) => /invalid vector/.test(e.message)) && nan.stats.tokens === 7, "NaN vector rejected, tokens still counted");
  const bad = embeddingFeaturizer<string>({ embed: async (t) => ({ vectors: t.map((x, i) => (i === 0 ? [1, 0] : [1, 0, 0])) }) });
  check(await bad.warm(["p", "q"]).then(() => false, (e: Error) => /dimensions/.test(e.message)), "dimension mismatch throws");
}

// ---------------------------------------------------------------- 2. Quality (SYNTHETIC)
// Side channel: the fake embedders read the hidden difficulty from the text the featurizer hands them.
const textOf = (q: Query) => `${q.text}\u0000${q.difficulty}`;
const oracleEmbed: EmbedFn = async (texts) => ({
  vectors: texts.map((t) => {
    const d = Number(t.split("\u0000")[1]);
    return [1, 2, 3].flatMap((k) => [Math.cos((k * Math.PI * d) / 2), Math.sin((k * Math.PI * d) / 2)]);
  }),
});
const noiseEmbed: EmbedFn = async (texts) => ({
  vectors: texts.map((t) => {
    let h = 2166136261;
    for (let i = 0; i < t.length; i++) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
    const r = rng(h);
    return Array.from({ length: 16 }, () => r() - 0.5);
  }),
});

type Kind = "hashed" | "oracle" | "noise";
async function once(seed: number, kind: Kind, guaranteed: boolean) {
  const { makeQueries } = createWorkload(seed, { jitter: 0.15 });
  const calibration = makeQueries(CALIBRATION);
  const production = makeQueries(1000);
  const hashed = hashedTextFeaturizer();
  const emb = kind === "hashed" ? undefined : embeddingFeaturizer<Query>({ embed: kind === "oracle" ? oracleEmbed : noiseEmbed, text: textOf });
  await emb?.warm([...calibration, ...production]);
  const planner = createPlanner<Query, { correct: boolean }>({
    space: { ...SPACE, model: [...SPACE.model], topK: [...SPACE.topK], rerank: [...SPACE.rerank] },
    pipeline: ragPipeline,
    scorer: (_q, o) => (o.correct ? 1 : 0),
    sla: guaranteed ? { minQuality: 0.9, confidence: 0.9, onUncertified: "safe" } : { minQuality: 0.9 },
    featurize: emb ?? ((q) => hashed(q.text)),
    feedbackRate: 0,
  });
  const calib = await planner.calibrate(calibration);
  let correct = 0;
  for (const q of production) if ((await planner.run(q)).output.correct) correct++;
  return { miss: correct / production.length < 0.9, acc: correct / production.length, prod: planner.report().totalCostUsd, calibRuns: calib.runs, certified: calib.slaCertified };
}

console.log(`SYNTHETIC, messy workload, SLA 0.9, ${SEEDS} seeds, ${CALIBRATION} calibration samples, 1000 production queries`);
console.log("mode        features  SLA misses  certified  mean accuracy  prod cost  calibration runs");
for (const guaranteed of [false, true]) {
  for (const kind of ["hashed", "oracle", "noise"] as Kind[]) {
    const rows: Awaited<ReturnType<typeof once>>[] = [];
    for (let s = 1; s <= SEEDS; s++) rows.push(await once(s, kind, guaranteed));
    const mean = (f: (r: (typeof rows)[number]) => number) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
    console.log(
      `${(guaranteed ? "guaranteed" : "mean").padEnd(11)} ${kind.padEnd(9)} ${`${rows.filter((r) => r.miss).length}/${SEEDS}`.padStart(9)}  ${`${rows.filter((r) => r.certified).length}/${SEEDS}`.padStart(9)}  ${(mean((r) => r.acc) * 100).toFixed(1).padStart(12)}%  $${mean((r) => r.prod).toFixed(2).padStart(7)}  ${mean((r) => r.calibRuns).toFixed(0).padStart(10)}`,
    );
  }
}

// ---------------------------------------------------------------- 3. Latency
{
  const { makeQueries } = createWorkload(1, { jitter: 0.15 });
  const qs = makeQueries(1000);
  const emb = embeddingFeaturizer<Query>({ embed: oracleEmbed, text: textOf });
  let t = performance.now();
  await emb.warm(qs);
  const warmMs = performance.now() - t;
  const planner = createPlanner<Query, { correct: boolean }>({
    space: { ...SPACE, model: [...SPACE.model], topK: [...SPACE.topK], rerank: [...SPACE.rerank] },
    pipeline: ragPipeline, scorer: (_q, o) => (o.correct ? 1 : 0), sla: { minQuality: 0.9 }, featurize: emb, feedbackRate: 0,
  });
  await planner.calibrate(qs.slice(0, 90));
  t = performance.now();
  for (let i = 0; i < 2000; i++) planner.decide(qs[i % qs.length]);
  const decideMs = (performance.now() - t) / 2000;
  console.log(`\nlatency (fake embedder, no network): warm(1000 inputs) ${warmMs.toFixed(1)} ms in ${emb.stats.calls} calls; decide() after warm ${decideMs.toFixed(3)} ms`);
  console.log("real embedders add network time or cost only for NEW texts: warm in batches, cache hits are free. Embedding cost = stats.tokens x price (OpenAI text-embedding-3-small $0.02/1M, Voyage voyage-4-lite $0.02/1M, local $0).");
}

console.log(failures === 0 ? "\nAll embeddings checks passed." : `\n${failures} check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
