/**
 * RUN B tool: does routing on real embeddings beat routing on hashed text features? Uses ONLY cached model answers
 * (never a Gemini call). The embeddings come from a local Ollama server (OpenAI-compatible /v1/embeddings, $0, no key).
 *
 *   npm run bench:rag:embed-compare -- --model nomic-embed-text [--base-url http://localhost:11434/v1]
 *   npm run bench:rag:embed-compare -- --fake        SYNTHETIC: fake embedders on the mock cache, no network
 *
 * Needs the complete calibration table (RUN A): otherwise it prints the missing count and exits 0. It never pulls or
 * installs a model: if the server or the model is missing it prints the `ollama pull` command and exits 1.
 *
 * What it measures, per featurizer (hashed text, then the embedding model; --fake adds an oracle and a noise embedder):
 *  - calibration-only 5-fold cross-validation: calibrate on 80 calibration questions, serve the 20 held out, with the SLA
 *    guarantee (confidence 0.9). Every pair is cached once the table is complete. This is NOT the test set.
 *  - the 200 test questions (calibrate on all 100, serve 200) ONLY if the test table is complete for all 12 plans, since the
 *    router may pick any of them. Otherwise "test: not measured" and the missing count. No partial test numbers.
 * SLA = --sla-of-max (default 0.90) of always-max's F1 on the calibration questions, floored to 3 decimals as run.ts does.
 */
const args = process.argv.slice(2);
const FAKE = args.includes("--fake");
if (!FAKE && process.env.PATHWISE_BENCH_MOCK === "1") {
  console.error("refused: PATHWISE_BENCH_MOCK=1 is set, so a real-embeddings run would read the mock cache and look real. Unset it, or use --fake.");
  process.exit(2);
}
process.env.PATHWISE_BENCH_OFFLINE = "1"; // before the imports below read it: a cache miss throws, nothing is requested
if (FAKE) process.env.PATHWISE_BENCH_MOCK = "1"; // the fake mode reads the mock cache, never the real one
const { createPlanner, enumeratePlans, hashedTextFeaturizer, planKey } = await import("../../src/index.js");
const { embeddingFeaturizer, openAIEmbed } = await import("../../src/adapters/embeddings.js");
type EmbedFn = import("../../src/adapters/embeddings.js").EmbedFn;
const { BASELINES, SPACE } = await import("./config.js");
const { loadHotpot, sample } = await import("./dataset.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");
type Question = import("./dataset.js").Question;
type Answer = import("./pipeline.js").Answer;

const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const num = (name: string, def: number) => {
  const v = opt(name);
  if (v === undefined) return def;
  const n = Number(v);
  if (!Number.isFinite(n) || n <= 0) {
    console.error(`--${name} must be a positive number (got "${v}")`);
    process.exit(2);
  }
  return n;
};
const MODEL = opt("model") ?? "nomic-embed-text";
const BASE_URL = opt("base-url") ?? "http://localhost:11434/v1";
const FRACTION = num("sla-of-max", 0.9);
const CAL = num("calibration", 100);
const TEST = num("test", 200);
const FOLDS = 5;
const CONFIDENCE = 0.9;
const tag = FAKE ? " SYNTHETIC" : "";

const { existsSync } = await import("node:fs");
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.error("The dataset is not downloaded (bench/real-rag/data/). This tool never uses the network for it: run `npm run bench:rag:pilot` once.");
  process.exit(1);
}
const { calibration, test } = sample(await loadHotpot(), CAL, TEST, 7);
const plans = enumeratePlans(SPACE);

/** Per-question scores for every plan, plus how many pairs are missing. */
async function load(questions: Question[]) {
  const f1: number[][] = [];
  const cost: number[][] = [];
  let missing = 0;
  for (const plan of plans) {
    const a: number[] = [];
    const c: number[] = [];
    for (const q of questions) {
      try {
        const r = await ragPipeline(q, plan);
        a.push(scorer(q, r.output));
        c.push(r.costUsd);
      } catch (err) {
        if (!(err instanceof OfflineMiss)) throw err;
        missing++;
      }
    }
    f1.push(a);
    cost.push(c);
  }
  return { f1, cost, missing };
}
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

if (FAKE) console.log("SYNTHETIC: --fake, mock model answers and fake embedders. Nothing below is real data.");
const cal = await load(calibration);
console.log(`== embed-compare${tag}: ${calibration.length} calibration / ${test.length} test questions, ${plans.length} plans, 5-fold cross-validation on calibration, SLA ${Math.round(FRACTION * 100)}% of always-max, confidence ${CONFIDENCE}`);
if (cal.missing > 0) {
  console.log(`\nincomplete table: ${cal.missing} calibration pairs missing, nothing measured. Finish RUN A first: npm run bench:rag:fill -- --phase calibration --yes-live`);
  process.exit(0);
}
const maxIdx = plans.findIndex((p) => planKey(p) === planKey(BASELINES.max));
const maxF1 = mean(cal.f1[maxIdx]);
const SLA = Math.floor(FRACTION * maxF1 * 1000) / 1000;
console.log(`always-max ${planKey(BASELINES.max)}: calibration F1 ${maxF1.toFixed(3)}, ${(mean(cal.cost[maxIdx])).toFixed(5)} $/query -> SLA ${SLA}`);

// ------------------------------------------------------------------ embedders
type Featurizer = (q: Question) => number[];
interface Candidate {
  name: string;
  featurize: Featurizer;
  stats?: () => { calls: number; texts: number; tokens: number };
  dimension?: number;
  embedMs?: number;
}
const textOf = (q: Question) => q.question;
const allQuestions = [...calibration, ...test];
const candidates: Candidate[] = [];
const hashed = hashedTextFeaturizer();
candidates.push({ name: "hashed text", featurize: (q) => hashed(q.question) });

async function addEmbedder(name: string, embed: EmbedFn) {
  const f = embeddingFeaturizer<Question>({ embed, text: textOf });
  const t0 = performance.now();
  await f.warm(allQuestions);
  const embedMs = performance.now() - t0;
  candidates.push({ name, featurize: f, stats: () => f.stats, dimension: f(allQuestions[0]).length, embedMs });
}

if (FAKE) {
  // Oracle (an upper bound): encodes how hard a question is = 1 - its mean F1 over the plans, taken from the table itself.
  const difficulty = new Map<string, number>();
  for (const q of allQuestions) {
    const scores: number[] = [];
    for (const plan of plans) {
      try {
        scores.push(scorer(q, (await ragPipeline(q, plan)).output));
      } catch (err) {
        if (!(err instanceof OfflineMiss)) throw err;
      }
    }
    difficulty.set(q.question, scores.length ? 1 - mean(scores) : 0.5);
  }
  const oracle: EmbedFn = async (texts) => ({
    vectors: texts.map((t) => {
      const d = difficulty.get(t) ?? 0.5;
      return [1, 2, 3].flatMap((k) => [Math.cos((k * Math.PI * d) / 2), Math.sin((k * Math.PI * d) / 2)]);
    }),
  });
  const noise: EmbedFn = async (texts) => ({
    vectors: texts.map((t) => {
      let h = 2166136261;
      for (let i = 0; i < t.length; i++) h = Math.imul(h ^ t.charCodeAt(i), 16777619);
      const v: number[] = [];
      for (let i = 0; i < 16; i++) {
        h = Math.imul(h ^ (h >>> 15), 2246822507) ^ i;
        v.push(((h >>> 0) / 4294967296) - 0.5);
      }
      return v;
    }),
  });
  await addEmbedder("fake oracle (upper bound)", oracle);
  await addEmbedder("fake noise (control)", noise);
} else {
  const embed = openAIEmbed({ model: MODEL, baseUrl: BASE_URL });
  try {
    await embed(["probe"]);
  } catch (err) {
    const msg = (err as Error).message;
    if (/HTTP 404|not found|try pulling/i.test(msg)) {
      console.error(`The model "${MODEL}" is not available on ${BASE_URL} (or --base-url has the wrong path: it should end in /v1). This tool never pulls models. Run:\n  ollama pull ${MODEL}\nthen run this command again.\n(server said: ${msg.slice(0, 200)})`);
    } else {
      console.error(`The embeddings server at ${BASE_URL} did not answer usefully (${msg.slice(0, 200)}).\nStart Ollama (the app, or \`ollama serve\`), run \`ollama pull ${MODEL}\` once, then run this command again.`);
    }
    process.exit(1);
  }
  await addEmbedder(`${MODEL} (Ollama)`, embed);
}

// ------------------------------------------------------------------ measurements
function plannerFor(c: Candidate) {
  return createPlanner<Question, Answer>({
    space: SPACE,
    pipeline: ragPipeline,
    scorer,
    sla: { minQuality: SLA, confidence: CONFIDENCE },
    featurize: c.featurize,
    feedbackRate: 0,
    concurrency: 2,
  });
}

interface Row {
  f1: number;
  cost: number;
  met: boolean;
  certified: string;
  calRuns: number;
}

async function crossValidate(c: Candidate): Promise<Row> {
  const f1: number[] = [];
  const cost: number[] = [];
  let certified = 0;
  let runs = 0;
  for (let k = 0; k < FOLDS; k++) {
    const train = calibration.filter((_, i) => i % FOLDS !== k);
    const heldOut = calibration.filter((_, i) => i % FOLDS === k);
    const planner = plannerFor(c);
    const report = await planner.calibrate(train);
    certified += report.slaCertified ? 1 : 0;
    runs += report.runs;
    for (const q of heldOut) {
      const r = await planner.run(q);
      f1.push(scorer(q, r.output));
      cost.push(r.costUsd);
    }
  }
  return { f1: mean(f1), cost: mean(cost), met: mean(f1) >= SLA, certified: `${certified}/${FOLDS} folds`, calRuns: runs / FOLDS };
}

async function serveTest(c: Candidate): Promise<Row> {
  const planner = plannerFor(c);
  const report = await planner.calibrate(calibration);
  const f1: number[] = [];
  const cost: number[] = [];
  for (const q of test) {
    const r = await planner.run(q);
    f1.push(scorer(q, r.output));
    cost.push(r.costUsd);
  }
  return { f1: mean(f1), cost: mean(cost), met: mean(f1) >= SLA, certified: String(report.slaCertified), calRuns: report.runs };
}

const row = (name: string, r: Row) =>
  `${name.padEnd(28)} ${r.f1.toFixed(3).padStart(7)}  ${r.met ? "met   " : "MISSED"}  $${r.cost.toFixed(5)}  ${r.certified.padStart(12)}  ${r.calRuns.toFixed(0).padStart(8)}`;
const header = "features                       F1     SLA     $/query    certified    calib runs";

console.log(`\n-- calibration-only ${FOLDS}-fold cross-validation (${calibration.length} held-out questions in total; NOT the test set)${tag}`);
console.log(header);
const cv = new Map<string, Row>();
for (const c of candidates) {
  const r = await crossValidate(c);
  cv.set(c.name, r);
  console.log(row(c.name, r));
}
console.log(`reference: always-max serves at ${mean(cal.cost[maxIdx]).toFixed(5)} $/query (calibration questions).`);
const h = cv.get("hashed text")!;
for (const c of candidates.slice(1)) {
  const r = cv.get(c.name)!;
  const better = r.cost < h.cost - 1e-9 && r.f1 >= h.f1 - 1e-9;
  const worse = r.cost > h.cost + 1e-9 && r.f1 <= h.f1 + 1e-9;
  console.log(`${c.name} vs hashed: ${better ? "cheaper at equal or better F1" : worse ? "dearer at equal or worse F1" : "mixed or equal (cost and F1 move in different directions, or tie)"} (F1 ${(r.f1 - h.f1 >= 0 ? "+" : "")}${(r.f1 - h.f1).toFixed(3)}, cost ${(r.cost - h.cost >= 0 ? "+" : "")}$${(r.cost - h.cost).toFixed(5)}).`);
  if (c.name.includes("noise")) console.log(`  control: the noise embedder ${better ? "BEAT hashed features (unexpected; report it)" : "did not beat hashed features, as it should not"}.`);
}
for (const c of candidates) {
  const r = cv.get(c.name)!;
  if (!r.certified.startsWith(`${FOLDS}/`)) {
    console.log(
      `warning: ${c.name} certified only ${r.certified}. A fold that cannot certify serves one safe plan for every request, which depends little on the features, so this row compares the featurizers only weakly (each fold calibrates on ${calibration.length - calibration.length / FOLDS} questions).`,
    );
  }
}
console.log("(100 questions, one dataset, one seed: differences of a few points are noise. Not a significance test. The SLA is derived from the same 100 questions, including each fold's held-out 20, identically for every featurizer.)");

const testTable = await load(test);
console.log(`\n-- test set (${test.length} questions)${tag}`);
if (testTable.missing > 0) {
  console.log(`test: not measured. ${testTable.missing} of ${plans.length * test.length} test pairs are missing and the router may pick any plan; no partial test numbers are shown.`);
} else {
  console.log(header);
  for (const c of candidates) console.log(row(c.name, await serveTest(c)));
}

console.log("\n-- embedding cost");
for (const c of candidates) {
  if (!c.stats) continue;
  const s = c.stats();
  console.log(`${c.name}: ${s.calls} embed calls, ${s.texts} texts, ${s.tokens} tokens reported by the server${s.tokens === 0 ? " (none reported)" : ""}, dimension ${c.dimension}, ${c.embedMs!.toFixed(0)} ms to embed ${allQuestions.length} question texts${FAKE ? " (fake embedder)" : ""}. Cost $0${FAKE ? "" : " (local model)"}.`);
}
console.log("\n(no Gemini call was made)");
