/**
 * Predictability probe (offline, $0, cache only, no key): does the question text tell which plan a question needs?
 * For a question set on which two named plans are cached for EVERY question, the label is "the cheap plan scores at least
 * as high as the max plan". Prints the base rate and the leave-one-out kNN AUC of hashed text features (and optionally a
 * local Ollama embedding), and the AUC of question metadata (`type`). About 0.5 = no signal. Not a standard threshold:
 * desktop's gate for filling a full table on a new dataset is "clearly above 0.5, say 0.65".
 *
 *   npm run bench:rag:probe [-- --dataset hotpot|musique --cheap <id> --max <id> --k 15 --n 300 --embed ollama --model nomic-embed-text]
 *
 * The AUC is also printed with the 5th to 95th percentile you get when the labels are shuffled (200 seeded shuffles), so
 * "0.455" can be read against what noise does at this n. REAL data when the cache is real; SYNTHETIC under PATHWISE_BENCH_MOCK=1.
 */
process.env.PATHWISE_BENCH_OFFLINE = "1";
const { hashedTextFeaturizer, planKey } = await import("../../src/index.js");
const { BASELINES } = await import("./config.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");
const { parseDataset, questionsFor } = await import("./questions.js");
const pc = await import("./probe-core.js");
type Question = import("./dataset.js").Question;

const args = process.argv.slice(2);
const opt = (name: string): string | undefined => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const num = (name: string, def: number) => (opt(name) === undefined ? def : Number(opt(name)));
let dataset;
try {
  dataset = parseDataset(opt("dataset"));
} catch (e) {
  console.error((e as Error).message);
  process.exit(2);
}
const cheapId = opt("cheap") ?? planKey(BASELINES.cheap);
const maxId = opt("max") ?? planKey(BASELINES.max);
const K = num("k", 15);
const { enumeratePlans, } = await import("../../src/index.js");
const { SPACE } = await import("./config.js");
const byId = new Map(enumeratePlans(SPACE).map((p) => [planKey(p), p]));
for (const id of [cheapId, maxId]) if (!byId.has(id)) {
  console.error(`unknown plan id "${id}". Known ids:\n  ${[...byId.keys()].join("\n  ")}`);
  process.exit(2);
}
let questions: Question[];
try {
  questions = await questionsFor(dataset, { n: opt("n") === undefined ? undefined : num("n", 300), seed: num("seed", 7), musiqueFile: opt("musique-file") });
} catch (e) {
  console.error((e as Error).message);
  process.exit(1);
}
const f1 = (plan: string) => async (qs: Question[]) => {
  const out: Array<number | null> = [];
  for (const q of qs) {
    try {
      out.push(scorer(q, (await ragPipeline(q, byId.get(plan)!)).output));
    } catch (err) {
      if (!(err instanceof OfflineMiss)) throw err;
      out.push(null);
    }
  }
  return out;
};
const cheap = await f1(cheapId)(questions);
const max = await f1(maxId)(questions);
const missing = cheap.filter((x) => x === null).length + max.filter((x) => x === null).length;
if (process.env.PATHWISE_BENCH_MOCK === "1") console.log("SYNTHETIC: PATHWISE_BENCH_MOCK=1, the model answers are mock answers, not real data.");
console.log(`== probe: dataset ${dataset}, ${questions.length} questions (loaded), cheap ${cheapId}, max ${maxId}, k ${K}`);
if (missing > 0) {
  console.log(`\nincomplete: ${missing} (question, plan) pairs missing for the two plans, nothing measured. Fill them with bench:rag:fill (see bench/real-rag/README.md).`);
  process.exit(0);
}
const labels: number[] = questions.map((_, i) => ((cheap[i] as number) >= (max[i] as number) ? 1 : 0));
const same = questions.filter((_, i) => cheap[i] === max[i]).length;
const base = labels.reduce((a, b) => a + b, 0) / labels.length;
console.log(`label = cheap plan scores at least as high as the max plan: base rate ${base.toFixed(3)} (${labels.reduce((a, b) => a + b, 0)} of ${labels.length}); identical score on both plans: ${same}`);
const fmt = (x: number | null) => (x === null ? "n/a" : x.toFixed(3));
function report(name: string, scoreOf: (l: number[]) => number[]) {
  const a = pc.auc(scoreOf(labels), labels);
  const band = pc.permutationBand(scoreOf, labels);
  console.log(`${name.padEnd(46)} AUC ${fmt(a)}   (shuffled labels: 5th to 95th percentile ${band ? `${band.lo.toFixed(3)} to ${band.hi.toFixed(3)}` : "n/a"})`);
}
const hashed = hashedTextFeaturizer();
const nb = pc.neighbours(questions.map((q) => hashed(q.question)), K);
report(`hashed text, leave-one-out kNN (k=${K})`, (l) => pc.knnScores(nb, l, false));
report(`hashed text, similarity-weighted kNN (k=${K})`, (l) => pc.knnScores(nb, l, true));
const types = [...new Set(questions.map((q) => q.type))];
report(`question type (${types.join(" vs ")}), group rate`, (l) => pc.groupScores(questions.map((q) => q.type), l));
report("question type, leave-one-out (artifact: ignore)", (l) => pc.groupScores(questions.map((q) => q.type), l, true));
if (opt("embed") !== undefined) {
  if (opt("embed") !== "ollama") {
    console.error('--embed supports only "ollama"');
    process.exit(2);
  }
  if (process.env.PATHWISE_BENCH_MOCK === "1") {
    console.error("refused: --embed with PATHWISE_BENCH_MOCK=1 would mix a real embedding with mock answers.");
    process.exit(2);
  }
  const { embeddingFeaturizer, openAIEmbed } = await import("../../src/adapters/embeddings.js");
  const MODEL = opt("model") ?? "nomic-embed-text";
  const BASE = opt("base-url") ?? "http://localhost:11434/v1";
  const f = embeddingFeaturizer<Question>({ embed: openAIEmbed({ model: MODEL, baseUrl: BASE }), text: (q) => q.question });
  try {
    await f.warm(questions);
  } catch (err) {
    console.error(`The embeddings server at ${BASE} failed (${(err as Error).message.slice(0, 200)}). Start Ollama and run \`ollama pull ${MODEL}\` yourself; this tool never pulls a model.`);
    process.exit(1);
  }
  const nbe = pc.neighbours(questions.map((q) => f(q)), K);
  report(`${MODEL} (Ollama), leave-one-out kNN (k=${K})`, (l) => pc.knnScores(nbe, l, false));
}
console.log("\n(offline: no Gemini call was made. About 0.5 = the feature does not separate the two cases. One dataset, one seed, in-sample.)");
