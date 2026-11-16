/**
 * Read-only check of the MuSiQue loader on the REAL local dev file (offline: no network, no key, no model call, no download).
 *
 *   npx tsx bench/real-rag/musique-verify.ts [--n 300 --seed 7 --musique-file <path>]
 *
 * Prints: questions loaded, count per `type` (the loader groups id prefixes by hop count: 3hop1/3hop2 -> 3hop, 4hop1..3 -> 4hop),
 * whether every question has a non-empty answer and at least one supporting paragraph, and for the sampled questions (the same
 * seeded sample `fill`/`probe` use) the share where BM25 over the question's own paragraphs puts at least one supporting
 * paragraph in the top 2 / 5 / 10. REAL data, one file, one seed.
 */
import { existsSync, readFileSync } from "node:fs";
import { retrieve } from "./dataset.js";
import { musiqueFile, parseMusique, sampleFirst } from "./musique.js";

const args = process.argv.slice(2);
const opt = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const N = Number(opt("n") ?? 300);
const SEED = Number(opt("seed") ?? 7);
const file = musiqueFile(opt("musique-file"));
if (!existsSync(file)) {
  console.error(`MuSiQue file not found: ${file}. This tool never downloads it.`);
  process.exit(1);
}
const text = readFileSync(file, "utf8");
const rawRows = text.split("\n").filter((l) => l.trim()).length;
const all = parseMusique(text, file);
console.log(`== musique verify: ${file}`);
console.log(`lines in file ${rawRows}; questions loaded ${all.length} (unanswerable rows are dropped; none expected in the ans file)`);

const byType = new Map<string, number>();
for (const q of all) byType.set(q.type, (byType.get(q.type) ?? 0) + 1);
console.log(`per type (id prefix grouped by hop count, so 3hop1+3hop2 -> 3hop and 4hop1+4hop2+4hop3 -> 4hop): ${[...byType].map(([k, v]) => `${k} ${v}`).join(", ")}`);
const rawPrefix = new Map<string, number>();
for (const line of text.split("\n")) {
  const m = /"id":\s*"([0-9a-z]+)__/.exec(line);
  if (m) rawPrefix.set(m[1], (rawPrefix.get(m[1]) ?? 0) + 1);
}
console.log(`raw id prefixes: ${[...rawPrefix].sort().map(([k, v]) => `${k} ${v}`).join(", ")}`);

const noAnswer = all.filter((q) => !q.answer.trim()).length;
const noSupport = all.filter((q) => q.goldTitles.length === 0).length;
const paragraphs = new Set(all.map((q) => q.passages.length));
console.log(`questions with an empty answer: ${noAnswer}; with no supporting paragraph: ${noSupport}; paragraphs per question: ${[...paragraphs].sort((a, b) => a - b).join(", ")}`);

const sampled = sampleFirst(all, Math.min(N, all.length), SEED);
const hit = (k: number) => sampled.filter((q) => retrieve(q, k).some((p) => q.goldTitles.includes(p.title))).length;
const pct = (x: number) => `${x}/${sampled.length} (${((100 * x) / sampled.length).toFixed(1)}%)`;
console.log(`BM25 retrieval over each question's own paragraphs, ${sampled.length} sampled questions (seed ${SEED}), at least one supporting paragraph in the top 2: ${pct(hit(2))}; top 5: ${pct(hit(5))}; top 10: ${pct(hit(10))}`);
console.log("(REAL data, offline, one file, one seed. Retrieval hit rate is a data-path check, not a quality claim about any model.)");
