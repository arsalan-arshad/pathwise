/** HotpotQA dev (distractor setting): download once, sample deterministically, retrieve with BM25. */
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { DATASET_ROWS_URL, PATHS } from "./config.js";

export interface HotpotExample {
  _id: string;
  question: string;
  answer: string;
  type: "comparison" | "bridge";
  level: "easy" | "medium" | "hard";
  supporting_facts: Array<[string, number]>;
  context: Array<[string, string[]]>;
}

export interface Passage {
  title: string;
  text: string;
}

export interface Question {
  id: string;
  question: string;
  answer: string;
  type: string;
  passages: Passage[];
  /** Titles of the 2 paragraphs that contain the answer (used only for diagnostics and the mock). */
  goldTitles: string[];
}

const FILE = new URL("hotpot_dev_distractor_v1.json", PATHS.data);

/** Hugging Face row shape (parallel lists instead of pairs). */
interface HfRow {
  id: string;
  question: string;
  answer: string;
  type: "comparison" | "bridge";
  level: "easy" | "medium" | "hard";
  supporting_facts: { title: string[]; sent_id: number[] };
  context: { title: string[]; sentences: string[][] };
}

/**
 * Downloads the dev set (7,405 questions, 75 pages of 100) on first use. Each page is saved as it arrives,
 * so an interrupted download (e.g. a Hugging Face rate limit) resumes where it stopped on the next run.
 */
export async function loadHotpot(): Promise<HotpotExample[]> {
  if (!existsSync(FILE)) {
    const pagesDir = new URL("pages/", PATHS.data);
    await mkdir(pagesDir, { recursive: true });
    const examples: HotpotExample[] = [];
    let total = Infinity;
    for (let offset = 0; offset < total; offset += 100) {
      const pageFile = new URL(`offset-${String(offset).padStart(5, "0")}.json`, pagesDir);
      let page: { total: number; examples: HotpotExample[] };
      if (existsSync(pageFile)) {
        page = JSON.parse(await readFile(pageFile, "utf8"));
      } else {
        if (offset > 0) await sleep(PAGE_DELAY_MS); // be polite: the viewer API rate-limits bursts
        page = await fetchPage(offset);
        await writeFile(pageFile, JSON.stringify(page));
      }
      total = page.total;
      examples.push(...page.examples);
      process.stdout.write(`\rHotpotQA dev set from Hugging Face: ${examples.length}/${total} questions`);
    }
    process.stdout.write("\n");
    await writeFile(FILE, JSON.stringify(examples));
  }
  return JSON.parse(await readFile(FILE, "utf8")) as HotpotExample[];
}

const PAGE_DELAY_MS = 1500;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function convert(row: HfRow): HotpotExample {
  return {
    _id: row.id,
    question: row.question,
    answer: row.answer,
    type: row.type,
    level: row.level,
    supporting_facts: row.supporting_facts.title.map((t, i) => [t, row.supporting_facts.sent_id[i]]),
    context: row.context.title.map((t, i) => [t, row.context.sentences[i]]),
  };
}

/** One page of 100 rows. Waits out rate limits (honouring Retry-After) instead of giving up. */
async function fetchPage(offset: number): Promise<{ total: number; examples: HotpotExample[] }> {
  const url = `${DATASET_ROWS_URL}&offset=${offset}&length=100`;
  for (let attempt = 1; ; attempt++) {
    let res: Response | undefined;
    try {
      res = await fetch(url, { signal: AbortSignal.timeout(60_000) });
    } catch (err) {
      if (attempt >= 8) throw err; // network trouble that does not go away
    }
    if (res?.ok) {
      const json = (await res.json()) as { num_rows_total: number; rows: Array<{ row: HfRow; truncated_cells: string[] }> };
      for (const { row, truncated_cells } of json.rows) {
        if (truncated_cells.length > 0) throw new Error(`Row ${row.id} came back truncated; run again later`);
      }
      return { total: json.num_rows_total, examples: json.rows.map(({ row }) => convert(row)) };
    }
    if (res && res.status !== 429 && res.status < 500) {
      throw new Error(`Dataset download failed at offset ${offset}: HTTP ${res.status}`);
    }
    if (attempt >= 12) {
      throw new Error(
        `Dataset download still failing at offset ${offset} (HTTP ${res?.status ?? "network error"}). ` +
          "Run the same command again later: pages downloaded so far are kept.",
      );
    }
    const retryAfter = Number(res?.headers.get("retry-after"));
    const waitS =
      res?.status === 429
        ? Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter
          : Math.min(300, 30 * attempt)
        : Math.min(60, 2 ** attempt);
    process.stdout.write(`\n  ${res?.status === 429 ? "Rate limited" : "Temporary error"} at offset ${offset}; waiting ${waitS}s ...`);
    await sleep(waitS * 1000);
  }
}

/** Deterministic shuffle, then disjoint calibration and test slices. */
export function sample(all: HotpotExample[], calibration: number, test: number, seed = 7) {
  const idx = all.map((_, i) => i);
  let s = seed;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  if (calibration + test > idx.length) throw new Error("Not enough questions");
  const toQ = (e: HotpotExample): Question => ({
    id: e._id,
    question: e.question,
    answer: e.answer,
    type: e.type,
    passages: e.context.map(([title, sents]) => ({ title, text: sents.join("").trim() })),
    goldTitles: [...new Set(e.supporting_facts.map(([t]) => t))],
  });
  return {
    calibration: idx.slice(0, calibration).map((i) => toQ(all[i])),
    test: idx.slice(calibration, calibration + test).map((i) => toQ(all[i])),
  };
}

const STOP = new Set("a an the of in on at to for and or is are was were be by with from as that this which who what when where how did does do".split(" "));
const tokens = (s: string) => (s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter((t) => !STOP.has(t));

/** BM25 over one question's own passages (title + text). Returns the top k, best first. */
export function retrieve(q: Question, k: number, k1 = 1.5, b = 0.75): Passage[] {
  const docs = q.passages.map((p) => tokens(`${p.title} ${p.text}`));
  const avgLen = docs.reduce((s, d) => s + d.length, 0) / Math.max(1, docs.length);
  const df = new Map<string, number>();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
  const N = docs.length;
  const qt = tokens(q.question);
  const scored = docs.map((d, i) => {
    const tf = new Map<string, number>();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const t of qt) {
      const f = tf.get(t) ?? 0;
      if (f === 0) continue;
      const n = df.get(t) ?? 0;
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
      score += (idf * f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.length) / avgLen));
    }
    return { i, score };
  });
  scored.sort((a, b) => b.score - a.score || a.i - b.i);
  return scored.slice(0, k).map(({ i }) => q.passages[i]);
}
