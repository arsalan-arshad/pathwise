/**
 * MuSiQue loader (CC BY 4.0, https://github.com/StonyBrookNLP/musique). Reads a LOCALLY downloaded jsonl file: this module
 * never downloads anything. Download it yourself with the repo's `download_data.sh` (it writes into that repo's `data/`),
 * then copy e.g. `musique_ans_v1.0_dev.jsonl` to bench/real-rag/data/ or point PATHWISE_MUSIQUE_FILE at it.
 *
 * NOT VERIFIED against the real file by the author: the README states the licence and the file names
 * (data/musique_ans_v1.0_dev.jsonl, data/musique_full_v1.0_dev.jsonl) but NOT the JSON field names, the id-prefix meaning
 * or the dev-set size. Assumed official format, one JSON object per line:
 *   id ("2hop__...", "3hop1__...", "4hop3__..."), question, answer, answerable (optional),
 *   paragraphs: [{ idx, title, paragraph_text, is_supporting }].
 * `type` = hop count taken from the id prefix ("2hop", "3hop", "4hop"). Anything missing makes the loader throw an error that
 * names the field and the line, so a wrong assumption fails loudly instead of producing garbage. Unanswerable rows are dropped.
 * The scorer uses `answer` only (token F1); `answer_aliases` are ignored.
 */
import { existsSync, readFileSync } from "node:fs";
import { PATHS } from "./config.js";
import { shuffleIndices, type Question } from "./dataset.js";

export const DEFAULT_MUSIQUE_FILE = new URL("musique_ans_v1.0_dev.jsonl", PATHS.data).pathname;

export function musiqueFile(override?: string): string {
  return override ?? process.env.PATHWISE_MUSIQUE_FILE ?? DEFAULT_MUSIQUE_FILE;
}

/** Parses jsonl text into questions. Pure (no file access), so tests can use a tiny SYNTHETIC fixture. */
export function parseMusique(text: string, source = "musique"): Question[] {
  const out: Question[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].trim();
    if (!raw) continue;
    const where = `${source} line ${i + 1}`;
    let row: Record<string, unknown>;
    try {
      row = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      throw new Error(`${where}: not valid JSON`);
    }
    const str = (k: string): string => {
      const v = row[k];
      if (typeof v !== "string" || v.length === 0) throw new Error(`${where}: expected a non-empty string field "${k}"`);
      return v;
    };
    if (row.answerable === false) continue;
    const id = str("id");
    const hop = /^(\d)hop/.exec(id);
    if (!hop) throw new Error(`${where}: id "${id}" does not start with "<n>hop"`);
    const paragraphs = row.paragraphs;
    if (!Array.isArray(paragraphs) || paragraphs.length === 0) throw new Error(`${where}: expected a non-empty array field "paragraphs"`);
    const passages = paragraphs.map((p: unknown, j) => {
      const r = p as Record<string, unknown>;
      if (typeof r?.title !== "string" || typeof r?.paragraph_text !== "string") throw new Error(`${where}: paragraphs[${j}] needs string "title" and "paragraph_text"`);
      return { title: r.title, text: r.paragraph_text.trim(), supporting: r.is_supporting === true };
    });
    out.push({
      id,
      question: str("question"),
      answer: str("answer"),
      type: `${hop[1]}hop`,
      passages: passages.map(({ title, text }) => ({ title, text })),
      goldTitles: [...new Set(passages.filter((p) => p.supporting).map((p) => p.title))],
    });
  }
  return out;
}

export function loadMusique(file?: string): Question[] {
  const path = musiqueFile(file);
  if (!existsSync(path)) {
    throw new Error(
      `MuSiQue file not found: ${path}. This tool never downloads it. Download with the repo's download_data.sh (https://github.com/StonyBrookNLP/musique, CC BY 4.0) and copy musique_ans_v1.0_dev.jsonl there, or set PATHWISE_MUSIQUE_FILE.`,
    );
  }
  return parseMusique(readFileSync(path, "utf8"), path);
}

/** The first `n` questions of a seeded shuffle of the whole file (the "sample" the fill and probe tools use). */
export function sampleFirst<T>(all: T[], n: number, seed = 7): T[] {
  if (n > all.length) throw new Error(`Only ${all.length} questions available, ${n} requested`);
  return shuffleIndices(all.length, seed)
    .slice(0, n)
    .map((i) => all[i]);
}
