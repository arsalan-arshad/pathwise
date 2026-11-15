/** One place that turns `--dataset hotpot|musique` into a list of questions (shared by probe.ts and fill.ts). */
import { loadHotpot, sample, type Question } from "./dataset.js";
import { loadMusique, sampleFirst } from "./musique.js";

export type DatasetName = "hotpot" | "musique";

export function parseDataset(v: string | undefined): DatasetName {
  if (v === undefined || v === "hotpot") return "hotpot";
  if (v === "musique") return "musique";
  throw new Error(`unknown --dataset "${v}" (hotpot|musique)`);
}

/**
 * hotpot: the existing seeded sample, calibration then test (default 100 + 200 = 300 questions).
 * musique: the first `n` questions of a seeded shuffle of the local file (`n` defaults to 300).
 */
export async function questionsFor(dataset: DatasetName, o: { n?: number; calibration?: number; test?: number; seed?: number; musiqueFile?: string }): Promise<Question[]> {
  const seed = o.seed ?? 7;
  if (dataset === "musique") return sampleFirst(loadMusique(o.musiqueFile), o.n ?? 300, seed);
  const all = await loadHotpot();
  if (o.n !== undefined) return sample(all, o.n, 0, seed).calibration; // the first n of the same seeded shuffle
  const { calibration, test } = sample(all, o.calibration ?? 100, o.test ?? 200, seed);
  return [...calibration, ...test];
}
