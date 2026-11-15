/** Pure helpers for probe.ts: AUC, leave-one-out kNN and group scores, and a label-permutation interval. No I/O. */
import { cosine } from "../../src/index.js";
import { mulberry32 } from "./analyze-core.js";

/** Area under the ROC curve (Mann-Whitney, ties count 0.5). null when one class is empty. */
export function auc(scores: number[], labels: number[]): number | null {
  let pos = 0;
  let neg = 0;
  let wins = 0;
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] === 1) {
      pos++;
      for (let j = 0; j < scores.length; j++) if (labels[j] === 0) wins += scores[i] > scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0;
    } else neg++;
  }
  return pos === 0 || neg === 0 ? null : wins / (pos * neg);
}

/** For each item, the indices of its k most similar OTHER items (cosine; ties by lower index) and their similarities. */
export function neighbours(vectors: number[][], k: number): Array<{ idx: number[]; sim: number[] }> {
  return vectors.map((v, i) => {
    const all = vectors.map((w, j) => ({ j, s: j === i ? -Infinity : cosine(v, w) })).filter((x) => x.j !== i);
    all.sort((a, b) => b.s - a.s || a.j - b.j);
    const top = all.slice(0, k);
    return { idx: top.map((x) => x.j), sim: top.map((x) => x.s) };
  });
}

/** Leave-one-out kNN score of each item: the mean label of its neighbours (similarity-weighted when `weighted`). */
export function knnScores(nb: Array<{ idx: number[]; sim: number[] }>, labels: number[], weighted: boolean): number[] {
  return nb.map(({ idx, sim }) => {
    if (!weighted) return idx.reduce((a, j) => a + labels[j], 0) / idx.length;
    const w = sim.map((s) => Math.max(s, 0));
    const tot = w.reduce((a, b) => a + b, 0);
    return tot === 0 ? idx.reduce((a, j) => a + labels[j], 0) / idx.length : idx.reduce((a, j, t) => a + w[t] * labels[j], 0) / tot;
  });
}

/**
 * Score from a categorical feature: the label rate of the item's group. `leaveOut` excludes the item itself, which for a
 * two-valued feature creates an ARTIFACT (removing an item moves its own group's rate away from its label, so the AUC falls
 * far below 0.5 with no information at all); the in-sample rate (default) has one parameter per group, so it is the fair
 * read for a feature with a handful of values.
 */
export function groupScores(groups: string[], labels: number[], leaveOut = false): number[] {
  const sum = new Map<string, number>();
  const cnt = new Map<string, number>();
  groups.forEach((g, i) => {
    sum.set(g, (sum.get(g) ?? 0) + labels[i]);
    cnt.set(g, (cnt.get(g) ?? 0) + 1);
  });
  if (!leaveOut) return groups.map((g) => (sum.get(g) ?? 0) / (cnt.get(g) ?? 1));
  return groups.map((g, i) => {
    const n = (cnt.get(g) ?? 0) - 1;
    return n <= 0 ? 0.5 : ((sum.get(g) ?? 0) - labels[i]) / n;
  });
}

/** 5th to 95th percentile of the AUC when the labels carry no information (shuffled), seeded. Shows what "about 0.5" means at this n. */
export function permutationBand(scoreOf: (labels: number[]) => number[], labels: number[], perms = 200, seed = 99): { lo: number; hi: number } | null {
  const rnd = mulberry32(seed);
  const vals: number[] = [];
  for (let p = 0; p < perms; p++) {
    const l = [...labels];
    for (let i = l.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [l[i], l[j]] = [l[j], l[i]];
    }
    const a = auc(scoreOf(l), l);
    if (a !== null) vals.push(a);
  }
  if (vals.length === 0) return null;
  vals.sort((a, b) => a - b);
  return { lo: vals[Math.floor(0.05 * (vals.length - 1))], hi: vals[Math.ceil(0.95 * (vals.length - 1))] };
}
