/** HotpotQA answer normalisation and token F1 (same rules as the official evaluation script). */

export function normalize(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\b(a|an|the)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function f1(prediction: string, gold: string): number {
  const p = normalize(prediction);
  const g = normalize(gold);
  // Official script: yes/no/noanswer must match exactly.
  for (const special of ["yes", "no", "noanswer"]) {
    if ((p === special || g === special) && p !== g) return 0;
  }
  const pt = p.split(" ").filter(Boolean);
  const gt = g.split(" ").filter(Boolean);
  if (pt.length === 0 || gt.length === 0) return pt.length === gt.length ? 1 : 0;
  const counts = new Map<string, number>();
  for (const t of gt) counts.set(t, (counts.get(t) ?? 0) + 1);
  let common = 0;
  for (const t of pt) {
    const c = counts.get(t) ?? 0;
    if (c > 0) {
      common++;
      counts.set(t, c - 1);
    }
  }
  if (common === 0) return 0;
  const precision = common / pt.length;
  const recall = common / gt.length;
  return (2 * precision * recall) / (precision + recall);
}
