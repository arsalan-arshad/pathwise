/**
 * Zero-dependency text featurizer: hashed bag of words + bigrams + length signal.
 * Good enough to group similar queries. Swap in real embeddings for production.
 */
export function hashedTextFeaturizer(dim = 512) {
  return (input: unknown): number[] => {
    const text = typeof input === "string" ? input : JSON.stringify(input);
    const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    const v = new Array<number>(dim + 1).fill(0);
    const add = (term: string, w: number) => {
      v[fnv1a(term) % dim] += w;
    };
    for (let i = 0; i < tokens.length; i++) {
      add(tokens[i], 1);
      if (i > 0) add(`${tokens[i - 1]}_${tokens[i]}`, 0.5);
    }
    // Longer questions tend to be harder: keep length as an explicit signal.
    v[dim] = Math.log1p(tokens.length);
    return normalize(v);
  };
}

export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return dot; // vectors are pre-normalised
}

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / norm);
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
