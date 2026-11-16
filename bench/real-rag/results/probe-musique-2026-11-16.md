# Predictability probe on MuSiQue: 2026-11-16

Date: 2026-11-16. Commands: `caffeinate -i npm run bench:rag:fill -- --dataset musique --phase sample --n 300 --plans "model=small|thinking=low|topK=2,model=medium|thinking=high|topK=10" --yes-live` (run by Arsalan on his Mac; terminal did not run it), then `npm run bench:rag:probe -- --dataset musique --n 300` (probe output pasted by Arsalan) and `npm run bench:rag:probe -- --dataset musique --n 300 --breakdown` (offline, rerun by terminal from the response cache).

REAL data: MuSiQue dev (answerable) 300 questions sampled with seed 7, Gemini free-tier models at list prices, cheap = `model=small|thinking=low|topK=2`, max = `model=medium|thinking=high|topK=10`; one sample per (question, plan) pair; produced offline from the response cache; the cache is not in the repo.

Fill: 600 new calls (small 300, medium 300), several HTTP 503 retries and one 90 s timeout that all recovered, "remaining 0". List-price cost of the 600 calls about $1.37 (the ledger went from $2.295 to $3.667); billed $0 on the free tier. (The mean costs per query in the breakdown below times 300 questions give about $1.41, 3% above the ledger difference; the cause, e.g. how the ledger counts retried calls, was not checked.)

## Probe output (verbatim, pasted by Arsalan)

```
== probe: dataset musique, 300 questions (loaded), cheap model=small|thinking=low|topK=2, max model=medium|thinking=high|topK=10, k 15
label = cheap plan scores at least as high as the max plan: base rate 0.603 (181 of 300); identical score on both plans: 174
hashed text, leave-one-out kNN (k=15)          AUC 0.546   (shuffled labels: 5th to 95th percentile 0.427 to 0.561)
hashed text, similarity-weighted kNN (k=15)    AUC 0.554   (shuffled labels: 5th to 95th percentile 0.426 to 0.565)
question type (4hop vs 2hop vs 3hop), group rate AUC 0.549   (shuffled labels: 5th to 95th percentile 0.507 to 0.568)
question type, leave-one-out (artifact: ignore) AUC 0.259   (shuffled labels: 5th to 95th percentile 0.145 to 0.366)

(offline: no Gemini call was made. About 0.5 = the feature does not separate the two cases. One dataset, one seed, in-sample.)
```

## Breakdown (`--breakdown`, offline, rerun by terminal)

```
mean F1 / list-price cost per query: cheap 0.210 / $0.00026; max 0.510 / $0.00443
ties 174 (both exactly 0: 113; both equal and above 0: 61); cheap strictly better 7; max strictly better 119
per type: n | mean F1 cheap | mean F1 max | cheap strictly better | tie | max strictly better
  4hop    49 | 0.249 | 0.538 | 1 | 30 | 18
  2hop   165 | 0.222 | 0.506 | 3 | 102 | 60
  3hop    86 | 0.164 | 0.503 | 3 | 42 | 41
tie-free label ("max strictly better" among the 126 non-tied questions): 119 positive, 7 negative
  minority class has 7 questions (< 20): an AUC would not mean anything, so none is printed. The cheapest plan is simply much worse than the max plan on this set, so this pair cannot show routing headroom.
```

## Reading

- All three AUCs (0.546, 0.554, 0.549) are inside or at the edge of what shuffled labels give and below desktop's 0.65 gate (not a standard). Question text and hop count do not clearly predict which of these two plans a MuSiQue question needs.
- The pair is the problem as much as the features: the cheap plan averages F1 0.210 against 0.510 for the max plan, is strictly better on only 7 of 300 questions and strictly worse on 119. Of the 174 ties, 113 are both exactly 0 (both plans wrong) and 61 are equal above 0. With the cheap plan this much worse, this probe pair cannot show routing headroom; it does not show that routing is useless on MuSiQue, only that this probe cannot find it.
- Not measured: embeddings on MuSiQue (`probe --embed ollama` is wired, not run), intermediate plan pairs, other seeds. One dataset file, one seed, 300 questions.
