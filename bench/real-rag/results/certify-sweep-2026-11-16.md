# Certification sweep (SYNTHETIC, known truth): 2026-11-16

Date: 2026-11-16. Commands: `npm run bench:rag:certify:sweep` (confidence 0.9, the planner's usual setting) and `npx tsx bench/real-rag/certify-sweep.ts --reps 300 --confidence 0.9909` (a Bonferroni-style stricter confidence, 1 - 0.1 / 11, for the 11 candidate thresholds tried between 0.90 and 1.00).

SYNTHETIC: generated workloads with KNOWN true quality per plan, no model call, no cache, no key. 300 independent repetitions per cell, 48 cells each. Nothing here is real data. The planner was not changed.

## Sweep 1: confidence 0.9 (nominal false-certification allowance 10%)

```
== certify sweep (SYNTHETIC, known truth): 300 independent repetitions per cell, 48 cells, SLA 0.9, confidence 0.9 (nominal false-certification allowance 10%), 10 parallel jobs, 365 s
plans cheap/mid/max, exact true means below: 0.550 / 0.800 / 0.880; at: 0.550 / 0.800 / 0.900; power: 0.550 / 0.800 / 0.950 (clipped success probabilities, so slightly off the nominal values); costs 1/2/5
false cert = certified AND the frozen policy's exact expected quality < SLA (checked against the known truth, not a finite test mean). split = one Wilson bound on 0.75 n fresh questions.

-- world BELOW: best plan truly SLA - 0.02
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |  36.3%        |   2.7%     |  36.3%         |   2.7%     |  63.7%
random       binary   100  |  16.7%        |   3.7%     |  16.7%         |   3.7%     |  83.3%
random       binary   300  |   3.0%        |   0.3%     |   3.0%         |   0.3%     |  97.0%
random       binary  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   100  |   0.3%        |   0.0%     |   0.3%         |   0.0%     |  99.7%
random       graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary    40  |  34.7%        |   1.0%     |  34.7%         |   1.0%     |  65.3%
informative  binary   100  |  16.7%        |   3.7%     |  16.7%         |   3.7%     |  83.3%
informative  binary   300  |   3.7%        |   2.3%     |   3.7%         |   2.3%     |  96.3%
informative  binary  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   100  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%

-- world AT: best plan truly = SLA
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |  40.3%        |   3.7%     |  40.3%         |   3.7%     |  59.7%
random       binary   100  |  20.0%        |  12.0%     |  20.0%         |  12.0%     |  80.0%
random       binary   300  |   5.0%        |   6.0%     |   5.0%         |   6.0%     |  95.0%
random       binary  1000  |   1.0%        |   9.3%     |   1.0%         |   9.3%     |  99.0%
random       graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   100  |   0.3%        |   0.0%     |   0.3%         |   0.0%     |  99.7%
random       graded   300  |   0.3%        |   0.0%     |   0.3%         |   0.0%     |  99.7%
random       graded  1000  |   0.0%        |   0.3%     |   0.0%         |   0.3%     | 100.0%
informative  binary    40  |  39.7%        |   2.7%     |  39.7%         |   2.7%     |  60.3%
informative  binary   100  |  18.7%        |  10.0%     |  18.7%         |  10.0%     |  81.3%
informative  binary   300  |   4.0%        |  10.0%     |   4.0%         |  10.0%     |  96.0%
informative  binary  1000  |   0.0%        |   8.3%     |   0.0%         |   8.3%     | 100.0%
informative  graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   100  |   0.7%        |   0.0%     |   0.7%         |   0.0%     |  99.3%
informative  graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%

-- world POWER: best plan truly SLA + 0.05
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |  51.0%        |  19.0%     |   6.7%         |   0.0%     |  49.0%
random       binary   100  |  31.7%        |  59.3%     |  10.0%         |   1.0%     |  68.3%
random       binary   300  |  13.3%        |  88.3%     |   6.3%         |   0.3%     |  86.7%
random       binary  1000  |   5.3%        |  96.7%     |   3.0%         |   0.0%     |  94.7%
random       graded    40  |  14.3%        |  12.0%     |   0.0%         |   0.0%     |  85.7%
random       graded   100  |  33.7%        |  65.0%     |   0.0%         |   0.0%     |  66.3%
random       graded   300  |  10.0%        |  99.0%     |   0.0%         |   0.0%     |  90.0%
random       graded  1000  |   1.7%        | 100.0%     |   0.0%         |   0.0%     |  98.3%
informative  binary    40  |  51.3%        |  20.3%     |  16.7%         |   0.3%     |  48.7%
informative  binary   100  |  27.3%        |  51.7%     |  20.3%         |   0.0%     |  72.7%
informative  binary   300  |   9.0%        |  88.0%     |   8.3%         |   0.7%     |  91.0%
informative  binary  1000  |   0.3%        |  99.7%     |   0.3%         |   0.0%     |  99.7%
informative  graded    40  |  13.0%        |  12.7%     |   0.0%         |   0.0%     |  87.0%
informative  graded   100  |  18.0%        |  58.0%     |   0.0%         |   0.0%     |  82.0%
informative  graded   300  |   2.3%        |  99.3%     |   0.0%         |   0.0%     |  97.7%
informative  graded  1000  |   0.0%        | 100.0%     |   0.0%         |   0.0%     | 100.0%

(SYNTHETIC. Read FALSE columns against the nominal 10%; with 300 repetitions a rate has a standard error of about 1.7 points at 10% and 2.8 points at 50%. Power rows: the certification rate when the best plan really is above the SLA.)
```

## Sweep 2: confidence 0.9909 (nominal allowance 0.91%)

```
== certify sweep (SYNTHETIC, known truth): 300 independent repetitions per cell, 48 cells, SLA 0.9, confidence 0.9909 (nominal false-certification allowance 0.91%), 10 parallel jobs, 639 s
plans cheap/mid/max, exact true means below: 0.550 / 0.800 / 0.880; at: 0.550 / 0.800 / 0.900; power: 0.550 / 0.800 / 0.950 (clipped success probabilities, so slightly off the nominal values); costs 1/2/5
false cert = certified AND the frozen policy's exact expected quality < SLA (checked against the known truth, not a finite test mean). split = one Wilson bound on 0.75 n fresh questions.

-- world BELOW: best plan truly SLA - 0.02
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       binary   100  |  13.0%        |   0.0%     |  13.0%         |   0.0%     |  87.0%
random       binary   300  |   3.0%        |   0.0%     |   3.0%         |   0.0%     |  97.0%
random       binary  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   100  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary   100  |  15.3%        |   0.0%     |  15.3%         |   0.0%     |  84.7%
informative  binary   300  |   3.7%        |   0.0%     |   3.7%         |   0.0%     |  96.3%
informative  binary  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   100  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%

-- world AT: best plan truly = SLA
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       binary   100  |  17.3%        |   0.7%     |  17.3%         |   0.7%     |  82.7%
random       binary   300  |   5.0%        |   0.0%     |   5.0%         |   0.0%     |  95.0%
random       binary  1000  |   1.0%        |   0.3%     |   1.0%         |   0.3%     |  99.0%
random       graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   100  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary   100  |  17.0%        |   0.7%     |  17.0%         |   0.7%     |  83.0%
informative  binary   300  |   4.0%        |   0.3%     |   4.0%         |   0.3%     |  96.0%
informative  binary  1000  |   0.0%        |   0.7%     |   0.0%         |   0.7%     | 100.0%
informative  graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   100  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   300  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded  1000  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%

-- world POWER: best plan truly SLA + 0.05
features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback
random       binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       binary   100  |  29.7%        |  11.3%     |   0.3%         |   0.0%     |  70.3%
random       binary   300  |  13.3%        |  57.3%     |   2.7%         |   0.0%     |  86.7%
random       binary  1000  |   5.3%        |  97.0%     |   2.3%         |   0.0%     |  94.7%
random       graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
random       graded   100  |   0.3%        |   0.0%     |   0.0%         |   0.0%     |  99.7%
random       graded   300  |   9.7%        |  64.7%     |   0.0%         |   0.0%     |  90.3%
random       graded  1000  |   1.7%        | 100.0%     |   0.0%         |   0.0%     |  98.3%
informative  binary    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  binary   100  |  27.0%        |   9.3%     |   4.3%         |   0.0%     |  73.0%
informative  binary   300  |   9.0%        |  51.3%     |   7.3%         |   0.0%     |  91.0%
informative  binary  1000  |   0.3%        |  99.3%     |   0.3%         |   0.0%     |  99.7%
informative  graded    40  |   0.0%        |   0.0%     |   0.0%         |   0.0%     | 100.0%
informative  graded   100  |   0.3%        |   0.0%     |   0.0%         |   0.0%     |  99.7%
informative  graded   300  |   2.3%        |  67.3%     |   0.0%         |   0.0%     |  97.7%
informative  graded  1000  |   0.0%        | 100.0%     |   0.0%         |   0.0%     | 100.0%

(SYNTHETIC. Read FALSE columns against the nominal allowance printed above; with 300 repetitions a rate has a standard error of about 1.7 points at 10% and 2.8 points at 50%. Power rows: the certification rate when the best plan really is above the SLA.)
```

## Reading (terminal's, SYNTHETIC only)

- With 0/1 scores the planner's in-sample certification is too easy at small calibration sizes when the best plan is truly below or at the SLA: false certification 34% to 36% (below) and 40% (at) at n=40, 17% and 19% to 20% at n=100, about 3% to 5% at n=300, about 0% to 1% at n=1000, against a nominal 10%. Near the nominal rate around n=100 to 300 for these workloads; above it at n=40 and n=100.
- With graded (Beta, concentration 4) scores the in-sample certification almost never fires at all (0% to 0.7%), so there is no false certification but also almost no power: the Wilson bound treats a graded score like a 0/1 verdict, which is conservative.
- The split protocol (one bound on fresh questions) stays at or below about 12% false certification in every cell and about 0% to 4% when the best plan is truly below the SLA, but at AT-the-SLA with 0/1 scores it reaches 6% to 12% at n >= 100 (a plan exactly at the SLA is certified about one time in ten, as expected from a 90% bound).
- Power (best plan truly SLA + 0.05): in-sample certification FALLS as n grows (0/1 random features: 51% at n=40, 32% at 100, 13% at 300, 5% at 1000) while the split certification rises (19%, 59%, 88%, 97%). In the power world the always-max fallback is 49% to 100% of runs, and the split protocol certifies mostly the always-max plan itself. So in this setup the in-sample machinery both certifies too easily when n is small and too rarely when n is large. The cause is NOT diagnosed (one hypothesis, untested: adaptive calibration measures the dearest plan selectively, and the replay credits unmeasured plans with the winner's score).
- Bonferroni-style stricter confidence (sweep 2) removes the n=40 false certifications (0%) but NOT the n=100 ones for 0/1 scores (13% to 15% in the below-SLA world; in the power world false certifications remain at up to 7%), so the optimism at n=100 is not only from trying about 11 thresholds. It also lowers power. (Hypothesis, untested: part of the bias comes from the replay itself.)
- Limits: one synthetic family (3 plans, costs 1/2/5, linear difficulty dependence, concentration 4 for graded scores), SLA 0.9 only, 2-dimensional synthetic features, 300 repetitions per cell (standard error about 1.7 points at 10% and 2.8 points at 50%). No real-data claim.
