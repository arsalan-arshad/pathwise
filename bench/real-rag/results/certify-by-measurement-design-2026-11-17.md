# Design note: certify what you serve, by measurement on fresh questions (T-029), 2026-11-17

Status: DESIGN NOTE plus a bench-only prototype (`bench/real-rag/certify-measure.ts`). Nothing in `src/` changed. Arsalan decides the direction; desktop reviews any `src/` change. Prior art: a held-out confidence bound on a fixed policy is ordinary practice (hold-out evaluation, one Wilson/Clopper-Pearson bound); nothing here is claimed as new.

## Why

T-025, T-027 and T-028 (SYNTHETIC with known truth, plus one REAL HotpotQA run) found that the planner's in-sample `slaCertified` (a) is optimistic at n <= 100 with 0/1 scores (17% to 40% false certification against 10% nominal) and (b) at larger n mostly follows the `assumeMonotone` flag, because unmeasured plans count as zero when it is false. A replay estimates the quality of a policy from data that were also used to build it. A measurement of the policy that is actually served, on questions it has not seen, avoids both problems: no threshold search, no replay, no credit for plans nobody ran.

## 1. API sketch (not implemented)

```ts
const cert = await planner.certify(freshSamples, { confidence?: number /* default sla.confidence ?? 0.9 */ });
// => { meanQuality, lowerBound, certified, n, costUsd, planShares: Record<planKey, share>, fellBackToSafe }
```

- It runs the CURRENT frozen policy (`decide()` on each fresh input, then the pipeline, then the scorer) and returns ONE Wilson lower bound of the mean quality at `confidence`; `certified = lowerBound >= sla.minQuality`. Prototype: `certifyByMeasurement` in `certify-measure.ts`.
- It does not search, does not learn, does not change `decide()` or the learned state. Its calls and dollars are reported (`n` runs of the plans the policy picks, not a full table).
- Relation to `report.slaCertified`: sit BESIDE it, not replace it. The replay result stays as a cheap early estimate and should be documented as such; `certify()` is the certificate. (Whether to rename or deprecate `slaCertified` is Arsalan's call.)
- Fresh questions: (a) a separate labelled set the planner never saw (reserve it BEFORE calibrating); (b) a time split of production traffic where the scorer runs on a sample (`feedbackRate`) after the policy is frozen. Both assume traffic like the calibration sample: drift is not covered, as with the current guarantee.
- Scorer error (`scorerErrorRate`): UNTESTED proposal: score each fresh question with the same `scoreVotes` majority, compute the bound on the scorer's own (observed) scale and map it back with the same `unbias` as `tuneThreshold()`, so the bound stays valid for a symmetric noisy judge. Not measured here.

## 2. Measurements

All below are the verbatim output of `npx tsx bench/real-rag/certify-measure.ts --reps 300 --real` (about 95 s, offline, no model call). Parts 1 to 3 are SYNTHETIC with known truth (the T-027 world family: 3 plans, costs 1/2/5, SLA 0.9, confidence 0.9, random features, 300 repetitions per cell, standard error about 1.7 points at 10% and 2.8 at 50%); part 4 is REAL (HotpotQA, 100 fully-tabled questions, 200 seeded splits, one seed, overlapping splits, test never read).

```
== certify-measure (SYNTHETIC unless marked REAL), SLA 0.9, confidence 0.9, 300 repetitions for part 3

-- part 1: fresh questions needed so that the Wilson lower bound clears the SLA when the OBSERVED mean equals the true mean (computed, not measured; about half the time more is needed). The bound depends on the mean only, so 0/1 and graded scores need the same n at the same observed mean.
  true quality SLA + 0.02 (0.92): n >= 370
  true quality SLA + 0.05 (0.95): n >= 60
  true quality SLA + 0.08 (0.98): n >= 24
  true quality SLA + 0.10 (1.00): n >= 15

-- part 2: a fixed plan certified by measurement on n fresh questions (iid scores with the stated true mean; 2000 repetitions per row)
true mean | scores |     20     30     50     75    100    150    200    300    500
0.88      | binary |   7.5%   1.8%   5.9%   4.7%   4.0%   2.3%   1.3%   0.9%   0.8%   (FALSE certification rate)
0.88      | graded |   0.0%   0.0%   0.0%   0.0%   0.0%   0.0%   0.0%   0.0%   0.0%   (FALSE certification rate)
0.90      | binary |  11.1%   5.0%  11.2%  11.6%  11.3%  10.1%   8.8%  10.1%   9.9%   (at the SLA: certification = false-certification)
0.90      | graded |   0.0%   0.1%   0.0%   0.1%   0.1%   0.1%   0.1%   0.3%   0.2%   (at the SLA: certification = false-certification)
0.95      | binary |  34.4%  21.9%  53.9%  69.1%  75.8%  86.8%  92.3%  98.5%  99.8%   (certification rate (power))
0.95      | graded |   2.0%  12.1%  39.1%  70.5%  86.4%  98.5%  99.9% 100.0% 100.0%   (certification rate (power))
0.98      | binary |  66.9%  54.6%  92.3%  97.8%  99.8% 100.0% 100.0% 100.0% 100.0%   (certification rate (power))
0.98      | graded |  40.1%  80.9%  98.9% 100.0% 100.0% 100.0% 100.0% 100.0% 100.0%   (certification rate (power))

-- part 3: composition (SYNTHETIC, known truth). Worlds: best plan truly SLA-0.02 (below), = SLA (at), SLA+0.05 (power). 'bad' = certified while the served policy's / the recommended plan's exact true quality < SLA. n_fresh = 100.
world  scores  n_cal | replay claim: cert / FALSE  calls | calibrate+certify(decide): cert / FALSE  calls | recommend+certify(plan): returned / cert / FALSE  calls
below  binary   100  |  15.7% /  15.7%    166 |   1.3% /   1.3%    266 |  34.0% /   1.0% /   1.0%     73
below  binary   300  |   3.3% /   3.3%    497 |   1.3% /   1.3%    597 |  24.0% /   0.7% /   0.7%     56
below  graded   100  |   0.0% /   0.0%    253 |   0.0% /   0.0%    353 |   8.3% /   0.0% /   0.0%     20
below  graded   300  |   0.0% /   0.0%    757 |   0.0% /   0.0%    857 |   0.3% /   0.0% /   0.0%      1
at     binary   100  |  19.0% /  19.0%    166 |   7.7% /   7.7%    266 |  55.0% /   5.0% /   0.0%    120
at     binary   300  |   4.7% /   4.7%    497 |   7.0% /   7.0%    597 |  54.7% /   3.7% /   0.0%    160
at     graded   100  |   1.3% /   1.3%    253 |   0.0% /   0.0%    353 |  54.0% /   0.0% /   0.0%    132
at     graded   300  |   0.0% /   0.0%    757 |   1.0% /   1.0%    857 |  48.3% /   0.0% /   0.0%    213
power  binary   100  |  32.3% /  10.7%    166 |  63.3% /   0.7%    266 |  96.3% /  70.3% /   0.0%    199
power  binary   300  |  16.7% /   7.7%    497 |  68.3% /   0.0%    597 |  98.7% /  73.7% /   0.0%    215
power  graded   100  |  35.0% /   0.0%    253 |  87.3% /   0.0%    353 | 100.0% /  88.0% /   0.0%    204
power  graded   300  |  15.3% /   0.0%    757 |  82.3% /   0.0%    857 | 100.0% /  84.3% /   0.0%    211
(calls = pipeline runs: replay claim = adaptive calibration; calibrate+certify = calibration + n_fresh served-policy runs; recommend+certify = racing calls + n_fresh runs of one plan. Standard error about 1.7 points at 10% and 2.8 at 50%.)

-- part 4 (REAL, offline, HotpotQA, 100 fully-tabled questions, 200 seeded splits 40/30/30, test never read): recommend() on A, certify that plan on B (one Wilson bound), miss = mean on C below the SLA
SLA 0.725: recommend returned a plan in 185/200; certified on B 11/200; of those, mean on C below the SLA 10/11; mean calls 406 (A=40 racing + 30 fresh)
SLA 0.687: recommend returned a plan in 198/200; certified on B 24/200; of those, mean on C below the SLA 7/24; mean calls 416 (A=40 racing + 30 fresh)
SLA 0.6 (LOOSE, harness check, not a product claim): recommend returned a plan in 200/200; certified on B 46/200; of those, mean on C below the SLA 9/46; mean calls 358 (A=40 racing + 30 fresh)
(compare T-025 with the replay planner: in-sample certified 3/200 and split-certified 18/200 at 0.725; 7 and 47 at 0.687; 3 and 124 at 0.60. 'Miss on C' is a 30-question mean: not comparable to 10%.)
```

### What n_fresh a plan 5 points above the SLA needs (SLA 0.9, confidence 0.9)

- Computed (part 1): the Wilson bound clears 0.9 when the observed mean is 0.95 from n = 60 up; for +0.02 it takes n = 370, for +0.08 n = 24 (computed, not measured; about half the time more is needed).
- Measured (part 2, iid scores with true mean 0.95): certification rate with 0/1 scores 21.9% (n=30), 53.9% (50), 69.1% (75), 75.8% (100), 86.8% (150), 92.3% (200), 98.5% (300); with graded (Beta) scores 12.1%, 39.1%, 70.5%, 86.4%, 98.5%, 99.9%, 100%. So about 100 fresh questions certify such a plan 76% (0/1) to 86% (graded) of the time, 150 give 87% to 99%, and 300 make it near certain.
- False certification of a fixed plan at or just below the SLA (part 2): 0.88 true: at most 7.5% (n=20), 0.8% to 4.0% from n=100; exactly at the SLA with 0/1 scores: 5.0% to 11.6% across n (about one in ten, as a 90% one-sided bound allows), graded about 0%.

### Replay claim vs calibrate + certify vs recommend + certify (part 3, n_fresh = 100)

- Power world (best plan truly 0.95): the replay claim certifies 32% (0/1, n=100) and 17% (n=300) and is falsely certified in 10.7% and 7.7% of runs; calibrate + certify(decide()) certifies 63% and 68% with 0.7% and 0.0% false; recommend + certify(plan) returns a plan in 96% to 100% of runs, certifies 70% to 88% and is falsely certified in 0.0%. Graded: replay 35% / 15% certified, calibrate + certify 87% / 82%, recommend + certify 88% / 84%.
- Below-SLA world: false certification replay 15.7% (0/1, n=100), 3.3% (n=300); calibrate + certify 1.3%; recommend + certify 1.0% and 0.7%.
- AT-the-SLA world: replay 19.0% / 4.7% false; calibrate + certify 7.7% / 7.0%; recommend + certify 0.0% (the recommended plan's exact true quality is just under the SLA only rarely: recommend returns a plan in about half the runs).
- Calls: the replay claim costs 166 (n=100) to 497 (n=300) adaptive calibration runs; calibrate + certify adds n_fresh = 100 runs of the plans the policy picks; recommend + certify spends racing calls plus 100 runs of ONE plan whenever a plan is recommended (mean calls per repetition 1 to 215 in the table, counting 0 when nothing was recommended). The dollar cost of the 100 fresh runs is 100 x the cost of the served plan (in this world 100 to 500 cost units), and it is spent per certificate, not per request.

### REAL, HotpotQA, T-025 splits (part 4): recommend() on A (40), certify the plan on B (30), mean on C (30)

At SLA 0.725 `recommend()` returned a plan in 185/200 splits and 11/200 were certified on B (the plan's true F1 is only about 1.3 points above the bar and n_B = 30), of which 10 had a C mean below the SLA; at 0.687: 198/200 returned, 24/200 certified, 7 of 24 below on C; LOOSE SLA 0.60 (harness check): 200/200, 46/200 certified, 9 of 46 below on C. The C-miss rates are a 30-question mean in a complementary pool of 100 (a B that looks good pushes C down) and are not comparable to 10%. At n_B = 30 certification is rare, as expected from the arithmetic above: on HotpotQA a certificate for the 0.738 plan against 0.725 needs far more fresh questions than the 100 that have a full table.

## 3. Costs and composition

- Cost: to certify you pay n_fresh runs of the served policy (about 60 to 150 for a plan 5 points above the SLA, more if the margin is smaller), plus nothing per request afterwards. In these worlds that is cheaper than a full-table replay (3n runs at n = 100 to 1000; counterfactual (a) in T-028 certified 64% at n=100 and 88% at n=1000 at 1.8x the adaptive runs) and it certifies the policy that is really served.
- Quality: for the certify-by-measurement rows, false certification was at most 11.6% in part 2 (a plan exactly at the SLA, 0/1 scores; nominal 10%), at most 7.5% when the plan was truly 0.88 (7.5% at n=20, 0.8% to 5.9% at larger n), and at most 7.7% in part 3 (calibrate + certify at the SLA) and 1.0% for recommend + certify; power is higher than the replay claim in the worlds where the best plan is above the SLA. The nominal rate is for ONE bound on iid fresh questions; it does not cover drift, a changed pipeline, or re-certifying repeatedly until it passes (a new certificate on new questions is a new test).
- Latency: none on `decide()`; `certify()` is a one-off batch of pipeline runs.
- Composition with `recommend()` (#42): `recommend(calibrationSamples)` finds the cheapest plan that meets the SLA on the observed mean WITHOUT credit tricks (same questions for every surviving plan); `certify(freshSamples)` then measures the policy actually served (the recommended plan as a fixed plan, or `decide()`) on questions neither step saw. Part 3 shows this composition (rows "recommend + certify") with 0.0% to 1.0% false certification in these worlds (0/1 and graded scores). The two use disjoint data on purpose: the plan is chosen on one set and certified on another, so the selection (winner's curse) cannot inflate the certificate.

## 4. README text that WOULD be true after this (not applied)

"`planner.certify(freshQuestions)` runs the policy you would serve on questions it has not seen and returns one Wilson lower bound of its quality. In synthetic checks with known truth the false-certification rate stayed near or below the nominal 10% and, for a plan 5 points above the SLA at 0.9 confidence, about 100 to 150 fresh questions certified it in 76% to 87% of runs (0/1 scores). On real HotpotQA with 30 fresh questions it almost never certified a plan 1.3 points above the bar. It covers traffic like the fresh sample, not drift. The built-in `slaCertified` is a replay estimate; in synthetic checks it was optimistic with a calibration set of 100 or fewer 0/1 scores and, at larger sizes, often false even when the SLA was met."

## 5. Limits

One synthetic family (3 plans, costs 1/2/5, linear difficulty dependence, SLA 0.9, confidence 0.9), random features, 0/1 and Beta scores; one real dataset with 100 fully-tabled questions; the real part has n_B = 30. `scorerErrorRate` handling is a proposal, untested. Fresh questions must be independent of calibration; a time split of live traffic adds drift risk that is not measured. Not implemented: no `src/` change, no API committed.
