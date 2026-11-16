# Why does in-sample certification get rarer as calibration grows? 2026-11-17

Date: 2026-11-17. Command: `npx tsx bench/real-rag/certify-diagnose.ts --reps 100` (a throwaway diagnostic; not an npm script).

SYNTHETIC with KNOWN truth: the T-027 "power" world (0/1 scores, random features, 3 plans cost 1/2/5, SLA 0.9, confidence 0.9, cheap truly 0.55, mid 0.80, max truly 0.95 = SLA + 0.05), 100 independent repetitions per row. No model call, no cache, no key. The planner was not changed: private fields are read through a cast, and the counterfactuals use only the public `calibration: { strategy: "exhaustive" }` option and the planner's own `realized()`.

## Output (verbatim)

```
== certify diagnose (SYNTHETIC, known truth): power world, 0/1 scores, random features, SLA 0.9, confidence 0.9, best plan (max, cost 5) truly 0.9500000000000001 (cheap 0.55, mid 0.80), 100 independent repetitions per row

-- n = 100, calibration adaptive (default)
in-sample certified 36.0%; safe-plan policy 64.0%; assumeMonotone true 36.0% (mean violation rate 8.0%); pruned plans mean 0.00; calibration runs mean 166 of 300
threshold picked: mean 0.914; the planner's reported bound (certifiedQuality) mean 0.455 vs SLA 0.9; frozen policy's TRUE expected quality mean 0.937
replay at that threshold (100 records): replayed mean quality 0.496; chosen plan share cheap/mid/max 0.01/0.24/0.75; chosen plan MEASURED on that question 26.7%, winner-credited 26.6%, counted as ZERO (unmeasured) 46.6%
the true best plan (max) was measured on 17.7% of the replay questions
counterfactual (b) force the true best plan as the replay choice for every question: replayed mean 0.465, Wilson bound mean 0.431, bound >= SLA in 36.0%; with winner credit forced on (b'): mean 0.993, bound >= SLA in 100.0%

-- n = 100, calibration exhaustive (counterfactual a: every plan measured on every question)
in-sample certified 64.0%; safe-plan policy 36.0%; assumeMonotone true 0.0% (mean violation rate 9.5%); pruned plans mean 0.00; calibration runs mean 300 of 300
threshold picked: mean 0.940; the planner's reported bound (certifiedQuality) mean 0.893 vs SLA 0.9; frozen policy's TRUE expected quality mean 0.947
replay at that threshold (100 records): replayed mean quality 0.931; chosen plan share cheap/mid/max 0.01/0.14/0.86; chosen plan MEASURED on that question 100.0%, winner-credited 0.0%, counted as ZERO (unmeasured) 0.0%
the true best plan (max) was measured on 100.0% of the replay questions
counterfactual (b) force the true best plan as the replay choice for every question: replayed mean 0.954, Wilson bound mean 0.919, bound >= SLA in 80.0%; with winner credit forced on (b'): mean 0.954, bound >= SLA in 80.0%

-- n = 1000, calibration adaptive (default)
in-sample certified 4.0%; safe-plan policy 96.0%; assumeMonotone true 4.0% (mean violation rate 9.8%); pruned plans mean 0.00; calibration runs mean 1651 of 3000
threshold picked: mean 0.911; the planner's reported bound (certifiedQuality) mean 0.228 vs SLA 0.9; frozen policy's TRUE expected quality mean 0.947
replay at that threshold (1000 records): replayed mean quality 0.243; chosen plan share cheap/mid/max 0.03/0.21/0.76; chosen plan MEASURED on that question 25.8%, winner-credited 2.8%, counted as ZERO (unmeasured) 71.4%
the true best plan (max) was measured on 16.5% of the replay questions
counterfactual (b) force the true best plan as the replay choice for every question: replayed mean 0.189, Wilson bound mean 0.175, bound >= SLA in 4.0%; with winner credit forced on (b'): mean 0.990, bound >= SLA in 100.0%

-- n = 1000, calibration exhaustive (counterfactual a: every plan measured on every question)
in-sample certified 88.0%; safe-plan policy 12.0%; assumeMonotone true 0.0% (mean violation rate 9.8%); pruned plans mean 0.00; calibration runs mean 3000 of 3000
threshold picked: mean 0.933; the planner's reported bound (certifiedQuality) mean 0.920 vs SLA 0.9; frozen policy's TRUE expected quality mean 0.943
replay at that threshold (1000 records): replayed mean quality 0.931; chosen plan share cheap/mid/max 0.00/0.12/0.88; chosen plan MEASURED on that question 100.0%, winner-credited 0.0%, counted as ZERO (unmeasured) 0.0%
the true best plan (max) was measured on 100.0% of the replay questions
counterfactual (b) force the true best plan as the replay choice for every question: replayed mean 0.950, Wilson bound mean 0.940, bound >= SLA in 100.0%; with winner credit forced on (b'): mean 0.950, bound >= SLA in 100.0%

(SYNTHETIC, one world family, instrument only: the planner was not changed. Private fields are read through a cast.)
```

## Reading (terminal's, SYNTHETIC only, one world family)

1. **The in-sample claim tracks `assumeMonotone` exactly.** With adaptive calibration, in-sample certified equals "assumeMonotone true" in both rows: 36.0% = 36.0% at n=100 and 4.0% = 4.0% at n=1000. In this world the dearest plan fails 5% to 10% of the time by chance, so the measured monotonicity violation rate is about 8% to 10%, above the planner's 5% allowance; whether the flag comes out true is decided by a noisy estimate at small n (often true by luck) and by an accurate estimate at large n (false).
2. **When `assumeMonotone` is false, the replay counts unmeasured plans as zero.** `realized()` returns 0 for a plan that was not measured on a question unless it is credited with the winner's score (only when monotone). Adaptive calibration measures the dearest plan only where cheaper plans fail: the max plan was measured on only 17.7% (n=100) and 16.5% (n=1000) of the replay questions. At n=1000 the replay chose a plan that was unmeasured and counted as ZERO for 71.4% of the questions, so the replayed mean quality is 0.243 and the bound 0.228, while the frozen policy's TRUE expected quality is 0.947. The certification fails because the replay scores the policy as if it were wrong where it simply has no measurement. This confirms the terminal's hypothesis (2) and shows (1) is the reason it bites; (3) and (4) were not needed to explain this world (no plan was pruned; assumeMonotone is the switch).
3. **At small n the same switch makes the claim too easy.** When the flag comes out true (36% of runs at n=100) unmeasured pricier plans are credited with the winner's score (26.6% of replay questions at n=100), which overstates a plan that really fails about 5% to 10% of the time. This is the same mechanism as the T-025 / T-027 false certifications (a hypothesis here, not separately measured).
4. **Counterfactual (a), full-table calibration (every plan on every question):** certification becomes possible and rises with n: 64.0% at n=100 and 88.0% at n=1000, the replay chooses measured plans only (100%), the replayed mean 0.931 sits close to the policy's true quality 0.943 to 0.947, and the reported bound is 0.893 to 0.920. It restores certification at n=1000. Price: 3,000 runs instead of 1,651 at n=1000 (1.8x) and 300 instead of 166 at n=100.
5. **Counterfactual (b), force the true best plan as the replay choice:** with the planner's own `realized()` the bound is still 0.175 at n=1000 (certified 4.0%): forcing the right plan does NOT restore certification because the plan is unmeasured on 83.5% of the questions and counts as zero. With winner credit forced on (b') the bound clears the SLA in 100.0% of runs, but the replayed mean is 0.990 against a true 0.95 (over-credit, because the dearest plan fails about 5% to 10% of the time): this "restores" certification by being optimistic, and is not a safe fix. Only measuring the plan (a) restores a truthful certification here.

## Proposed fix directions (NOT implemented, not tested here; Arsalan decides, desktop reviews any change to `tuneThreshold()`, `realized()` or the bound)

- **Measure what the replay will choose.** When `assumeMonotone` is false, run the plan the replay picks on the questions where it is unmeasured (up to the full table in the worst case; this is counterfactual (a) in the limit), or verify a random subsample of them and use only measured questions for the bound. Price: more calibration calls, between 1x and 1.8x the adaptive cost in this world; latency of `decide()` none.
- **Use measured evidence for the credit.** Replace "credit with the winner's score" by the measured pass rate of the pricier plan on the audited queries (the audits already measure some pricier plans), with its own bound. Price: needs enough audits; untested.
- **Bound only measured replay questions** (and say that the claim is about the measured sample). Price: biased toward questions where cheap plans fail; needs an explicit statement.
- **Document** (already in the README Limit sentence): `slaCertified` can be false although the SLA is met, and the safe plan then runs.
