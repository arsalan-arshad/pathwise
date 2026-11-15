# Embeddings vs hashed features (RUN B): 2026-11-15

Date: 2026-11-15. Command: `npm run bench:rag:embed-compare -- --model nomic-embed-text`.

REAL data: HotpotQA dev distractor, Gemini free-tier models at list prices, seed 7, 100 calibration / 200 test, one sample per (question, plan) pair; produced offline from the response cache; the cache is not in the repo.
Run by Arsalan on his Mac with a local Ollama (nomic-embed-text); terminal did not run it. Output pasted verbatim from `comms/attachments/run-b-output-2026-11-15.txt`.

```
RUN B real output, pasted verbatim by Arsalan (run on his Mac, Nov 15 2026 about 05:45 PKT, local Ollama, nomic-embed-text, no API key, no Gemini call).
Command: npm run bench:rag:embed-compare -- --model nomic-embed-text

== embed-compare: 100 calibration / 200 test questions, 12 plans, 5-fold cross-validation on calibration, SLA 90% of always-max; two planners: average-based (no confidence) and guaranteed (confidence 0.9)
always-max model=medium|thinking=high|topK=10: calibration F1 0.764, 0.00176 $/query -> SLA 0.687

-- average-based (no guarantee): calibration-only 5-fold cross-validation (100 held-out questions in total; NOT the test set)
features                       F1     SLA     $/query    certified    calib runs
hashed text                    0.653  MISSED  $0.00051           n/a       363
nomic-embed-text (Ollama)      0.654  MISSED  $0.00051           n/a       364
reference: always-max serves at 0.00176 $/query (calibration questions).
nomic-embed-text (Ollama) vs hashed: mixed or equal (cost and F1 move in different directions, or tie) (F1 +0.002, cost +$0.00000).

-- guaranteed (confidence 0.9): calibration-only 5-fold cross-validation (100 held-out questions in total; NOT the test set)
features                       F1     SLA     $/query    certified    calib runs
hashed text                    0.758  met     $0.00163     0/5 folds       363
nomic-embed-text (Ollama)      0.758  met     $0.00163     0/5 folds       364
reference: always-max serves at 0.00176 $/query (calibration questions).
nomic-embed-text (Ollama) vs hashed: mixed or equal (cost and F1 move in different directions, or tie) (F1 +0.000, cost +$0.00000).
warning: hashed text certified only 0/5 folds. A fold that cannot certify serves one safe plan for every request, which depends little on the features, so this row compares the featurizers only weakly (each fold calibrates on 80 questions). See the average-based table, which always routes with the features.
warning: nomic-embed-text (Ollama) certified only 0/5 folds. A fold that cannot certify serves one safe plan for every request, which depends little on the features, so this row compares the featurizers only weakly (each fold calibrates on 80 questions). See the average-based table, which always routes with the features.
(100 questions, one dataset, one seed: differences of a few points are noise. Not a significance test. The SLA is derived from the same 100 questions, including each fold's held-out 20, identically for every featurizer. Average-based mode can miss the SLA by design: read F1 and SLA together with cost.)

-- test set (200 questions)
test: not measured. 1317 of 2400 test pairs are missing and the router may pick any plan; no partial test numbers are shown.

-- embedding cost
nomic-embed-text (Ollama): 3 embed calls, 300 texts, 6426 tokens reported by the server, dimension 768, 1809 ms to embed 300 question texts. Cost $0 (local model).

(no Gemini call was made)
```
