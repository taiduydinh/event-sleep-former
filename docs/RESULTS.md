# Expected results

The files in this directory are compact reference outputs from the final experiments used in the paper. They are included so a new run can be checked without shipping model checkpoints or datasets.

## EventSleep1

Five-seed test results (seeds 13, 37, 73, 101, 137):

| Metric | Mean ± sample SD |
|---|---:|
| Accuracy | 0.8131 ± 0.0247 |
| Macro-F1 | 0.7683 ± 0.0097 |
| Balanced accuracy | 0.7735 ± 0.0054 |

The official test partition contains 259 clips. Exact per-seed values are in `results/eventsleep1_seed_metrics.csv`.

## EventSleep2

Five-seed per-frame test results:

| Metric | Mean ± sample SD |
|---|---:|
| Movement macro-F1 | 0.4248 ± 0.0239 |
| Posture macro-F1 | 0.2427 ± 0.0150 |
| Joint macro-F1 | 0.3520 ± 0.0179 |

Here `joint = 0.6 × movement + 0.4 × posture`. Exact per-seed values are in `results/eventsleep2_seed_metrics.csv`.

The separate equal-probability five-member ensemble gives movement/posture/joint macro-F1 of approximately 0.4564 / 0.2314 / 0.3664.

The proposed model improves per-frame EventSleep2 movement recognition relative to the paper's matched comparison models, but posture remains weaker than the adapted EvS2-net baseline. These results should not be summarized as overall EventSleep2 superiority.
