# EventSleepFormer

Official implementation for **EventSleepFormer: An Efficient Framework for Event-Camera Sleep Activity, Movement, and Posture Recognition**.

EventSleepFormer adapts a frozen DINOv2 ViT-S/14 backbone with LoRA and models event-derived visual features with a causal temporal Transformer. This repository contains the **final configurations used for the paper** on EventSleep1 and EventSleep2. Development/tuning experiments are intentionally not included.

## Paper results

| Benchmark | Metric | EventSleepFormer |
|---|---:|---:|
| EventSleep1 | Accuracy | 0.8131 ± 0.0247 |
| EventSleep1 | Macro-F1 | **0.7683 ± 0.0097** |
| EventSleep1 | Balanced accuracy | 0.7735 ± 0.0054 |
| EventSleep2 | Movement macro-F1 | **0.4248 ± 0.0239** |
| EventSleep2 | Posture macro-F1 | 0.2427 ± 0.0150 |
| EventSleep2 | Joint macro-F1 | 0.3520 ± 0.0179 |

EventSleep2 values above are five-seed per-frame results under the timestamp-aligned protocol used in the paper. The proposed model improves movement recognition, while posture remains a limitation; the repository does not claim overall EventSleep2 superiority.

## Repository layout

```text
configs/                         final paper configurations
data/file_lists/                 exact Synapse files used by the paper
docs/DATA.md                     dataset download details
docs/REPRODUCIBILITY.md          complete from-scratch protocol
docs/RESULTS.md                  expected outputs and metrics
src/eventsleepformer.py          EventSleepFormer / EventSleep1 runtime
src/eventsleep2.py               EventSleep2 dual-head runtime
src/eventsleep2_timestamp_protocol_v1.py  timestamp alignment protocol
src/eventsleep_preprocess.py     EventSleep1 event preprocessing
scripts/download_data.py         obtain datasets from Synapse
scripts/preprocess_eventsleep1.py
scripts/preprocess_eventsleep2.py
scripts/train_eventsleep1.py
scripts/train_eventsleep2.py
scripts/summarize_results.py
```

## 1. Environment

The reported runs used Python 3.11.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an NVIDIA RTX A6000. For a matching CUDA 12.8 installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

EventSleep2 frame generation is packet-boundary dependent and uses a separate authors-matched preprocessing environment. Create it with:

```bash
conda env create -f environment-eventsleep2-preprocess.yml
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before running the experiments.

## 2. Download data

The datasets are **not redistributed here**. They are hosted by the EventSleep authors on Synapse project `syn54156328`. Create a Synapse account, obtain/accept access as required by the upstream dataset, create a personal access token, then:

```bash
export SYNAPSE_AUTH_TOKEN='YOUR_TOKEN'
python scripts/download_data.py --dataset all
```

The downloader uses exact Synapse entity IDs recorded for the paper rather than downloading unrelated files. See [docs/DATA.md](docs/DATA.md).

## 3. Reproduce EventSleep1

```bash
source .venv/bin/activate
python scripts/preprocess_eventsleep1.py
python scripts/train_eventsleep1.py --device cuda
```

The five fixed seeds are `13 37 73 101 137`. The subject-independent protocol uses Subjects 1–8 and 10 for training, Subject 11 for validation, and Subjects 9, 12, 13, and 14 for test.

## 4. Reproduce EventSleep2

First build the EventSleep2 frame cache and timestamp-aligned labels in the exact preprocessing environment:

```bash
conda activate eventsleep2-preprocess
python scripts/preprocess_eventsleep2.py
```

Then train/evaluate the final five-seed proposed model in the main environment:

```bash
conda deactivate
source .venv/bin/activate
python scripts/train_eventsleep2.py --device cuda
```

The rebuild must reproduce the stable EventSleep2 source-cache fingerprint
`0339ef7ad3c0e622e584762cc5a34e0679f26e7e9ffd2b6de511fec775e4905c`, the label-file SHA-256
`094f5ac69ae89680e81f48f901e43b4a5cbe6664f041be186c01e32ab5cc9f17`, and 5,153/1,055/10,536 train/validation/test claims.

The paper audit also records timestamp-bundle fingerprint `7e27838c...`. That hash includes provenance metadata containing generation-specific paths/timestamps, so a clean rebuild can have a different bundle fingerprint even when the source data, alignment rule, claims, and final model configuration are identical. The training wrapper binds the unchanged paper model configuration to the verified rebuilt bundle.

## 5. Summarize reproduced results

```bash
python scripts/summarize_results.py
```

Exact per-seed paper results are stored in `results/` for comparison.

## Data and upstream code

- EventSleep project and dataset: https://github.com/ropertunizar/EventSleep
- EventSleep2 code: https://github.com/ropertunizar/EventSleep2
- Dataset: https://www.synapse.org/Synapse:syn54156328/wiki/626824

The EventSleep datasets remain subject to their original access and usage terms. This repository does not redistribute them.

## Citation

If you use this repository or build on EventSleepFormer, please cite the following paper:

**Vidushika Neranjani Perera, Tai Dinh, and Yasuki Saito, “EventSleepFormer: An Efficient Framework for Event-Camera Sleep Activity, Movement, and Posture Recognition.”**

```bibtex
@article{perera2026eventsleepformer,
  title={EventSleepFormer: An Efficient Framework for Event-Camera Sleep Activity, Movement, and Posture Recognition},
  author={Perera, Vidushika Neranjani and Dinh, Tai and Saito, Yasuki},
  year={2026}
}
```

The BibTeX entry above intentionally contains only the currently available paper information. Venue, volume, pages, and other publication metadata can be added after publication.
