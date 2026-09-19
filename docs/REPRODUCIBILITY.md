# Reproducing the paper from scratch

This document describes the final experiments reported in the EventSleepFormer paper. It intentionally omits development sweeps and abandoned configurations.

## A. Main environment

Reported server environment:

- Python 3.11.15
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- timm 1.0.28
- CUDA 12.8
- NVIDIA RTX A6000 (48 GB class)

Install the CUDA build of PyTorch first, then `requirements.txt` as shown in the root README. Different hardware can be used, but exact wall-clock times are not expected to match.

The DINOv2 ViT-S/14 pretrained weights are requested through `timm` on the first model run and may require network access to the upstream model cache.

## B. Download the exact data files

```bash
export SYNAPSE_AUTH_TOKEN='YOUR_TOKEN'
python scripts/download_data.py --dataset all
```

Expected acquisition inventory:

- EventSleep1: 995 labeled NPY clips.
- EventSleep2: 11 files: the label CSV plus 10 AEDAT4 recordings needed by the fixed split.

## C. EventSleep1 preprocessing

```bash
source .venv/bin/activate
python scripts/preprocess_eventsleep1.py
```

The release clips are converted/validated using two-polarity time surfaces and then expanded into the temporal representation used by EventSleepFormer:

- sequence length: 32
- channels: 2
- spatial resolution: 120 × 160
- history: 512 ms
- descriptors: active fraction, polarity balance, mean recency, spatial spread, frame-to-frame change.

The final temporal cache contains 995 sequences. The training wrapper uses the exact configuration in `configs/eventsleep1.json` and fixed seeds 13, 37, 73, 101, and 137:

```bash
python scripts/train_eventsleep1.py --device cuda
```

The protocol is subject-independent:

- optimization: Subjects 1–8 and 10 (666 clips)
- validation: Subject 11 (70 clips)
- test: Subjects 9, 12, 13, and 14 (259 clips).

No test clip is used for configuration selection.

## D. EventSleep2 preprocessing

The EventSleep2 source renderer is packet-boundary dependent. To recreate the frame cache used by the paper protocol, preprocessing is deliberately isolated in the authors-matched environment:

```bash
conda env create -f environment-eventsleep2-preprocess.yml
conda activate eventsleep2-preprocess
python scripts/preprocess_eventsleep2.py
```

This environment fixes:

- Python 3.9.16
- `aedat==2.0.2`
- NumPy 1.24.3
- pandas 1.5.3.

The script first builds the 320×240 EventSleep2 frame cache using the declared EventSleep2 representation, then creates the timestamp-aligned supervision used in this paper. A cached frame is supervised only when its anchor timestamp `t` satisfies:

```text
TSInit <= t <= TSEnd
```

No temporal shift, nearest-frame assignment, interpolation, tolerance, or label propagation is used. Overlapping source-row claims are retained separately, and annotated intervals with no cached anchor are omitted from supervision and recorded in the audit.

The resulting timestamp dataset must have:

- train: 5,153 claims
- validation: 1,055 claims
- test: 10,536 claims
- stable source-cache fingerprint: `0339ef7ad3c0e622e584762cc5a34e0679f26e7e9ffd2b6de511fec775e4905c`
- label-file SHA-256: `094f5ac69ae89680e81f48f901e43b4a5cbe6664f041be186c01e32ab5cc9f17`.

The archived paper run has timestamp-bundle fingerprint `7e27838c43449abcb72c06641031e46cb2b2140581c75a6cc6f55f68fedb1efa`. The original audit fingerprint deliberately included hashes of cache metadata files, some of which contain absolute paths and generation timestamps. Therefore it is a provenance identifier for the archived paper artifact, not a portable content hash that every fresh checkout can reproduce byte-for-byte. The public scripts verify the stable source-data fingerprint, label hash, protocol implementation, split, and claim counts, then bind the unchanged paper model configuration to the newly generated bundle fingerprint.

The protocol adapter file `src/eventsleep2_timestamp_protocol_v1.py` is deliberately kept byte-identical to the audited paper implementation because its SHA-256 is part of the dataset fingerprint.

## E. EventSleep2 training and evaluation

Return to the main training environment and run:

```bash
source .venv/bin/activate
python scripts/train_eventsleep2.py --device cuda
```

The final configuration in `configs/eventsleep2.json` uses:

- sequence length 64
- DINOv2 ViT-S/14 + LoRA
- temporal dimension 256
- 4 temporal blocks / 4 heads
- LoRA rank 8 in the last four attention blocks
- dropout 0.30
- learning rate 1.5e-4
- weight decay 0.05
- warmup 4 epochs
- batch size 4, accumulation 2
- movement/posture weights 0.6/0.4
- boundary weight 0.20
- maximum 50 epochs, patience 10.

The five seeds are 13, 37, 73, 101, and 137. Checkpoint selection is based on validation joint macro-F1. Each selected model is then evaluated on the fixed test recordings; an equal-probability five-member ensemble is also reported separately.

The fixed validation and test posture subsets contain no `LieDown` examples. Four-class posture macro-F1 still retains all four labels, so the unsupported class contributes zero under the fixed metric definition.

## F. Expected results

```bash
python scripts/summarize_results.py
```

Reference seed-level values are committed under `results/`; see `docs/RESULTS.md`. Small floating-point/runtime differences can occur across software or hardware stacks, but the data split, dataset fingerprint, configuration, and metric definitions should match exactly.
