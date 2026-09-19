# Data acquisition

EventSleepFormer uses two releases from the EventSleep project. The data are not stored in this repository because the event recordings are large and are distributed by the original authors through Synapse.

## Official source

- Synapse project: `syn54156328`
- Dataset page: https://www.synapse.org/Synapse:syn54156328/wiki/626824
- EventSleep repository: https://github.com/ropertunizar/EventSleep
- EventSleep2 repository: https://github.com/ropertunizar/EventSleep2

A Synapse account and authentication token are required. Upstream access conditions or terms must be accepted by the user where Synapse requests them.

```bash
export SYNAPSE_AUTH_TOKEN='YOUR_TOKEN'
python scripts/download_data.py --dataset all
```

The script uses the Synapse Python client and the exact entity IDs in `data/file_lists/`.

## EventSleep1 files used

The paper uses the **995 released labeled clips** in the official `TRAIN` and `TEST` folders. The seven `TEST_FULL_SEQUENCE` recordings are not required for the reported EventSleep1 classifier and are therefore not downloaded by the reproducibility script.

Expected structure:

```text
data/raw/eventsleep1/
├── TRAIN/
│   ├── subject01_config1/
│   ├── ...
│   └── subject11_config3/
└── TEST/
    ├── subject09_config1/
    ├── ...
    └── subject14_config3/
```

Expected counts: 736 TRAIN clips + 259 TEST clips = 995 clips.

## EventSleep2 files used

The paper protocol requires `LabelsDatasetV2.csv` plus the 10 recordings in the fixed train/validation/test split. The unlabelled `subject03_seq02` recording is not required by the paper experiment and is not downloaded by default.

```text
data/raw/eventsleep2/TEST_FULL_SEQUENCE_V2/
├── LabelsDatasetV2.csv
├── subject01_seq01.aedat4
├── subject01_seq02.aedat4
├── subject01_seq03.aedat4
├── subject01_seq04.aedat4
├── subject02_seq01.aedat4
├── subject02_seq02.aedat4
├── subject02_seq03.aedat4
├── subject02_seq04.aedat4
├── subject02_seq05.aedat4
└── subject03_seq01.aedat4
```

Fixed split:

- train: `subject01_seq01`, `subject02_seq01`, `subject03_seq01`
- validation: `subject02_seq05`
- test: `subject01_seq02`–`04`, `subject02_seq02`–`04`

## Why exact file lists are committed

Synapse entity IDs provide an explicit acquisition manifest while keeping all large files outside GitHub. After EventSleep2 preprocessing, the repository additionally verifies the derived timestamp-aligned dataset fingerprint and the expected train/validation/test claim counts.

## Stable EventSleep2 data checks

A clean preprocessing run is accepted only when the stable source-cache fingerprint is `0339ef7ad3c0e622e584762cc5a34e0679f26e7e9ffd2b6de511fec775e4905c`, the label CSV SHA-256 is `094f5ac69ae89680e81f48f901e43b4a5cbe6664f041be186c01e32ab5cc9f17`, and the timestamp protocol emits 5,153/1,055/10,536 train/validation/test claims.
