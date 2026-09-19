#!/usr/bin/env python3
"""Auditable EventSleep2 reproduction and EventSleepFormer experiments.

This module deliberately separates two protocols:

``published``
    The fixed EventSleep2 split used by Gallego et al. (CVIU 2026).  It is
    a necessary condition for comparison with Table 2. Data release,
    annotation alignment, preprocessing and metric definitions must also
    match; selecting this split alone does not establish exact reproduction.

``loso``
    The subject-independent EventSleepFormer protocol implemented by
    ``eventsleepformer.py``.  It is a stronger generalisation
    test, but its scores are not numerically comparable with Table 2.

The ``paper_declared`` profile uses the article's declared hyperparameters;
data alignment still requires independent verification. The
``released_code`` profile is an explicitly labelled diagnostic for observable
implementation choices in the authors' public repository.  Test recordings
are never instantiated by the tuning command.

An explicit ``--timestamp-dataset`` selects the separate timestamp_intervals_v1
annotation protocol. It reuses the fixed split and rendered arrays, but its
claims differ from the authors' released frame-index slices. Such results are
adapted-protocol results, even with the ``paper_declared`` parameter profile.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import dataclasses
import hashlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


VERSION = "1.1.6"
PROPOSED_TRAINING_REVISION = "amp_update_guard_and_partial_accumulation_v1"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESS_ROOT = PROJECT_ROOT.parent / "data" / "processed" / "eventsleep2_published"
PUBLISHED_PROFILES = ("paper_declared", "released_code")
AUTHORS_AEDAT_VERSION = "2.0.2"
AUTHORS_REPOSITORY_COMMIT = "496bf16825a50500d64c009c1a0617d55783e560"
AUTHORS_PYTHON_VERSION = "3.9.16"
AUTHORS_PREPROCESS_DEPENDENCIES = {
    "aedat": AUTHORS_AEDAT_VERSION,
    "numpy": "1.24.3",
    "pandas": "1.5.3",
}
PUBLISHED_TRAIN_RECORDINGS = ((1, 1), (2, 1), (3, 1))
PUBLISHED_VALIDATION_RECORDINGS = ((2, 5),)
PUBLISHED_TEST_RECORDINGS = ((1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4))
PUBLISHED_ALL_RECORDINGS = (
    *PUBLISHED_TRAIN_RECORDINGS,
    *PUBLISHED_VALIDATION_RECORDINGS,
    *PUBLISHED_TEST_RECORDINGS,
)
MOVEMENT_NAMES = ("HeadMove", "Hands2FaceHead", "RollLeft", "RollRight", "LegsShake", "ArmsShake")
STATIC_NAMES = ("LieLeft", "LieRight", "LieUp", "LieDown")
MOVEMENT_FLIP = {2: 3, 3: 2}
STATIC_FLIP = {0: 1, 1: 0}

# Values declared in the article.
PAPER_FRAME_INTERVAL_US = 150_000
PAPER_HISTORY_US = 512_000
PAPER_SENSOR_HEIGHT = 480
PAPER_SENSOR_WIDTH = 640
PAPER_OUTPUT_HEIGHT = 240
PAPER_OUTPUT_WIDTH = 320
PAPER_BASE_EPOCHS = 25
PAPER_FINETUNE_EPOCHS = 50
PAPER_LEARNING_RATE = 1e-4
PAPER_BATCH_SIZE = 32
PAPER_MOTION_LOSS_WEIGHT = 0.6
PAPER_STATIC_LOSS_WEIGHT = 0.4
PAPER_ENSEMBLE_SIZE = 5
# These values were used as strict inventory assertions by v1.1.2.  They are
# retained only as a non-authoritative comparison with that earlier audit.
# Neither aggregate is a model hyperparameter or a value declared in the
# paper.  The actual LabelsDatasetV2.csv selected by the user is the source of
# truth, and its SHA-256 plus derived inventory are recorded in every cache.
LEGACY_V112_REFERENCE_LABEL_ROWS = 1_022
LEGACY_V112_REFERENCE_LABEL_INDEX_EXTENT = 17_834
PUBLISHED_CACHE_VERSION = "evs2net_published_v3"

# These choices are present in the released code but not stated in the paper.
RELEASED_WEIGHT_DECAY = 0.01
RELEASED_MOTION_FREQUENCIES = np.asarray(
    [0.2986, 0.0906, 0.1431, 0.1768, 0.1331, 0.1579], dtype=np.float32
)
RELEASED_STATIC_FREQUENCIES = np.asarray(
    [0.2922, 0.3611, 0.3351, 0.0116], dtype=np.float32
)

SOTA_REFERENCE = {
    "source": "Gallego et al., EventSleep2, CVIU 264 (2026), Table 2",
    "doi": "10.1016/j.cviu.2025.104619",
    "paper_metric_definition": "top-1 sample accuracy",
    "released_evaluator_discrepancy": "also prints macro-average class recall under the label Accuracy",
    "motion_per_frame_accuracy": 0.46,
    "static_per_frame_accuracy": 0.43,
    "per_frame_average_accuracy": 0.45,
    "motion_per_gt_clip_accuracy": 0.50,
    "static_per_gt_clip_accuracy": 0.51,
    "per_gt_clip_average_accuracy": 0.50,
}

PRIMARY_SOURCES = {
    "paper": "https://doi.org/10.1016/j.cviu.2025.104619",
    "repository": "https://github.com/ropertunizar/EventSleep2",
    "base_training": "https://github.com/ropertunizar/EventSleep2/blob/main/train_ResNet-Ev2.py",
    "fine_tuning": "https://github.com/ropertunizar/EventSleep2/blob/main/ft-Ev2.py",
    "evaluation": "https://github.com/ropertunizar/EventSleep2/blob/main/test_ResNet-Ev2.py",
    "preprocessing": "https://github.com/ropertunizar/EventSleep2/blob/main/events_to_frames.py",
    "label_indexing": "https://github.com/ropertunizar/EventSleep2/blob/main/data_tools.py",
    "repository_commit_audited": AUTHORS_REPOSITORY_COMMIT,
}


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # CLI help and data preflight remain usable without torch.
    torch = None  # type: ignore[assignment]

    class _NNPlaceholder:
        Module = object

    nn = _NNPlaceholder()  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure_logging(path: Path, verbose: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def stable_hash(payload: Any, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for model commands; install requirements_eventsleep.txt")
    return torch


def resolve_device(requested: str) -> str:
    module = require_torch()
    if requested == "auto":
        return "cuda" if module.cuda.is_available() else "cpu"
    if requested == "cuda" and not module.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return requested


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    module = require_torch()
    module.manual_seed(seed)
    if module.cuda.is_available():
        module.cuda.manual_seed_all(seed)
    if deterministic:
        module.use_deterministic_algorithms(True, warn_only=False)
        module.backends.cudnn.benchmark = False
        module.backends.cudnn.deterministic = True
        if hasattr(module.backends, "cuda") and hasattr(module.backends.cuda, "enable_flash_sdp"):
            module.backends.cuda.enable_flash_sdp(False)
            module.backends.cuda.enable_mem_efficient_sdp(False)
            module.backends.cuda.enable_math_sdp(True)


def recording_key(subject: int, sequence: int) -> str:
    return f"subject{int(subject):02d}_seq{int(sequence):02d}"


def parse_recording_key(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"subject(\d+)_seq(\d+)", str(value))
    if not match:
        raise ValueError(f"Invalid EventSleep2 recording key: {value}")
    return int(match.group(1)), int(match.group(2))


def profile_settings(profile: str) -> dict[str, Any]:
    if profile not in PUBLISHED_PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose from {PUBLISHED_PROFILES}")
    if profile == "paper_declared":
        return {
            "profile": profile,
            "frame_pixel_rule": "most_recent_event_per_pixel_as_declared",
            "loss": "unweighted_cross_entropy_as_declared",
            "training_augmentation": "disabled_not_declared_in_paper",
            "validation_augmentation": False,
            "validation_batchnorm_mode": "eval",
            "optimizer_weight_decay": 0.0,
            "checkpoint_selection": "mean_motion_static_top1_accuracy_operational_interpretation",
            "deterministic_ensemble": "mean_member_softmax_probabilities",
            "eventsleep1_static_class_zero_loss": "included_as_required_by_four_class_task",
            "ensemble_base_initialization": "independent_full_two_stage_members",
            "clip_grouping": "explicit_ground_truth_segment",
            "laplace_clip_probabilities": "single_softmax_or_posterior_predictive",
            "scientific_role": "primary_paper_comparison",
        }
    return {
        "profile": profile,
        "frame_pixel_rule": "released_events_to_frame_indexing",
        "loss": "released_fixed_class_weighted_cross_entropy",
        "training_augmentation": "horizontal_flip_with_left_right_label_swap",
        "validation_augmentation": True,
        "validation_batchnorm_mode": "train",
        "optimizer_weight_decay": RELEASED_WEIGHT_DECAY,
        "checkpoint_selection": "movement_balanced_accuracy",
        "deterministic_ensemble": "softmax_of_mean_member_logits",
        "eventsleep1_static_class_zero_loss": "excluded_by_released_targets_st_gt_zero_mask",
        "ensemble_base_initialization": "single_shared_eventsleep1_checkpoint_as_released_code",
        "clip_grouping": "released_contiguous_equal_label_runs",
        "laplace_clip_probabilities": "released_second_softmax",
        "scientific_role": "diagnostic_released_code_replication_not_primary",
    }


def published_configuration(profile: str) -> dict[str, Any]:
    return {
        "pipeline_version": VERSION,
        "profile": profile,
        "profile_settings": profile_settings(profile),
        "architecture": {
            "encoder": "ImageNet-pretrained ResNet18",
            "input_channels": 2,
            "encoder_output": 1000,
            "motion_head": "Linear(1000, 6)",
            "static_head": "Linear(1000, 4)",
        },
        "representation": {
            "frame_interval_us": PAPER_FRAME_INTERVAL_US,
            "history_us": PAPER_HISTORY_US,
            "sensor_resolution": [PAPER_SENSOR_WIDTH, PAPER_SENSOR_HEIGHT],
            "output_resolution": [PAPER_OUTPUT_WIDTH, PAPER_OUTPUT_HEIGHT],
            "storage_dtype": "float32",
            "polarity_order": ["negative", "positive"],
            "frame_origin": "first raw event timestamp plus one 150 ms interval",
            "released_label_indexing": (
                "generate the full recording-relative stream, then apply released "
                "FrameIni/FrameEnd indices directly"
            ),
            "label_claim_order": "original LabelsDatasetV2.csv source-row order",
            "overlapping_label_policy": (
                "preserve every inclusive source-row slice as a separate supervised claim, "
                "matching the authors' GetLabelsFullSequencev21 loader"
            ),
            "timestamp_alignment_policy": (
                "audit released index/timestamp drift without label-driven realignment"
            ),
            "decoder": {
                "package": "aedat",
                "version": AUTHORS_AEDAT_VERSION,
                "reason": "exact EventSleepEnv.yml dependency; released renderer is packet-boundary-dependent",
            },
            "authors_preprocessing_environment": {
                "python": AUTHORS_PYTHON_VERSION,
                "dependencies": AUTHORS_PREPROCESS_DEPENDENCIES,
            },
        },
        "split": {
            "fine_tune": [recording_key(*value) for value in PUBLISHED_TRAIN_RECORDINGS],
            "validation": [recording_key(*value) for value in PUBLISHED_VALIDATION_RECORDINGS],
            "test": [recording_key(*value) for value in PUBLISHED_TEST_RECORDINGS],
        },
        "base_training": {
            "dataset": "EventSleep1-data original training/validation subjects",
            "epochs": PAPER_BASE_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": PAPER_LEARNING_RATE,
            "batch_size": PAPER_BATCH_SIZE,
            "weight_decay": profile_settings(profile)["optimizer_weight_decay"],
        },
        "fine_tuning": {
            "epochs": PAPER_FINETUNE_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": PAPER_LEARNING_RATE,
            "batch_size": PAPER_BATCH_SIZE,
            "motion_loss_weight": PAPER_MOTION_LOSS_WEIGHT,
            "static_loss_weight": PAPER_STATIC_LOSS_WEIGHT,
            "weight_decay": profile_settings(profile)["optimizer_weight_decay"],
        },
        "released_code_only": {"weight_decay": RELEASED_WEIGHT_DECAY},
        "laplace_ensemble": {
            "members": PAPER_ENSEMBLE_SIZE,
            "subset_of_weights": "all weights in each final head",
            "hessian_structure": "full",
            "predictive": "glm bridge",
        },
        "sources": PRIMARY_SOURCES,
    }


def load_companion_module() -> Any:
    path = PROJECT_ROOT / "eventsleepformer.py"
    if not path.exists():
        raise FileNotFoundError(f"Required companion pipeline not found: {path}")
    name = "eventsleepformer_companion"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import companion pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Exact EventSleep2 event-frame cache
# ---------------------------------------------------------------------------


REQUIRED_ES2_LABEL_COLUMNS = (
    "Subject", "Sequence", "FrameIni", "FrameEnd", "Label_mv", "Label_st",
    "nFrames", "TSInit", "TSEnd",
)


def locate_es2_labels(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"EventSleep2 label file does not exist: {explicit}")
        return explicit.resolve()
    matches = sorted(root.rglob("LabelsDatasetV2.csv"), key=lambda value: (len(value.parts), str(value)))
    if not matches:
        raise FileNotFoundError(f"LabelsDatasetV2.csv was not found below {root}")
    return matches[0].resolve()


def load_es2_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_ES2_LABEL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"EventSleep2 labels are missing columns: {missing}")
    output = frame[list(REQUIRED_ES2_LABEL_COLUMNS)].copy()
    output.insert(0, "source_row", np.arange(len(output), dtype=np.int64))
    for column in REQUIRED_ES2_LABEL_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    mandatory = [column for column in REQUIRED_ES2_LABEL_COLUMNS if column != "Label_st"]
    if output[mandatory].isna().to_numpy().any():
        raise ValueError("EventSleep2 label file has missing or non-numeric mandatory values")
    integer_columns = [*mandatory, "Label_st"]
    for column in integer_columns:
        present = output[column].notna()
        values = output.loc[present, column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise ValueError(f"EventSleep2 label column {column} contains non-integer values")
        if column != "Label_st":
            output[column] = output[column].astype(np.int64)
        else:
            output.loc[present, column] = np.rint(values)
    if (output["Subject"] <= 0).any() or (output["Sequence"] <= 0).any():
        raise ValueError("EventSleep2 Subject and Sequence identifiers must be positive")
    if (output["FrameIni"] < 0).any():
        raise ValueError("EventSleep2 FrameIni values must be non-negative")
    if (output["FrameEnd"] < output["FrameIni"]).any():
        raise ValueError("EventSleep2 labels contain FrameEnd < FrameIni")
    if (output["nFrames"] <= 0).any():
        raise ValueError("EventSleep2 nFrames values must be positive")
    if (output["TSEnd"] < output["TSInit"]).any():
        raise ValueError("EventSleep2 labels contain TSEnd < TSInit")
    calculated = output["FrameEnd"] - output["FrameIni"] + 1
    if not np.array_equal(calculated.to_numpy(), output["nFrames"].to_numpy()):
        raise ValueError("EventSleep2 labels violate nFrames = FrameEnd - FrameIni + 1")
    movement_values = set(output["Label_mv"].astype(int).unique().tolist())
    if not movement_values.issubset({-1, *range(6)}):
        raise ValueError(f"Unsupported EventSleep2 movement labels: {sorted(movement_values)}")
    _detect_static_label_encoding(output)
    return output


def _detect_static_label_encoding(labels: pd.DataFrame) -> str:
    present = labels.loc[labels["Label_st"].notna(), "Label_st"].astype(int)
    values = set(present.unique().tolist()) - {-1}
    if not values:
        return "absent"
    if values.issubset(set(range(6, 10))):
        return "global_class_ids_6_to_9"
    if values.issubset(set(range(4))):
        return "task_local_ids_0_to_3"
    raise ValueError(
        "EventSleep2 Label_st must use one consistent encoding: missing/-1 plus either "
        f"global IDs 6..9 or task-local IDs 0..3; observed {sorted(values)}"
    )


def _normalise_static_label(raw_value: Any, movement: int, encoding: str) -> int:
    # The authors' released full-sequence loader supervises posture only for
    # RollLeft/RollRight movement rows. Preserve that semantic while accepting
    # either of the two unambiguous numeric encodings found in derived releases.
    if movement not in (2, 3) or pd.isna(raw_value) or int(raw_value) == -1:
        return -100
    value = int(raw_value)
    if encoding == "global_class_ids_6_to_9" and value in range(6, 10):
        return value - 6
    if encoding == "task_local_ids_0_to_3" and value in range(4):
        return value
    raise ValueError(
        f"Static label {value} is incompatible with detected encoding {encoding!r}"
    )


def derive_release_inventory(
    labels: pd.DataFrame,
    required_recordings: Sequence[tuple[int, int]] = PUBLISHED_ALL_RECORDINGS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Derive release-specific counts without changing the paper protocol.

    Aggregate row/frame totals are properties of a particular label file, not
    model settings. They are therefore recorded and compared with the legacy
    v1.1.2 audit, but only missing fixed-split recordings are fatal here.

    The released ``GetLabelsFullSequencev21`` implementation iterates source
    rows and concatenates every inclusive ``FrameIni:FrameEnd+1`` slice. Thus
    overlapping intervals are distinct supervised claims, not a malformed
    cache. We preserve and audit that behavior rather than resolving labels or
    silently dropping duplicated physical frames.
    """
    required = tuple((int(subject), int(sequence)) for subject, sequence in required_recordings)
    required_set = set(required)
    file_pairs = list(
        zip(labels["Subject"].astype(int).tolist(), labels["Sequence"].astype(int).tolist())
    )
    file_recordings = sorted(set(file_pairs))
    missing = sorted(required_set - set(file_recordings))
    extra = sorted(set(file_recordings) - required_set)
    static_encoding = _detect_static_label_encoding(labels)
    rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for subject, sequence in required:
        key = recording_key(subject, sequence)
        selected = labels[
            (labels["Subject"] == subject) & (labels["Sequence"] == sequence)
        ].sort_values(["source_row"], kind="stable")
        if selected.empty:
            rows.append(
                {
                    "recording_key": key,
                    "label_rows": 0,
                    "released_minimum_frame_index": None,
                    "released_maximum_frame_index": None,
                    "released_label_index_extent": 0,
                    "claimed_index_positions": 0,
                    "unclaimed_index_positions": 0,
                    "multiply_claimed_index_positions": 0,
                    "annotation_overlap_extra_claims": 0,
                    "supervised_unique_physical_positions": 0,
                    "supervised_overlap_positions": 0,
                    "supervised_overlap_extra_claims": 0,
                    "conflicting_movement_overlap_positions": 0,
                    "conflicting_static_overlap_positions": 0,
                    "overlapping_row_pairs": 0,
                    "supervised_samples": 0,
                    "supervised_static_samples": 0,
                }
            )
            continue
        extent = int(selected["FrameEnd"].max()) + 1
        annotation_coverage = np.zeros(extent, dtype=np.uint16)
        supervised_coverage = np.zeros(extent, dtype=np.uint16)
        first_movement = np.full(extent, -100, dtype=np.int16)
        first_static = np.full(extent, -100, dtype=np.int16)
        movement_conflict = np.zeros(extent, dtype=bool)
        static_conflict = np.zeros(extent, dtype=bool)
        supervised_samples = 0
        supervised_static_samples = 0
        for row in selected.itertuples(index=False):
            start, stop = int(row.FrameIni), int(row.FrameEnd) + 1
            annotation_coverage[start:stop] += 1
            movement = int(row.Label_mv)
            if movement != -1:
                supervised_samples += stop - start
                supervised_coverage[start:stop] += 1
                previous_movement = first_movement[start:stop]
                seen_movement = previous_movement >= 0
                movement_conflict[start:stop] |= seen_movement & (
                    previous_movement != movement
                )
                previous_movement[~seen_movement] = movement
                static = _normalise_static_label(row.Label_st, movement, static_encoding)
                if static >= 0:
                    supervised_static_samples += stop - start
                    previous_static = first_static[start:stop]
                    seen_static = previous_static >= 0
                    static_conflict[start:stop] |= seen_static & (
                        previous_static != static
                    )
                    previous_static[~seen_static] = static

        records = list(selected.itertuples(index=False))
        pair_count = 0
        for left_position, left in enumerate(records):
            left_start, left_end = int(left.FrameIni), int(left.FrameEnd)
            left_movement = int(left.Label_mv)
            left_static = _normalise_static_label(
                left.Label_st, left_movement, static_encoding
            )
            for right in records[left_position + 1 :]:
                right_start, right_end = int(right.FrameIni), int(right.FrameEnd)
                overlap_start = max(left_start, right_start)
                overlap_end = min(left_end, right_end)
                if overlap_start > overlap_end:
                    continue
                right_movement = int(right.Label_mv)
                right_static = _normalise_static_label(
                    right.Label_st, right_movement, static_encoding
                )
                pair_count += 1
                overlap_rows.append(
                    {
                        "recording_key": key,
                        "left_source_row": int(left.source_row),
                        "right_source_row": int(right.source_row),
                        "left_csv_row": int(left.source_row) + 2,
                        "right_csv_row": int(right.source_row) + 2,
                        "left_frame_ini": left_start,
                        "left_frame_end": left_end,
                        "right_frame_ini": right_start,
                        "right_frame_end": right_end,
                        "overlap_frame_ini": overlap_start,
                        "overlap_frame_end": overlap_end,
                        "overlap_positions": overlap_end - overlap_start + 1,
                        "shared_boundary_only": overlap_start == overlap_end,
                        "left_movement_label": left_movement,
                        "right_movement_label": right_movement,
                        "movement_label_conflict": (
                            left_movement >= 0
                            and right_movement >= 0
                            and left_movement != right_movement
                        ),
                        "left_static_label": left_static,
                        "right_static_label": right_static,
                        "static_label_conflict": (
                            left_static >= 0
                            and right_static >= 0
                            and left_static != right_static
                        ),
                        "both_rows_supervised": (
                            left_movement >= 0 and right_movement >= 0
                        ),
                    }
                )

        annotation_overlap = annotation_coverage > 1
        supervised_overlap = supervised_coverage > 1
        rows.append(
            {
                "recording_key": key,
                "label_rows": int(len(selected)),
                "released_minimum_frame_index": int(selected["FrameIni"].min()),
                "released_maximum_frame_index": int(selected["FrameEnd"].max()),
                "released_label_index_extent": extent,
                "claimed_index_positions": int(np.count_nonzero(annotation_coverage)),
                "unclaimed_index_positions": int(
                    np.count_nonzero(annotation_coverage == 0)
                ),
                "multiply_claimed_index_positions": int(
                    np.count_nonzero(annotation_overlap)
                ),
                "annotation_overlap_extra_claims": int(
                    np.maximum(annotation_coverage.astype(np.int64) - 1, 0).sum()
                ),
                "supervised_unique_physical_positions": int(
                    np.count_nonzero(supervised_coverage)
                ),
                "supervised_overlap_positions": int(
                    np.count_nonzero(supervised_overlap)
                ),
                "supervised_overlap_extra_claims": int(
                    np.maximum(supervised_coverage.astype(np.int64) - 1, 0).sum()
                ),
                "conflicting_movement_overlap_positions": int(
                    np.count_nonzero(movement_conflict & supervised_overlap)
                ),
                "conflicting_static_overlap_positions": int(
                    np.count_nonzero(static_conflict & supervised_overlap)
                ),
                "overlapping_row_pairs": pair_count,
                "supervised_samples": supervised_samples,
                "supervised_static_samples": supervised_static_samples,
            }
        )
    recording_frame = pd.DataFrame(rows)
    overlap_columns = [
        "recording_key",
        "left_source_row",
        "right_source_row",
        "left_csv_row",
        "right_csv_row",
        "left_frame_ini",
        "left_frame_end",
        "right_frame_ini",
        "right_frame_end",
        "overlap_frame_ini",
        "overlap_frame_end",
        "overlap_positions",
        "shared_boundary_only",
        "left_movement_label",
        "right_movement_label",
        "movement_label_conflict",
        "left_static_label",
        "right_static_label",
        "static_label_conflict",
        "both_rows_supervised",
    ]
    overlap_frame = pd.DataFrame(overlap_rows, columns=overlap_columns)
    protocol_rows = int(recording_frame["label_rows"].sum())
    protocol_extent = int(recording_frame["released_label_index_extent"].sum())
    multiply_claimed = int(recording_frame["multiply_claimed_index_positions"].sum())
    annotation_extra_claims = int(
        recording_frame["annotation_overlap_extra_claims"].sum()
    )
    supervised_overlap_positions = int(
        recording_frame["supervised_overlap_positions"].sum()
    )
    supervised_overlap_extra_claims = int(
        recording_frame["supervised_overlap_extra_claims"].sum()
    )
    movement_conflicts = int(
        recording_frame["conflicting_movement_overlap_positions"].sum()
    )
    static_conflicts = int(
        recording_frame["conflicting_static_overlap_positions"].sum()
    )
    reference_match = bool(
        len(labels) == LEGACY_V112_REFERENCE_LABEL_ROWS
        and protocol_extent == LEGACY_V112_REFERENCE_LABEL_INDEX_EXTENT
    )
    warnings: list[str] = []
    if not reference_match:
        warnings.append(
            "Observed label inventory differs from the legacy v1.1.2 reference; this is "
            "non-fatal because those aggregates are not declared paper hyperparameters"
        )
    if multiply_claimed:
        warnings.append(
            "Released annotation intervals overlap; each inclusive source-row slice is "
            "preserved as a separate sample claim to match GetLabelsFullSequencev21"
        )
    if extra:
        warnings.append(
            "The label file contains recordings outside the fixed paper split; they are "
            "inventoried but excluded from all training, selection, and test partitions"
        )
    inventory = {
        "policy": (
            "derive counts from the selected LabelsDatasetV2.csv; keep the fixed paper split "
            "and fail only on structural/protocol incompatibility"
        ),
        "label_rows_in_file": int(len(labels)),
        "protocol_label_rows": protocol_rows,
        "recordings_in_label_file": [recording_key(*value) for value in file_recordings],
        "required_protocol_recordings": [recording_key(*value) for value in required],
        "missing_protocol_recordings": [recording_key(*value) for value in missing],
        "extra_label_recordings": [recording_key(*value) for value in extra],
        "static_label_encoding": static_encoding,
        "protocol_released_label_index_extent_total": protocol_extent,
        "protocol_multiply_claimed_index_positions": multiply_claimed,
        "protocol_annotation_overlap_extra_claims": annotation_extra_claims,
        "protocol_supervised_overlap_positions": supervised_overlap_positions,
        "protocol_supervised_overlap_extra_claims": supervised_overlap_extra_claims,
        "protocol_conflicting_movement_overlap_positions": movement_conflicts,
        "protocol_conflicting_static_overlap_positions": static_conflicts,
        "protocol_overlapping_row_pairs": int(len(overlap_frame)),
        "overlap_policy": (
            "preserve each original source-row inclusive slice as a separate supervised "
            "claim; never resolve conflicts from labels"
        ),
        "structurally_compatible": not missing,
        "legacy_v112_reference": {
            "authoritative": False,
            "declared_in_paper": False,
            "label_rows_in_file": LEGACY_V112_REFERENCE_LABEL_ROWS,
            "protocol_released_label_index_extent_total": (
                LEGACY_V112_REFERENCE_LABEL_INDEX_EXTENT
            ),
        },
        "legacy_v112_reference_comparison": {
            "exact_match": reference_match,
            "label_row_delta": int(len(labels) - LEGACY_V112_REFERENCE_LABEL_ROWS),
            "protocol_label_index_extent_delta": int(
                protocol_extent - LEGACY_V112_REFERENCE_LABEL_INDEX_EXTENT
            ),
            "effect_on_validation": "warning_only",
        },
        "warnings": warnings,
    }
    return inventory, recording_frame, overlap_frame


def discover_es2_recordings(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    pattern = re.compile(r"subject(?P<subject>\d+)_seq(?P<sequence>\d+)\.aedat4$", re.I)
    for path in sorted(root.rglob("*.aedat4")):
        match = pattern.search(path.name)
        if not match:
            continue
        key = recording_key(int(match.group("subject")), int(match.group("sequence")))
        if key in result:
            raise ValueError(f"Duplicate EventSleep2 recording {key}: {result[key]} and {path}")
        result[key] = path.resolve()
    return result


def _event_columns(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = events.dtype.names or ()
    lower = {name.lower(): name for name in names}

    def field(*candidates: str) -> str:
        for candidate in candidates:
            if candidate.lower() in lower:
                return lower[candidate.lower()]
        raise ValueError(f"Missing AEDAT event field {candidates}; found {list(names)}")

    x = np.asarray(events[field("x")], dtype=np.int64).reshape(-1)
    y = np.asarray(events[field("y")], dtype=np.int64).reshape(-1)
    timestamp = np.asarray(events[field("t", "timestamp", "ts")], dtype=np.int64).reshape(-1)
    polarity = np.asarray(events[field("on", "p", "polarity")], dtype=np.int64).reshape(-1)
    if len(timestamp) and np.any(np.diff(timestamp) < 0):
        order = np.argsort(timestamp, kind="stable")
        x, y, timestamp, polarity = x[order], y[order], timestamp[order], polarity[order]
    return x, y, timestamp, polarity


def require_authors_preprocess_environment() -> tuple[Any, dict[str, str]]:
    observed_python = ".".join(str(value) for value in sys.version_info[:3])
    observed_dependencies: dict[str, str] = {}
    missing: list[str] = []
    for package, required in AUTHORS_PREPROCESS_DEPENDENCIES.items():
        try:
            observed_dependencies[package] = distribution_version(package)
        except PackageNotFoundError:
            missing.append(f"{package}=={required}")
    mismatches = {
        package: {"required": required, "observed": observed_dependencies.get(package)}
        for package, required in AUTHORS_PREPROCESS_DEPENDENCIES.items()
        if observed_dependencies.get(package) != required
    }
    if observed_python != AUTHORS_PYTHON_VERSION or missing or mismatches:
        raise RuntimeError(
            "Published preprocessing must run in the authors-exact preprocessing environment "
            f"(Python {AUTHORS_PYTHON_VERSION}, dependencies {AUTHORS_PREPROCESS_DEPENDENCIES}); "
            f"observed Python {observed_python}, dependencies {observed_dependencies}, missing {missing}. "
            "Use environment_eventsleep2_published_preprocess_v1.1.4.yml. The released renderer is "
            "packet-boundary-dependent, so this environment cannot be substituted."
        )
    try:
        import aedat  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"Published preprocessing requires aedat=={AUTHORS_AEDAT_VERSION}, matching the "
            f"authors' EventSleepEnv.yml at commit {AUTHORS_REPOSITORY_COMMIT}"
        ) from exc
    return aedat, observed_dependencies


def iter_aedat_events(path: Path) -> Iterator[np.ndarray]:
    aedat, _ = require_authors_preprocess_environment()
    prior_max: int | None = None
    for packet in aedat.Decoder(str(path)):
        if "events" not in packet or len(packet["events"]) == 0:
            continue
        x, y, timestamp, polarity = _event_columns(packet["events"])
        if prior_max is not None and int(timestamp.min()) < prior_max:
            raise ValueError(f"AEDAT packets are not timestamp-monotonic in {path}")
        prior_max = int(timestamp.max())
        yield np.column_stack([x, y, timestamp, polarity]).astype(np.int64, copy=False)


def _released_temporal_suppression(events: np.ndarray, minimum_time_us: int) -> np.ndarray:
    if not len(events):
        return events
    reversed_events = np.asarray(events[::-1], dtype=np.int64).copy()
    original_time = reversed_events[:, 2].copy()
    reversed_events[:, 2] -= np.mod(reversed_events[:, 2], int(minimum_time_us))
    _, indices = np.unique(reversed_events, return_index=True, axis=0)
    selected = reversed_events[indices].copy()
    selected[:, 2] = original_time[indices]
    return selected


def _polarity_update(
    events: np.ndarray,
    height: int,
    width: int,
    released_indexing: bool,
) -> np.ndarray:
    output = np.zeros(height * width, dtype=np.float64)
    if not len(events):
        return output.reshape(height, width)
    ordered = events[np.lexsort((events[:, 2], events[:, 1], events[:, 0]))]
    coordinates, starts = np.unique(ordered[:, :2], return_index=True, axis=0)
    if released_indexing:
        selected_indices: list[np.ndarray] = []
        previous = -1
        for start in starts.astype(int):
            selected_indices.append(1 + np.arange(max(previous, start - 1, 0), start, dtype=np.int64))
            previous = int(start)
        indices = np.concatenate(selected_indices) if selected_indices else np.empty(0, dtype=np.int64)
        if len(indices):
            selected = ordered[indices]
            flat = np.ravel_multi_index(
                (selected[:, 1].astype(int), selected[:, 0].astype(int)), (height, width)
            )
            output[flat] = selected[:, 2]
    else:
        stops = np.concatenate([starts[1:], np.asarray([len(ordered)], dtype=np.int64)])
        for coordinate, start, stop in zip(coordinates, starts, stops):
            x, y = int(coordinate[0]), int(coordinate[1])
            output[y * width + x] = float(ordered[int(stop) - 1, 2])
    return output.reshape(height, width)


def published_time_surface(events: np.ndarray, profile: str) -> tuple[np.ndarray, int]:
    """Build one two-polarity surface exactly at the latest event in a chunk."""
    if not len(events):
        raise ValueError("Cannot create an event frame from an empty event set")
    suppressed = _released_temporal_suppression(events, PAPER_FRAME_INTERVAL_US)
    anchor = int(suppressed[:, 2].max())
    released = profile == "released_code"
    channels: list[np.ndarray] = []
    for polarity in (0, 1):  # paper/repository storage order: negative, positive
        selected = suppressed[suppressed[:, 3] == polarity]
        channel = _polarity_update(selected, PAPER_SENSOR_HEIGHT, PAPER_SENSOR_WIDTH, released)
        values = channel + (PAPER_HISTORY_US - anchor)
        values[values < 0] = 0
        values /= float(PAPER_HISTORY_US)
        channels.append(values[::2, ::2])
    frame = np.stack(channels).astype(np.float32)
    if frame.shape != (2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH):
        raise RuntimeError(f"Unexpected published frame shape: {frame.shape}")
    return frame, anchor


def _stream_published_packets(
    packets: Iterator[np.ndarray],
    source_name: str,
    pass_name: str,
    emit: Callable[[int, np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Recreate the authors' packet-triggered stream from the first raw event.

    The released ``*_nogt`` converter assumes timestamps have already been
    normalised and starts its first boundary at 150 ms.  AEDAT4 timestamps in
    the public release are absolute, so the equivalent origin is the first raw
    event timestamp plus 150 ms.  Long gaps advance the nominal boundary but
    produce at most one frame per packet, matching the released loop.
    """
    expected_max: int | None = None
    expected_min: int | None = None
    raw_initial: int | None = None
    observed_max: int | None = None
    buffer = np.empty((0, 4), dtype=np.int64)
    event_packets = 0
    event_count = 0
    trigger_packets = 0
    emitted_frames = 0
    multi_boundary_packets = 0
    skipped_nominal_boundaries = 0
    maximum_advances = 0

    for packet in packets:
        if not len(packet):
            continue
        packet = np.asarray(packet, dtype=np.int64)
        if packet.ndim != 2 or packet.shape[1] != 4:
            raise ValueError(f"Expected Nx4 event packets for {source_name}; found {packet.shape}")
        packet_min = int(packet[:, 2].min())
        packet_max = int(packet[:, 2].max())
        event_packets += 1
        event_count += int(len(packet))
        if raw_initial is None:
            raw_initial = packet_min
            expected_max = raw_initial + PAPER_FRAME_INTERVAL_US
            expected_min = expected_max - PAPER_HISTORY_US
        observed_max = packet_max if observed_max is None else max(observed_max, packet_max)
        assert expected_max is not None and expected_min is not None
        if event_packets % 10_000 == 0:
            logging.info(
                "%s %s: packets=%d events=%d frames=%d",
                source_name,
                pass_name,
                event_packets,
                event_count,
                emitted_frames,
            )
        if packet_max < expected_min:
            continue
        buffer = np.concatenate([buffer, packet], axis=0)
        maximum = int(buffer[:, 2].max())
        if maximum < expected_max:
            continue
        trigger_packets += 1
        frame_events = buffer[buffer[:, 2] < expected_max]
        advances = 0
        while maximum > expected_max:
            expected_max += PAPER_FRAME_INTERVAL_US
            advances += 1
        maximum_advances = max(maximum_advances, advances)
        if advances > 1:
            multi_boundary_packets += 1
            skipped_nominal_boundaries += advances - 1
        expected_min = expected_max - PAPER_HISTORY_US
        buffer = buffer[buffer[:, 2] > expected_min]
        if len(frame_events):
            if emit is not None:
                emit(emitted_frames, frame_events)
            emitted_frames += 1
    if raw_initial is None or observed_max is None:
        raise RuntimeError(f"No event packets were decoded from {source_name}")
    return {
        "raw_initial_timestamp_us": raw_initial,
        "raw_final_timestamp_us": observed_max,
        "event_packets": event_packets,
        "events": event_count,
        "trigger_packets": trigger_packets,
        "frames": emitted_frames,
        "multi_boundary_packets": multi_boundary_packets,
        "skipped_nominal_boundaries": skipped_nominal_boundaries,
        "maximum_advances_in_one_packet": maximum_advances,
    }


def _alignment_audit(labels: pd.DataFrame, timestamps: np.ndarray) -> dict[str, Any]:
    anchors = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if not len(anchors):
        raise RuntimeError("Cannot audit alignment for an empty rendered stream")
    if np.any(np.diff(anchors) < 0):
        raise RuntimeError("Rendered frame anchors are not timestamp-monotonic")
    released_initial = labels["FrameIni"].to_numpy(dtype=np.int64)
    released_final = labels["FrameEnd"].to_numpy(dtype=np.int64)
    timestamp_initial = np.searchsorted(
        anchors, labels["TSInit"].to_numpy(dtype=np.int64), side="left"
    ).astype(np.int64)
    timestamp_final = (
        np.searchsorted(anchors, labels["TSEnd"].to_numpy(dtype=np.int64), side="right") - 1
    ).astype(np.int64)
    initial_difference = timestamp_initial - released_initial
    final_difference = timestamp_final - released_final
    extent = int(released_final.max()) + 1
    all_valid = bool(int(released_initial.min()) >= 0 and extent <= len(anchors))
    exactly_aligned = int(
        np.sum((initial_difference == 0) & (final_difference == 0), dtype=np.int64)
    )

    def difference_summary(values: np.ndarray) -> dict[str, float | int]:
        return {
            "minimum": int(values.min()),
            "median": float(np.median(values)),
            "maximum": int(values.max()),
        }

    first_position = int(np.argmin(released_initial))
    last_position = int(np.argmax(released_final))
    status = (
        "released_indices_valid_and_timestamp_exact"
        if all_valid and exactly_aligned == len(labels)
        else "released_indices_valid_with_timestamp_drift"
        if all_valid
        else "released_indices_out_of_range"
    )
    return {
        "policy": "direct_released_FrameIni_FrameEnd_indexing_without_label_driven_realignment",
        "status": status,
        "all_released_indices_valid": all_valid,
        "rendered_frames": int(len(anchors)),
        "released_label_index_extent": extent,
        "released_minimum_frame_index": int(released_initial.min()),
        "released_maximum_frame_index": int(released_final.max()),
        "trailing_frame_margin": int(len(anchors) - extent),
        "timestamp_anchor_range_us": [int(anchors[0]), int(anchors[-1])],
        "first_annotation": {
            "timestamp_us": int(labels.iloc[first_position]["TSInit"]),
            "released_frame_index": int(released_initial[first_position]),
            "timestamp_derived_frame_index": int(timestamp_initial[first_position]),
            "difference": int(initial_difference[first_position]),
        },
        "last_annotation": {
            "timestamp_us": int(labels.iloc[last_position]["TSEnd"]),
            "released_frame_index": int(released_final[last_position]),
            "timestamp_derived_frame_index": int(timestamp_final[last_position]),
            "difference": int(final_difference[last_position]),
        },
        "label_rows": int(len(labels)),
        "exactly_timestamp_aligned_rows": exactly_aligned,
        "initial_index_difference": difference_summary(initial_difference),
        "final_index_difference": difference_summary(final_difference),
    }


def _cache_semantic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return representation-defining fields independent of wrapper version."""
    output = dict(payload)
    output.pop("pipeline_version", None)
    return output


def _render_published_recording(
    source: Path,
    labels: pd.DataFrame,
    frame_path: Path,
    timestamp_path: Path,
    profile: str,
    overwrite: bool,
) -> dict[str, Any]:
    released_index_extent = int(labels["FrameEnd"].max()) + 1
    _, preprocessing_versions = require_authors_preprocess_environment()
    decoder_version = preprocessing_versions["aedat"]
    source_digest = file_sha256(source)
    signature_payload = {
        "algorithm": "eventsleep2_authors_full_recording_stream_v2",
        "profile": profile,
        "source_name": source.name,
        "source_size": source.stat().st_size,
        "source_sha256": source_digest,
        "labels": stable_hash(labels.to_dict(orient="records"), 64),
        "frame_origin": "first_raw_event_timestamp_plus_frame_interval",
        "released_index_policy": "direct_FrameIni_FrameEnd_without_realignment",
        "frame_interval_us": PAPER_FRAME_INTERVAL_US,
        "history_us": PAPER_HISTORY_US,
        "resolution": [PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH],
        "dtype": "float32",
        "aedat_version": decoder_version,
        "preprocessing_dependencies": preprocessing_versions,
        "python_version": AUTHORS_PYTHON_VERSION,
        "authors_repository_commit": AUTHORS_REPOSITORY_COMMIT,
    }
    signature = stable_hash(signature_payload, 64)
    metadata_path = frame_path.with_suffix(".json")
    if frame_path.exists() and timestamp_path.exists() and metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
        recorded_frames = int(metadata.get("frames", -1))
        recorded_payload = metadata.get("signature_payload", {})
        recorded_signature_valid = bool(
            isinstance(recorded_payload, dict)
            and metadata.get("signature") == stable_hash(recorded_payload, 64)
        )
        semantic_payload_matches = bool(
            isinstance(recorded_payload, dict)
            and _cache_semantic_payload(recorded_payload)
            == _cache_semantic_payload(signature_payload)
        )
        valid = (
            recorded_signature_valid
            and semantic_payload_matches
            and frames.shape == (
                recorded_frames, 2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH
            )
            and len(timestamps) == recorded_frames
            and recorded_frames >= released_index_extent
            and str(frames.dtype) == "float32"
            and str(timestamps.dtype) == "int64"
            and not np.any(np.diff(timestamps) < 0)
            and metadata.get("alignment_audit", {}).get("all_released_indices_valid") is True
        )
        if valid:
            producer = str(metadata.get("pipeline_version", "unknown"))
            status = "reused" if producer == VERSION else "reused_compatible_previous_pipeline"
            return {
                **metadata,
                "cache_status": status,
                "cache_producer_pipeline_version": producer,
                "semantic_signature": stable_hash(
                    _cache_semantic_payload(recorded_payload), 64
                ),
            }
        raise RuntimeError(f"Stale or invalid cache exists for {source}; use --overwrite-cache")

    logging.info("Counting full recording-relative frames for %s", source.name)
    count_scan = _stream_published_packets(
        iter_aedat_events(source), source.name, "count-pass"
    )
    expected_frames = int(count_scan["frames"])
    if expected_frames < released_index_extent:
        raise RuntimeError(
            f"Full-recording render for {source} contains {expected_frames} frames, but released "
            f"FrameIni/FrameEnd indices require at least {released_index_extent}. The raw/label "
            "release pair is not reproducible under direct released indexing."
        )
    logging.info(
        "Counted %d frames for %s; released labels require indices [0, %d]. Rendering frames.",
        expected_frames,
        source.name,
        released_index_extent - 1,
    )

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_path.parent.mkdir(parents=True, exist_ok=True)
    partial_frames = frame_path.with_suffix(".partial.npy")
    partial_timestamps = timestamp_path.with_suffix(".partial.npy")
    frames = np.lib.format.open_memmap(
        partial_frames,
        mode="w+",
        dtype=np.float32,
        shape=(expected_frames, 2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH),
    )
    timestamps = np.lib.format.open_memmap(
        partial_timestamps, mode="w+", dtype=np.int64, shape=(expected_frames,)
    )

    def emit(position: int, frame_events: np.ndarray) -> None:
        if position >= expected_frames:
            raise RuntimeError(
                f"Render pass generated more frames than the count pass for {source}"
            )
        frame, anchor = published_time_surface(frame_events, profile)
        frames[position] = frame
        timestamps[position] = anchor

    try:
        render_scan = _stream_published_packets(
            iter_aedat_events(source), source.name, "render-pass", emit
        )
        if int(render_scan["frames"]) != expected_frames:
            raise RuntimeError(
                f"Non-deterministic decoder frame count for {source}: count pass "
                f"{expected_frames}, render pass {render_scan['frames']}"
            )
        for key in ("raw_initial_timestamp_us", "raw_final_timestamp_us", "event_packets", "events"):
            if render_scan[key] != count_scan[key]:
                raise RuntimeError(
                    f"Non-deterministic decoder observation for {source}: {key} changed from "
                    f"{count_scan[key]} to {render_scan[key]}"
                )
        frames.flush()
        timestamps.flush()
        alignment = _alignment_audit(labels, timestamps)
        if not bool(alignment["all_released_indices_valid"]):
            raise RuntimeError(
                f"Released label indices are outside the full render for {source}: {alignment}"
            )
        del frames, timestamps
        os.replace(partial_frames, frame_path)
        os.replace(partial_timestamps, timestamp_path)
    except Exception:
        for path in (partial_frames, partial_timestamps):
            if path.exists():
                path.unlink()
        raise
    metadata = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "signature": signature,
        "signature_payload": signature_payload,
        "frames": expected_frames,
        "shape": [expected_frames, 2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH],
        "stream_audit": render_scan,
        "alignment_audit": alignment,
        "aedat_version": decoder_version,
        "preprocessing_dependencies": preprocessing_versions,
        "python_version": AUTHORS_PYTHON_VERSION,
        "cache_producer_pipeline_version": VERSION,
        "semantic_signature": stable_hash(_cache_semantic_payload(signature_payload), 64),
        "cache_status": "created",
    }
    atomic_json(metadata_path, metadata)
    return metadata


def build_published_es2_cache(
    raw_root: Path,
    preprocess_root: Path,
    labels_path: Path | None,
    profile: str,
    overwrite: bool,
    recordings_limit: int | None = None,
) -> pd.DataFrame:
    labels_file = locate_es2_labels(raw_root, labels_path)
    labels = load_es2_labels(labels_file)
    recordings = discover_es2_recordings(raw_root)
    required = [recording_key(*value) for value in PUBLISHED_ALL_RECORDINGS]
    cache_root = preprocess_root / "eventsleep2" / PUBLISHED_CACHE_VERSION / profile
    labels_digest = file_sha256(labels_file)
    release_inventory, release_recordings, release_overlaps = derive_release_inventory(labels)
    release_inventory = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "labels_path": str(labels_file),
        "labels_sha256": labels_digest,
        **release_inventory,
    }
    atomic_json(cache_root / "release_inventory.json", release_inventory)
    atomic_csv(cache_root / "release_inventory_recordings.csv", release_recordings)
    atomic_csv(cache_root / "release_inventory_overlaps.csv", release_overlaps)
    release_recordings_digest = file_sha256(
        cache_root / "release_inventory_recordings.csv"
    )
    release_overlaps_digest = file_sha256(cache_root / "release_inventory_overlaps.csv")
    if not bool(release_inventory["structurally_compatible"]):
        raise RuntimeError(
            "EventSleep2 labels are structurally incompatible with the fixed paper protocol: "
            f"missing={release_inventory['missing_protocol_recordings']}"
        )
    for warning in release_inventory["warnings"]:
        logging.warning("EventSleep2 release inventory: %s", warning)
    missing = sorted(set(required) - set(recordings))
    if missing:
        raise FileNotFoundError(f"Published EventSleep2 recordings are missing: {missing}")
    if recordings_limit is not None:
        if int(recordings_limit) <= 0 or int(recordings_limit) > len(required):
            raise ValueError(
                f"--recordings-limit must be between 1 and {len(required)}, got {recordings_limit}"
            )
        required = required[: int(recordings_limit)]
    static_label_encoding = str(release_inventory["static_label_encoding"])
    all_samples: list[pd.DataFrame] = []
    recording_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    for position, key in enumerate(required, start=1):
        subject, sequence = parse_recording_key(key)
        selected = labels[(labels["Subject"] == subject) & (labels["Sequence"] == sequence)].copy()
        if selected.empty:
            raise RuntimeError(f"No label rows exist for required recording {key}")
        # Match GetLabelsFullSequencev21: concatenate each inclusive label-row
        # slice in the original CSV order.  Sorting by frame index would change
        # the released sample stream when intervals overlap or are nested.
        selected = selected.sort_values(["source_row"], kind="stable")
        frame_path = cache_root / "frames" / f"{key}.npy"
        timestamp_path = cache_root / "timestamps" / f"{key}.npy"
        logging.info("Published cache %d/%d: %s", position, len(required), key)
        cache_info = _render_published_recording(
            recordings[key], selected, frame_path, timestamp_path, profile, overwrite
        )
        logging.info(
            "Published cache %s status=%s producer_pipeline=%s frames=%d",
            key,
            cache_info["cache_status"],
            cache_info["cache_producer_pipeline_version"],
            int(cache_info["frames"]),
        )
        timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
        released_index_extent = int(selected["FrameEnd"].max()) + 1
        if len(timestamps) < released_index_extent:
            raise RuntimeError(
                f"Released indices for {key} require {released_index_extent} frames, but the "
                f"full recording cache contains only {len(timestamps)}"
            )
        sample_rows: list[dict[str, Any]] = []
        for row in selected.itertuples(index=False):
            movement = int(row.Label_mv)
            if movement == -1:
                continue
            if movement not in range(6):
                raise ValueError(f"Invalid movement label {movement} in {key}, source row {row.source_row}")
            static = _normalise_static_label(
                row.Label_st, movement, static_label_encoding
            )
            for frame_index in range(int(row.FrameIni), int(row.FrameEnd) + 1):
                sample_rows.append(
                    {
                        "sample_id": f"{key}_row{int(row.source_row):05d}_frame{frame_index:06d}",
                        "recording_key": key,
                        "subject": subject,
                        "sequence": sequence,
                        "source_row": int(row.source_row),
                        "gt_segment_id": f"{key}_row{int(row.source_row):05d}",
                        "array_index": frame_index,
                        "timestamp_us": int(timestamps[frame_index]),
                        "movement_label": movement,
                        "static_label": static,
                        "frame_path": str(frame_path.relative_to(preprocess_root)),
                    }
                )
        sample_frame = pd.DataFrame(sample_rows)
        all_samples.append(sample_frame)
        alignment = dict(cache_info["alignment_audit"])
        alignment_rows.append({"recording_key": key, **alignment})
        release_row = release_recordings.set_index("recording_key").loc[key]
        recording_rows.append(
            {
                "recording_key": key,
                "subject": subject,
                "sequence": sequence,
                "raw_path": str(recordings[key]),
                "frame_path": str(frame_path.relative_to(preprocess_root)),
                "timestamp_path": str(timestamp_path.relative_to(preprocess_root)),
                "frames": int(cache_info["frames"]),
                "released_label_index_extent": released_index_extent,
                "trailing_frame_margin": int(cache_info["frames"]) - released_index_extent,
                "alignment_status": str(alignment["status"]),
                "first_annotation_index_difference": int(
                    alignment["first_annotation"]["difference"]
                ),
                "last_annotation_index_difference": int(
                    alignment["last_annotation"]["difference"]
                ),
                "raw_sha256": str(cache_info["signature_payload"]["source_sha256"]),
                "semantic_signature": str(cache_info["semantic_signature"]),
                "cache_producer_pipeline_version": str(
                    cache_info["cache_producer_pipeline_version"]
                ),
                "supervised_samples": len(sample_frame),
                "supervised_static_samples": int((sample_frame["static_label"] >= 0).sum()),
                "multiply_claimed_index_positions": int(
                    release_row["multiply_claimed_index_positions"]
                ),
                "annotation_overlap_extra_claims": int(
                    release_row["annotation_overlap_extra_claims"]
                ),
                "supervised_overlap_positions": int(
                    release_row["supervised_overlap_positions"]
                ),
                "supervised_overlap_extra_claims": int(
                    release_row["supervised_overlap_extra_claims"]
                ),
                "conflicting_movement_overlap_positions": int(
                    release_row["conflicting_movement_overlap_positions"]
                ),
                "conflicting_static_overlap_positions": int(
                    release_row["conflicting_static_overlap_positions"]
                ),
                "overlapping_row_pairs": int(release_row["overlapping_row_pairs"]),
                "cache_status": cache_info["cache_status"],
            }
        )
    manifest = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    rendered_frames = int(sum(int(row["frames"]) for row in recording_rows))
    released_index_extent_total = int(
        sum(int(row["released_label_index_extent"]) for row in recording_rows)
    )
    inventory_extent_by_key = release_recordings.set_index("recording_key")[
        "released_label_index_extent"
    ].astype(int)
    expected_selected_extent = int(inventory_extent_by_key.loc[required].sum())
    if released_index_extent_total != expected_selected_extent:
        raise RuntimeError(
            "Internal label-inventory inconsistency: rendered recording extent total "
            f"{released_index_extent_total}, label-derived total {expected_selected_extent}"
        )
    if not all(bool(row["all_released_indices_valid"]) for row in alignment_rows):
        raise RuntimeError("At least one recording has released frame indices outside its cache")
    if manifest.empty:
        multiply_claimed_positions = 0
        overlap_extra_sample_claims = 0
        physical_claim_duplicate_rows = 0
    else:
        claim_sizes = manifest.groupby(
            ["recording_key", "array_index"], sort=False
        ).size()
        overlapping_claim_sizes = claim_sizes[claim_sizes > 1]
        multiply_claimed_positions = int(len(overlapping_claim_sizes))
        overlap_extra_sample_claims = int((overlapping_claim_sizes - 1).sum())
        physical_claim_duplicate_rows = int(overlapping_claim_sizes.sum())
    expected_overlap_positions = int(
        release_recordings.set_index("recording_key")
        .loc[required, "supervised_overlap_positions"]
        .astype(int)
        .sum()
    )
    expected_extra_claims = int(
        release_recordings.set_index("recording_key")
        .loc[required, "supervised_overlap_extra_claims"]
        .astype(int)
        .sum()
    )
    if (
        multiply_claimed_positions != expected_overlap_positions
        or overlap_extra_sample_claims != expected_extra_claims
    ):
        raise RuntimeError(
            "Internal overlap-accounting inconsistency: manifest has "
            f"positions={multiply_claimed_positions}, extra_claims={overlap_extra_sample_claims}; "
            "label inventory expects "
            f"positions={expected_overlap_positions}, extra_claims={expected_extra_claims}"
        )
    if multiply_claimed_positions:
        logging.warning(
            "Preserving %d overlapping physical positions as %d extra sample claims "
            "in original label-row order",
            multiply_claimed_positions,
            overlap_extra_sample_claims,
        )
    fingerprint_payload = {
        "cache_version": PUBLISHED_CACHE_VERSION,
        "profile": profile,
        "labels_sha256": labels_digest,
        "release_inventory_recordings_sha256": release_recordings_digest,
        "release_inventory_overlaps_sha256": release_overlaps_digest,
        "paper_split": published_configuration(profile)["split"],
        "label_application": {
            "order": "original_csv_source_row",
            "interval": "inclusive_FrameIni_through_FrameEnd",
            "overlap_policy": "preserve_each_source_row_claim",
        },
        "recordings": [
            {
                key: row[key]
                for key in (
                    "recording_key",
                    "raw_sha256",
                    "semantic_signature",
                    "frames",
                    "released_label_index_extent",
                    "supervised_samples",
                    "supervised_static_samples",
                    "multiply_claimed_index_positions",
                    "annotation_overlap_extra_claims",
                    "supervised_overlap_positions",
                    "supervised_overlap_extra_claims",
                    "conflicting_movement_overlap_positions",
                    "conflicting_static_overlap_positions",
                    "overlapping_row_pairs",
                )
            }
            for row in recording_rows
        ],
    }
    dataset_fingerprint = stable_hash(fingerprint_payload, 64)
    atomic_csv(cache_root / "samples.csv", manifest)
    atomic_csv(cache_root / "recordings.csv", pd.DataFrame(recording_rows))
    atomic_json(
        cache_root / "alignment_audit.json",
        {
            "created_utc": utc_now(),
            "pipeline_version": VERSION,
            "policy": "direct released indices; timestamp drift is reported and never corrected",
            "recordings": alignment_rows,
        },
    )
    atomic_json(
        cache_root / "completed_successfully.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "profile": profile,
            "partial": recordings_limit is not None,
            "recordings": len(required),
            "supervised_samples": len(manifest),
            "rendered_frame_total": rendered_frames,
            "released_label_index_extent_total": released_index_extent_total,
            "release_inventory": release_inventory,
            "dataset_fingerprint": dataset_fingerprint,
            "dataset_fingerprint_payload": fingerprint_payload,
            "physical_frame_claim_duplicate_rows": physical_claim_duplicate_rows,
            "multiply_claimed_physical_frame_positions": multiply_claimed_positions,
            "overlap_extra_sample_claims": overlap_extra_sample_claims,
            "overlap_policy": (
                "preserve each original source-row inclusive slice as a separate sample claim"
            ),
            "all_released_indices_valid": all(
                bool(row["all_released_indices_valid"]) for row in alignment_rows
            ),
            "cache_version": PUBLISHED_CACHE_VERSION,
            "published_configuration": published_configuration(profile),
            "labels_path": str(labels_file),
            "labels_sha256": labels_digest,
            "release_inventory_recordings_sha256": release_recordings_digest,
            "release_inventory_overlaps_sha256": release_overlaps_digest,
            "aedat_version": AUTHORS_AEDAT_VERSION,
            "preprocessing_dependencies": AUTHORS_PREPROCESS_DEPENDENCIES,
            "python_version": AUTHORS_PYTHON_VERSION,
        },
    )
    return manifest


def published_manifest_path(preprocess_root: Path, profile: str) -> Path:
    return preprocess_root / "eventsleep2" / PUBLISHED_CACHE_VERSION / profile / "samples.csv"


def published_dataset_provenance(preprocess_root: Path, profile: str) -> dict[str, Any]:
    cache_root = published_manifest_path(preprocess_root, profile).parent
    marker_path = cache_root / "completed_successfully.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing published cache completion marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    fingerprint_payload = marker.get("dataset_fingerprint_payload")
    fingerprint = marker.get("dataset_fingerprint")
    if not isinstance(fingerprint_payload, dict) or not isinstance(fingerprint, str):
        raise RuntimeError(
            f"Cache marker predates dataset fingerprinting at {marker_path}; rerun "
            "prepare-published with v1.1.4"
        )
    if stable_hash(fingerprint_payload, 64) != fingerprint:
        raise RuntimeError(f"Dataset fingerprint integrity check failed at {marker_path}")
    inventory_path = cache_root / "release_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing release inventory: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("labels_sha256") != marker.get("labels_sha256"):
        raise RuntimeError("Release inventory and completion marker label hashes disagree")
    if marker.get("release_inventory") != inventory:
        raise RuntimeError(
            "Release inventory file and completion marker contain different inventories"
        )
    return {
        "cache_version": marker.get("cache_version"),
        "cache_pipeline_version": marker.get("pipeline_version"),
        "profile": marker.get("profile"),
        "dataset_fingerprint": fingerprint,
        "labels_path": marker.get("labels_path"),
        "labels_sha256": marker.get("labels_sha256"),
        "rendered_frame_total": marker.get("rendered_frame_total"),
        "released_label_index_extent_total": marker.get(
            "released_label_index_extent_total"
        ),
        "supervised_samples": marker.get("supervised_samples"),
        "multiply_claimed_physical_frame_positions": marker.get(
            "multiply_claimed_physical_frame_positions"
        ),
        "overlap_extra_sample_claims": marker.get("overlap_extra_sample_claims"),
        "physical_frame_claim_duplicate_rows": marker.get(
            "physical_frame_claim_duplicate_rows"
        ),
        "overlap_policy": marker.get("overlap_policy"),
        "release_inventory": inventory,
    }


def load_published_manifest(preprocess_root: Path, profile: str) -> pd.DataFrame:
    path = published_manifest_path(preprocess_root, profile)
    if not path.exists():
        raise FileNotFoundError(f"Missing published EventSleep2 cache manifest: {path}; run prepare-published")
    marker_path = path.parent / "completed_successfully.json"
    if not marker_path.exists():
        raise FileNotFoundError(f"Missing published cache completion marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("partial") is not False:
        raise RuntimeError(
            f"Published cache is marked partial at {marker_path}; rebuild without --recordings-limit"
        )
    if marker.get("profile") != profile:
        raise RuntimeError(f"Published cache profile marker mismatch at {marker_path}")
    if marker.get("cache_version") != PUBLISHED_CACHE_VERSION:
        raise RuntimeError(f"Published cache version marker mismatch at {marker_path}")
    if marker.get("all_released_indices_valid") is not True:
        raise RuntimeError(f"Published cache does not validate all released indices at {marker_path}")
    published_dataset_provenance(preprocess_root, profile)
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "sample_id", "recording_key", "subject", "sequence", "source_row", "gt_segment_id",
        "array_index", "timestamp_us", "movement_label", "static_label", "frame_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Published cache manifest is missing columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Published cache manifest contains duplicate sample IDs")
    movement = pd.to_numeric(frame["movement_label"], errors="coerce")
    static = pd.to_numeric(frame["static_label"], errors="coerce")
    if movement.isna().any() or not movement.astype(int).isin(range(6)).all():
        raise ValueError("Published cache manifest contains invalid movement labels")
    if static.isna().any() or not static.astype(int).isin([-100, 0, 1, 2, 3]).all():
        raise ValueError("Published cache manifest contains invalid static labels")
    return frame


def timestamp_adapter() -> Any:
    name = "eventsleep2_timestamp_protocol_v1"
    if name not in sys.modules:
        path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load timestamp adapter: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_experiment_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    root, profile = args.preprocess_root.resolve(), str(args.profile)
    if getattr(args, "timestamp_dataset", None) is None:
        return load_published_manifest(root, profile), published_dataset_provenance(root, profile)
    frame, marker = timestamp_adapter().load_bundle(
        sys.modules[__name__], args.timestamp_dataset, root, profile
    )
    logging.info("Annotation protocol=%s fingerprint=%s; adapted dataset, exact published reproduction=False",
                 marker["annotation_protocol"], marker["dataset_fingerprint"])
    return frame, marker


def split_published_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = frame[["subject", "sequence"]].astype(int).apply(tuple, axis=1)
    train = frame[key.isin(PUBLISHED_TRAIN_RECORDINGS)].copy()
    validation = frame[key.isin(PUBLISHED_VALIDATION_RECORDINGS)].copy()
    test = frame[key.isin(PUBLISHED_TEST_RECORDINGS)].copy()
    observed = set(key.unique())
    expected = set(PUBLISHED_ALL_RECORDINGS)
    if observed != expected:
        raise RuntimeError(f"Published manifest recordings differ: observed={sorted(observed)}, expected={sorted(expected)}")
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        if subset.empty:
            raise RuntimeError(f"Published {name} partition is empty")
    if (
        set(train["sample_id"]) & set(validation["sample_id"])
        or set(train["sample_id"]) & set(test["sample_id"])
        or set(validation["sample_id"]) & set(test["sample_id"])
    ):
        raise RuntimeError("Published split sample leakage")
    return train, validation, test


# ---------------------------------------------------------------------------
# Datasets and models
# ---------------------------------------------------------------------------


def _flip_frame_and_labels(
    frame: np.ndarray,
    movement: int,
    static: int,
) -> tuple[np.ndarray, int, int]:
    flipped = np.ascontiguousarray(frame[..., ::-1])
    return flipped, MOVEMENT_FLIP.get(int(movement), int(movement)), STATIC_FLIP.get(int(static), int(static))


class PublishedES2FrameDataset:
    def __init__(
        self,
        preprocess_root: Path,
        samples: pd.DataFrame,
        augment_flip: bool,
        cache_size: int = 8,
    ) -> None:
        self.preprocess_root = preprocess_root
        self.samples = samples.reset_index(drop=True).copy()
        self.cache_size = max(1, int(cache_size))
        self.arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self.items: list[tuple[int, bool]] = [(index, False) for index in range(len(self.samples))]
        if augment_flip:
            self.items.extend((index, True) for index in range(len(self.samples)))

    def __len__(self) -> int:
        return len(self.items)

    def _array(self, relative: str) -> np.ndarray:
        if relative in self.arrays:
            value = self.arrays.pop(relative)
            self.arrays[relative] = value
            return value
        value = np.load(self.preprocess_root / relative, mmap_mode="r", allow_pickle=False)
        self.arrays[relative] = value
        while len(self.arrays) > self.cache_size:
            self.arrays.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        module = require_torch()
        row_index, flip = self.items[index]
        row = self.samples.iloc[row_index]
        frame = np.asarray(self._array(str(row["frame_path"]))[int(row["array_index"])], dtype=np.float32)
        movement = int(row["movement_label"])
        static = int(row["static_label"])
        if flip:
            frame, movement, static = _flip_frame_and_labels(frame, movement, static)
        return {
            "frames": module.from_numpy(np.ascontiguousarray(frame)),
            "movement_labels": module.tensor(movement, dtype=module.long),
            "static_labels": module.tensor(static, dtype=module.long),
            "static_loss_mask": module.tensor(static >= 0, dtype=module.bool),
            "sample_id": str(row["sample_id"]),
            "recording_key": str(row["recording_key"]),
            "gt_segment_id": str(row["gt_segment_id"]),
            "source_row": int(row["source_row"]),
            "array_index": int(row["array_index"]),
            "timestamp_us": int(row["timestamp_us"]),
            "subject": int(row["subject"]),
            "sequence": int(row["sequence"]),
            "augmented": bool(flip),
        }


def locate_es1_dual_labels(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"EventSleep1 dual-label CSV does not exist: {explicit}")
        return explicit.resolve()
    preferred = list(root.rglob("LabelsRollV2_movV2_2.csv"))
    candidates = preferred or list(root.rglob("*Labels*Roll*csv"))
    candidates = sorted(candidates, key=lambda value: (len(value.parts), str(value)))
    if not candidates:
        raise FileNotFoundError(
            "The EventSleep1 dual-label CSV was not found. Supply --eventsleep1-labels; "
            "the authors' filename is LabelsRollV2_movV2_2.csv."
        )
    return candidates[0].resolve()


def build_es1_dual_items(
    cache_manifest: pd.DataFrame,
    labels_path: Path,
    subjects: Sequence[int],
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    required = {"Subject", "Config", "Clip", "LabelF-G", "Label_roll", "NUMFRAMES"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"EventSleep1 dual-label CSV is missing columns: {missing}")
    selected_manifest = cache_manifest[cache_manifest["subject"].astype(int).isin(subjects)].copy()
    rows: list[dict[str, Any]] = []
    for cache_row in selected_manifest.itertuples(index=False):
        clip_match = re.search(r"(\d+)$", str(cache_row.clip_id))
        if not clip_match:
            raise ValueError(f"Cannot parse EventSleep1 clip ID: {cache_row.clip_id}")
        clip = int(clip_match.group(1))
        claims = labels[
            (pd.to_numeric(labels["Subject"], errors="coerce") == int(cache_row.subject))
            & (pd.to_numeric(labels["Config"], errors="coerce") == int(cache_row.configuration))
            & (pd.to_numeric(labels["Clip"], errors="coerce") == clip)
        ]
        if claims.empty:
            continue
        cursor = 0
        for claim_index, claim in claims.iterrows():
            count = int(claim["NUMFRAMES"])
            movement = int(claim["LabelF-G"])
            static_raw = int(claim["Label_roll"])
            static = static_raw - 6 if static_raw in range(6, 10) else -100
            if movement in range(6):
                for frame_index in range(cursor, cursor + count):
                    rows.append(
                        {
                            "sample_id": f"{cache_row.sample_id}_dualrow{int(claim_index):05d}_frame{frame_index:05d}",
                            "subject": int(cache_row.subject),
                            "configuration": int(cache_row.configuration),
                            "clip": clip,
                            "movement_label": movement,
                            "static_label": static,
                            "array_index": frame_index,
                            "frame_path": str(cache_row.resnete_frame_path),
                            "claim_row": int(claim_index),
                        }
                    )
            cursor += count
        if cursor != int(cache_row.resnete_frame_count):
            raise RuntimeError(
                f"EventSleep1 frame/label mismatch for {cache_row.sample_id}: "
                f"labels={cursor}, cache={int(cache_row.resnete_frame_count)}"
            )
    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No EventSleep1 dual-head training items were constructed")
    return output


class PublishedES1DualDataset:
    def __init__(
        self,
        preprocess_root: Path,
        samples: pd.DataFrame,
        augment_flip: bool,
        exclude_static_zero: bool,
        cache_size: int = 12,
    ) -> None:
        self.preprocess_root = preprocess_root
        self.samples = samples.reset_index(drop=True).copy()
        self.cache_size = max(1, int(cache_size))
        self.exclude_static_zero = bool(exclude_static_zero)
        self.arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self.items: list[tuple[int, bool]] = [(index, False) for index in range(len(self.samples))]
        if augment_flip:
            self.items.extend((index, True) for index in range(len(self.samples)))

    def __len__(self) -> int:
        return len(self.items)

    def _array(self, relative: str) -> np.ndarray:
        if relative in self.arrays:
            value = self.arrays.pop(relative)
            self.arrays[relative] = value
            return value
        value = np.load(self.preprocess_root / relative, mmap_mode="r", allow_pickle=False)
        self.arrays[relative] = value
        while len(self.arrays) > self.cache_size:
            self.arrays.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        module = require_torch()
        row_index, flip = self.items[index]
        row = self.samples.iloc[row_index]
        frame = np.asarray(self._array(str(row["frame_path"]))[int(row["array_index"])], dtype=np.float32)
        movement, static = int(row["movement_label"]), int(row["static_label"])
        if flip:
            frame, movement, static = _flip_frame_and_labels(frame, movement, static)
        return {
            "frames": module.from_numpy(np.ascontiguousarray(frame)),
            "movement_labels": module.tensor(movement, dtype=module.long),
            "static_labels": module.tensor(static, dtype=module.long),
            "static_loss_mask": module.tensor(
                static >= 0 and not (self.exclude_static_zero and static == 0),
                dtype=module.bool,
            ),
            "sample_id": str(row["sample_id"]),
        }


class PublishedEvS2Net(nn.Module):
    """ResNet18 encoder and two heads used by EventSleep2-net."""

    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as exc:
            raise RuntimeError("torchvision is required for EventSleep2-net") from exc
        weights = tvm.ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        backbone = tvm.resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pretrained = backbone
        self.fc_mv = nn.Linear(1000, 6)
        self.fc_st = nn.Linear(1000, 4)
        nn.init.xavier_normal_(self.fc_mv.weight)
        nn.init.xavier_normal_(self.fc_st.weight)

    def features(self, frames: Any) -> Any:
        return self.pretrained(frames)

    def forward(self, frames: Any) -> tuple[Any, Any]:
        features = self.features(frames)
        return self.fc_mv(features), self.fc_st(features)


def parameter_counts(model: Any) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


def make_loader(dataset: Any, batch_size: int, workers: int, shuffle: bool, seed: int) -> Any:
    module = require_torch()
    generator = module.Generator()
    generator.manual_seed(seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": module.cuda.is_available(),
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return module.utils.data.DataLoader(**arguments)


def _fixed_released_weights(device: str) -> tuple[Any, Any]:
    module = require_torch()
    motion = module.tensor(RELEASED_MOTION_FREQUENCIES.max() / RELEASED_MOTION_FREQUENCIES, device=device)
    static = module.tensor(RELEASED_STATIC_FREQUENCIES.max() / RELEASED_STATIC_FREQUENCIES, device=device)
    return motion, static


def dual_head_loss(
    movement_logits: Any,
    static_logits: Any,
    movement_labels: Any,
    static_labels: Any,
    profile: str,
    static_loss_mask: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    motion_weights, static_weights = _fixed_released_weights(str(movement_logits.device))
    if profile == "paper_declared":
        motion_weights = static_weights = None
    movement_loss = F.cross_entropy(movement_logits, movement_labels, weight=motion_weights)
    static_mask = static_labels >= 0 if static_loss_mask is None else static_loss_mask.bool()
    if bool(static_mask.any().item()):
        static_loss = F.cross_entropy(static_logits[static_mask], static_labels[static_mask], weight=static_weights)
        total = PAPER_MOTION_LOSS_WEIGHT * movement_loss + PAPER_STATIC_LOSS_WEIGHT * static_loss
    else:
        static_loss = movement_loss.new_zeros(())
        total = movement_loss
    return total, {
        "movement_loss": float(movement_loss.detach().cpu()),
        "static_loss": float(static_loss.detach().cpu()),
    }


def confusion_metrics(labels: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str], *, explicit_supported_recall: bool = False) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    ids = np.arange(len(class_names), dtype=np.int64)
    matrix = confusion_matrix(labels, predictions, labels=ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=ids, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": (
            float(recall[support > 0].mean()) if explicit_supported_recall
            else float(balanced_accuracy_score(labels, predictions))
        ),
        "macro_f1": float(f1_score(labels, predictions, labels=ids, average="macro", zero_division=0)),
        "samples": int(len(labels)),
    }
    per_class = pd.DataFrame(
        {
            "class_id": ids,
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return metrics, per_class, matrix


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def prediction_table(samples: pd.DataFrame, probabilities: np.ndarray, task: str) -> pd.DataFrame:
    output = samples.reset_index(drop=True).copy()
    labels = output[f"{task}_label"].to_numpy(dtype=np.int64)
    predictions = np.asarray(probabilities).argmax(axis=1)
    output["y_true"] = labels
    output["y_pred"] = predictions
    output["confidence"] = np.asarray(probabilities).max(axis=1)
    for index in range(probabilities.shape[1]):
        output[f"prob_{index}"] = np.asarray(probabilities)[:, index]
    return output


def aggregate_explicit_gt_segments(
    samples: pd.DataFrame,
    probabilities: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = samples.reset_index(drop=True).copy()
    frame["_position"] = np.arange(len(frame), dtype=np.int64)
    label_column = f"{task}_label"
    frame = frame[frame[label_column].astype(int) >= 0].copy()
    labels: list[int] = []
    scores: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for (recording, segment), group in frame.groupby(["recording_key", "gt_segment_id"], sort=False):
        target_values = group[label_column].astype(int).unique()
        if len(target_values) != 1:
            raise RuntimeError(f"GT segment {segment} contains multiple {task} labels")
        positions = group["_position"].to_numpy(dtype=np.int64)
        score = np.asarray(probabilities)[positions].sum(axis=0)
        labels.append(int(target_values[0]))
        scores.append(score)
        rows.append(
            {
                "recording_key": recording,
                "gt_segment_id": segment,
                "first_array_index": int(group["array_index"].min()),
                "last_array_index": int(group["array_index"].max()),
                "frames": len(group),
                "y_true": int(target_values[0]),
                "y_pred": int(score.argmax()),
            }
        )
    return np.asarray(labels, dtype=np.int64), np.stack(scores), pd.DataFrame(rows)


def aggregate_released_label_runs(
    samples: pd.DataFrame,
    probabilities: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    label_column = f"{task}_label"
    all_labels: list[int] = []
    all_scores: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    samples = samples.reset_index(drop=True).copy()
    samples["_position"] = np.arange(len(samples), dtype=np.int64)
    for recording, recording_frame in samples.groupby("recording_key", sort=False):
        recording_frame = recording_frame[recording_frame[label_column].astype(int) >= 0].copy()
        if recording_frame.empty:
            continue
        values = recording_frame[label_column].to_numpy(dtype=np.int64)
        starts = np.concatenate([[0], np.flatnonzero(values[1:] != values[:-1]) + 1])
        stops = np.concatenate([starts[1:], [len(values)]])
        for run_index, (start, stop) in enumerate(zip(starts, stops)):
            group = recording_frame.iloc[int(start):int(stop)]
            positions = group["_position"].to_numpy(dtype=np.int64)
            score = np.asarray(probabilities)[positions].sum(axis=0)
            target = int(values[int(start)])
            all_labels.append(target)
            all_scores.append(score)
            rows.append(
                {
                    "recording_key": recording,
                    "run_index": run_index,
                    "frames": int(stop - start),
                    "y_true": target,
                    "y_pred": int(score.argmax()),
                }
            )
    return np.asarray(all_labels, dtype=np.int64), np.stack(all_scores), pd.DataFrame(rows)


def evaluate_dual_predictions(
    samples: pd.DataFrame,
    movement_probabilities: np.ndarray,
    static_probabilities: np.ndarray,
) -> dict[str, Any]:
    samples = samples.reset_index(drop=True)
    movement_labels = samples["movement_label"].to_numpy(dtype=np.int64)
    static_mask = samples["static_label"].to_numpy(dtype=np.int64) >= 0
    static_labels = samples.loc[static_mask, "static_label"].to_numpy(dtype=np.int64)
    motion_frame, motion_pc, motion_cm = confusion_metrics(movement_labels, movement_probabilities, MOVEMENT_NAMES)
    static_frame, static_pc, static_cm = confusion_metrics(
        static_labels, static_probabilities[static_mask], STATIC_NAMES
    )
    motion_clip_y, motion_clip_p, motion_clip_rows = aggregate_explicit_gt_segments(
        samples, movement_probabilities, "movement"
    )
    static_clip_y, static_clip_p, static_clip_rows = aggregate_explicit_gt_segments(
        samples, static_probabilities, "static"
    )
    motion_clip, motion_clip_pc, motion_clip_cm = confusion_metrics(
        motion_clip_y, motion_clip_p, MOVEMENT_NAMES
    )
    static_clip, static_clip_pc, static_clip_cm = confusion_metrics(
        static_clip_y, static_clip_p, STATIC_NAMES
    )
    released_motion_y, released_motion_p, released_motion_rows = aggregate_released_label_runs(
        samples, movement_probabilities, "movement"
    )
    released_static_y, released_static_p, released_static_rows = aggregate_released_label_runs(
        samples, static_probabilities, "static"
    )
    released_motion, _, _ = confusion_metrics(released_motion_y, released_motion_p, MOVEMENT_NAMES)
    released_static, _, _ = confusion_metrics(released_static_y, released_static_p, STATIC_NAMES)
    headline = {
        "motion_per_frame_accuracy": motion_frame["accuracy"],
        "static_per_frame_accuracy": static_frame["accuracy"],
        "per_frame_average_accuracy": (motion_frame["accuracy"] + static_frame["accuracy"]) / 2,
        "motion_per_frame_balanced_accuracy": motion_frame["balanced_accuracy"],
        "static_per_frame_balanced_accuracy": static_frame["balanced_accuracy"],
        "per_frame_average_balanced_accuracy": (
            motion_frame["balanced_accuracy"] + static_frame["balanced_accuracy"]
        ) / 2,
        "motion_per_frame_macro_f1": motion_frame["macro_f1"],
        "static_per_frame_macro_f1": static_frame["macro_f1"],
        "motion_per_gt_clip_accuracy": motion_clip["accuracy"],
        "static_per_gt_clip_accuracy": static_clip["accuracy"],
        "per_gt_clip_average_accuracy": (motion_clip["accuracy"] + static_clip["accuracy"]) / 2,
        "motion_per_gt_clip_balanced_accuracy": motion_clip["balanced_accuracy"],
        "static_per_gt_clip_balanced_accuracy": static_clip["balanced_accuracy"],
        "per_gt_clip_average_balanced_accuracy": (
            motion_clip["balanced_accuracy"] + static_clip["balanced_accuracy"]
        ) / 2,
        "motion_per_gt_clip_macro_f1": motion_clip["macro_f1"],
        "static_per_gt_clip_macro_f1": static_clip["macro_f1"],
        "beats_published_sota_under_paper_declared_top1": {
            key: bool(value > SOTA_REFERENCE[key])
            for key, value in {
                "motion_per_frame_accuracy": motion_frame["accuracy"],
                "static_per_frame_accuracy": static_frame["accuracy"],
                "per_frame_average_accuracy": (motion_frame["accuracy"] + static_frame["accuracy"]) / 2,
                "motion_per_gt_clip_accuracy": motion_clip["accuracy"],
                "static_per_gt_clip_accuracy": static_clip["accuracy"],
                "per_gt_clip_average_accuracy": (motion_clip["accuracy"] + static_clip["accuracy"]) / 2,
            }.items()
        },
        "beats_published_sota_under_released_balanced_accuracy_interpretation": {
            key: bool(value > SOTA_REFERENCE[key])
            for key, value in {
                "motion_per_frame_accuracy": motion_frame["balanced_accuracy"],
                "static_per_frame_accuracy": static_frame["balanced_accuracy"],
                "per_frame_average_accuracy": (
                    motion_frame["balanced_accuracy"] + static_frame["balanced_accuracy"]
                ) / 2,
                "motion_per_gt_clip_accuracy": motion_clip["balanced_accuracy"],
                "static_per_gt_clip_accuracy": static_clip["balanced_accuracy"],
                "per_gt_clip_average_accuracy": (
                    motion_clip["balanced_accuracy"] + static_clip["balanced_accuracy"]
                ) / 2,
            }.items()
        },
    }
    return {
        "headline": headline,
        "movement_frame": motion_frame,
        "static_frame": static_frame,
        "movement_gt_clip": motion_clip,
        "static_gt_clip": static_clip,
        "released_label_run_diagnostic": {
            "movement": released_motion,
            "static": released_static,
        },
        "tables": {
            "movement_frame_per_class": motion_pc,
            "static_frame_per_class": static_pc,
            "movement_clip_per_class": motion_clip_pc,
            "static_clip_per_class": static_clip_pc,
            "movement_clip_predictions": motion_clip_rows,
            "static_clip_predictions": static_clip_rows,
            "released_movement_runs": released_motion_rows,
            "released_static_runs": released_static_rows,
            "movement_frame_confusion": pd.DataFrame(motion_cm, index=MOVEMENT_NAMES, columns=MOVEMENT_NAMES),
            "static_frame_confusion": pd.DataFrame(static_cm, index=STATIC_NAMES, columns=STATIC_NAMES),
            "movement_clip_confusion": pd.DataFrame(motion_clip_cm, index=MOVEMENT_NAMES, columns=MOVEMENT_NAMES),
            "static_clip_confusion": pd.DataFrame(static_clip_cm, index=STATIC_NAMES, columns=STATIC_NAMES),
        },
    }


def save_evaluation(output: Path, evaluation: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in evaluation.items() if key != "tables"}
    atomic_json(output / "metrics.json", serializable)
    for name, table in evaluation["tables"].items():
        frame = table.reset_index(names="true_class") if name.endswith("confusion") else table
        atomic_csv(output / f"{name}.csv", frame)


def headline_scalars(evaluation: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in evaluation["headline"].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def summarize_member_metrics(
    rows: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Published EventSleep2-net training and evaluation
# ---------------------------------------------------------------------------


def _batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    module = require_torch()
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, module.Tensor) else value
        for key, value in batch.items()
    }


def train_published_epoch(model: Any, loader: Any, optimizer: Any, device: str, profile: str) -> dict[str, float]:
    model.train()
    total = total_motion = total_static = 0.0
    batches = 0
    for raw in loader:
        batch = _batch_to_device(raw, device)
        optimizer.zero_grad(set_to_none=True)
        movement_logits, static_logits = model(batch["frames"])
        loss, parts = dual_head_loss(
            movement_logits,
            static_logits,
            batch["movement_labels"],
            batch["static_labels"],
            profile,
            batch.get("static_loss_mask"),
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite EventSleep2-net training loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        total_motion += parts["movement_loss"]
        total_static += parts["static_loss"]
        batches += 1
    if not batches:
        raise RuntimeError("No EventSleep2-net training batches were available")
    return {
        "loss": total / batches,
        "movement_loss": total_motion / batches,
        "static_loss": total_static / batches,
    }


def collect_published_outputs(
    model: Any,
    loader: Any,
    device: str,
    train_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, float]:
    if train_mode:
        model.train()
        inference_context: Any = torch.no_grad()
    else:
        model.eval()
        inference_context = torch.inference_mode()
    movement_outputs: list[np.ndarray] = []
    static_outputs: list[np.ndarray] = []
    movement_labels: list[np.ndarray] = []
    static_labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with inference_context:
        for raw in loader:
            batch = _batch_to_device(raw, device)
            motion, static = model(batch["frames"])
            movement_outputs.append(motion.detach().float().cpu().numpy())
            static_outputs.append(static.detach().float().cpu().numpy())
            movement_labels.append(batch["movement_labels"].detach().cpu().numpy())
            static_labels.append(batch["static_labels"].detach().cpu().numpy())
            count = len(raw["sample_id"])
            for index in range(count):
                row: dict[str, Any] = {"sample_id": str(raw["sample_id"][index])}
                # EventSleep1 validation items intentionally carry only a
                # sample ID. EventSleep2 evaluation items carry the complete
                # provenance fields used for clip aggregation and leakage
                # checks. Keeping this conditional lets the same inference
                # path serve both stages without inventing metadata.
                for key in (
                    "recording_key", "gt_segment_id", "source_row", "array_index",
                    "timestamp_us", "subject", "sequence",
                ):
                    if key not in raw:
                        continue
                    value = raw[key][index]
                    row[key] = str(value) if key in {"recording_key", "gt_segment_id"} else int(value)
                rows.append(row)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (
        np.concatenate(movement_outputs),
        np.concatenate(static_outputs),
        np.concatenate(movement_labels),
        np.concatenate(static_labels),
        pd.DataFrame(rows),
        time.perf_counter() - start,
    )


def validation_score(
    movement_logits: np.ndarray,
    static_logits: np.ndarray,
    movement_labels: np.ndarray,
    static_labels: np.ndarray,
    profile: str,
) -> tuple[float, dict[str, float]]:
    motion_metrics, _, _ = confusion_metrics(movement_labels, softmax_numpy(movement_logits), MOVEMENT_NAMES)
    mask = static_labels >= 0
    static_metrics, _, _ = confusion_metrics(
        static_labels[mask], softmax_numpy(static_logits[mask]), STATIC_NAMES
    )
    if profile == "released_code":
        score = motion_metrics["balanced_accuracy"]
    else:
        score = (motion_metrics["accuracy"] + static_metrics["accuracy"]) / 2
    return float(score), {
        "movement_accuracy": motion_metrics["accuracy"],
        "movement_balanced_accuracy": motion_metrics["balanced_accuracy"],
        "movement_macro_f1": motion_metrics["macro_f1"],
        "static_accuracy": static_metrics["accuracy"],
        "static_balanced_accuracy": static_metrics["balanced_accuracy"],
        "static_macro_f1": static_metrics["macro_f1"],
    }


def train_published_stage(
    model: Any,
    train_dataset: Any,
    validation_dataset: Any,
    epochs: int,
    checkpoint_path: Path,
    history_path: Path,
    seed: int,
    device: str,
    workers: int,
    profile: str,
    overwrite: bool,
    stage: str,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    settings = profile_settings(profile)
    if checkpoint_path.exists() and history_path.exists() and not overwrite:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "pipeline_version": VERSION,
            "stage": stage,
            "profile": profile,
            "seed": seed,
            "lr": PAPER_LEARNING_RATE,
            "batch_size": PAPER_BATCH_SIZE,
            "weight_decay": float(settings["optimizer_weight_decay"]),
            "dataset_fingerprint": dataset_fingerprint,
        }
        mismatches = {
            key: {"expected": value, "observed": checkpoint.get(key)}
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing baseline checkpoint metadata mismatch at {checkpoint_path}: {mismatches}; "
                "use --overwrite-runs only after reviewing the old result"
            )
        return {"status": "reused", "checkpoint": str(checkpoint_path)}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_dataset, PAPER_BATCH_SIZE, workers, True, seed)
    validation_loader = make_loader(
        validation_dataset,
        PAPER_BATCH_SIZE,
        workers,
        bool(settings["validation_augmentation"]),
        seed + 1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PAPER_LEARNING_RATE,
        weight_decay=float(settings["optimizer_weight_decay"]),
    )
    best_score = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        training = train_published_epoch(model, train_loader, optimizer, device, profile)
        motion, static, motion_y, static_y, _, seconds = collect_published_outputs(
            model,
            validation_loader,
            device,
            train_mode=settings["validation_batchnorm_mode"] == "train",
        )
        score, validation = validation_score(motion, static, motion_y, static_y, profile)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
            "validation_selection_score": score,
            "validation_seconds": seconds,
            "learning_rate": PAPER_LEARNING_RATE,
        }
        history.append(row)
        atomic_csv(history_path, pd.DataFrame(history))
        logging.info(
            "%s seed=%d epoch=%d/%d loss=%.4f val-score=%.4f",
            stage, seed, epoch, epochs, training["loss"], score,
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "pipeline_version": VERSION,
                    "stage": stage,
                    "profile": profile,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "lr": PAPER_LEARNING_RATE,
                    "batch_size": PAPER_BATCH_SIZE,
                    "weight_decay": float(settings["optimizer_weight_decay"]),
                    "dataset_fingerprint": dataset_fingerprint,
                    "validation_selection_score": score,
                    "validation_metrics": validation,
                },
                checkpoint_path,
            )
    if not checkpoint_path.exists():
        raise RuntimeError(f"No checkpoint was written for {stage}")
    return {
        "status": "trained",
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "dataset_fingerprint": dataset_fingerprint,
        "training_seconds": time.perf_counter() - start,
    }


def prepare_es1_dual_datasets(
    preprocess_root: Path,
    eventsleep1_root: Path,
    eventsleep1_labels: Path | None,
    cache_workers: int,
    overwrite_cache: bool,
    profile: str,
) -> tuple[Any, Any, dict[str, Any]]:
    companion = load_companion_module()
    paths = companion.DataPaths(preprocess_root)
    cache_profile = "authors_code_exact" if profile == "released_code" else "paper_intended"
    if not paths.resnete_manifest(cache_profile).exists() or overwrite_cache:
        companion.validate_preprocessed(paths, False)
        companion.build_published_resnete_cache(paths, cache_profile, cache_workers, overwrite_cache)
    cache_manifest = pd.read_csv(paths.resnete_manifest(cache_profile), low_memory=False)
    labels_path = locate_es1_dual_labels(eventsleep1_root, eventsleep1_labels)
    train_items = build_es1_dual_items(
        cache_manifest, labels_path, subjects=(1, 2, 3, 4, 5, 6, 7, 8, 10)
    )
    validation_items = build_es1_dual_items(cache_manifest, labels_path, subjects=(11,))
    settings = profile_settings(profile)
    train = PublishedES1DualDataset(
        preprocess_root,
        train_items,
        augment_flip=settings["training_augmentation"] == "horizontal_flip_with_left_right_label_swap",
        exclude_static_zero=profile == "released_code",
    )
    validation = PublishedES1DualDataset(
        preprocess_root,
        validation_items,
        augment_flip=bool(settings["validation_augmentation"]),
        exclude_static_zero=False,
    )
    report = {
        "labels_path": str(labels_path),
        "labels_sha256": file_sha256(labels_path),
        "cache_profile": cache_profile,
        "cache_manifest_sha256": file_sha256(paths.resnete_manifest(cache_profile)),
        "train_items_without_augmentation": len(train_items),
        "validation_items_without_augmentation": len(validation_items),
    }
    report["dataset_fingerprint"] = stable_hash(report, 64)
    return train, validation, report


def train_published_baseline(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    profile = str(args.profile)
    preprocess_root = args.preprocess_root.resolve()
    output_root = args.output.resolve() / "published_evs2net" / profile
    manifest, es2_provenance = load_experiment_dataset(args)
    es2_fingerprint = str(es2_provenance["dataset_fingerprint"])
    train_frame, validation_frame, _ = split_published_manifest(manifest)
    settings = profile_settings(profile)
    es2_train = PublishedES2FrameDataset(
        preprocess_root,
        train_frame,
        augment_flip=settings["training_augmentation"] == "horizontal_flip_with_left_right_label_swap",
    )
    es2_validation = PublishedES2FrameDataset(
        preprocess_root, validation_frame, augment_flip=bool(settings["validation_augmentation"])
    )
    es1_train = es1_validation = None
    es1_report: dict[str, Any] = {}
    if args.stage in {"all", "base"}:
        es1_train, es1_validation, es1_report = prepare_es1_dual_datasets(
            preprocess_root,
            args.eventsleep1_root.resolve(),
            None if args.eventsleep1_labels is None else args.eventsleep1_labels.resolve(),
            int(args.cache_workers),
            bool(args.overwrite_cache),
            profile,
        )
    shared_base_report: dict[str, Any] | None = None
    shared_base_checkpoint = output_root / f"shared_base_seed_{int(args.member_seeds[0])}" / "base_best.pt"
    if profile == "released_code" and args.stage in {"all", "base"}:
        assert es1_train is not None and es1_validation is not None
        shared_dir = shared_base_checkpoint.parent
        set_seed(int(args.member_seeds[0]), True)
        shared_model = PublishedEvS2Net(imagenet_pretrained=bool(args.imagenet_pretrained)).to(device)
        shared_base_report = train_published_stage(
            shared_model,
            es1_train,
            es1_validation,
            PAPER_BASE_EPOCHS,
            shared_base_checkpoint,
            shared_dir / "base_history.csv",
            int(args.member_seeds[0]),
            device,
            int(args.workers),
            profile,
            bool(args.overwrite_runs),
            "eventsleep1_shared_base",
            str(es1_report["dataset_fingerprint"]),
        )
        del shared_model
    member_reports: list[dict[str, Any]] = []
    for member, seed in enumerate(args.member_seeds, start=1):
        member_dir = output_root / f"member_{member:02d}_seed_{int(seed)}"
        base_checkpoint = (
            shared_base_checkpoint if profile == "released_code" else member_dir / "base_best.pt"
        )
        finetuned_checkpoint = member_dir / "finetuned_best.pt"
        report: dict[str, Any] = {"member": member, "seed": int(seed)}
        set_seed(int(seed), True)
        model = PublishedEvS2Net(imagenet_pretrained=bool(args.imagenet_pretrained)).to(device)
        if profile == "paper_declared" and args.stage in {"all", "base"}:
            assert es1_train is not None and es1_validation is not None
            report["base"] = train_published_stage(
                model,
                es1_train,
                es1_validation,
                PAPER_BASE_EPOCHS,
                base_checkpoint,
                member_dir / "base_history.csv",
                int(seed),
                device,
                int(args.workers),
                profile,
                bool(args.overwrite_runs),
                "eventsleep1_base",
                str(es1_report["dataset_fingerprint"]),
            )
        elif profile == "released_code":
            report["base"] = {
                "status": "shared_released_code_base",
                "checkpoint": str(shared_base_checkpoint),
            }
        if args.stage in {"all", "finetune"}:
            if not base_checkpoint.exists():
                raise FileNotFoundError(f"Missing base checkpoint for member {member}: {base_checkpoint}")
            checkpoint = torch.load(base_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            report["fine_tune"] = train_published_stage(
                model,
                es2_train,
                es2_validation,
                PAPER_FINETUNE_EPOCHS,
                finetuned_checkpoint,
                member_dir / "finetune_history.csv",
                int(seed) + 1000,
                device,
                int(args.workers),
                profile,
                bool(args.overwrite_runs),
                "eventsleep2_finetune",
                es2_fingerprint,
            )
        member_reports.append(report)
    summary = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "stage": args.stage,
        "device": device,
        "member_seeds": [int(value) for value in args.member_seeds],
        "eventsleep2_dataset": es2_provenance,
        "eventsleep1": es1_report,
        "shared_base": shared_base_report,
        "members": member_reports,
        "published_configuration": published_configuration(profile),
    }
    atomic_json(output_root / f"{args.stage}_completed.json", summary)
    return summary


def _feature_loader(model: Any, dataset: Any, device: str, workers: int, seed: int, task: str) -> Any:
    module = require_torch()
    source_loader = make_loader(dataset, PAPER_BATCH_SIZE, workers, False, seed)
    all_features: list[Any] = []
    all_labels: list[Any] = []
    model.eval()
    with module.inference_mode():
        for raw in source_loader:
            batch = _batch_to_device(raw, device)
            features = model.features(batch["frames"])
            labels = batch[f"{task}_labels"]
            if task == "static":
                mask = labels >= 0
                features, labels = features[mask], labels[mask]
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())
    tensor_dataset = module.utils.data.TensorDataset(module.cat(all_features), module.cat(all_labels))
    return module.utils.data.DataLoader(tensor_dataset, batch_size=PAPER_BATCH_SIZE, shuffle=False)


def laplace_predict_member(
    model: Any,
    fit_dataset: Any,
    test_dataset: Any,
    device: str,
    workers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    try:
        from laplace import Laplace
    except ImportError as exc:
        raise RuntimeError(
            "The published full model requires laplace-torch==0.2.2.2. Install it in the active environment."
        ) from exc
    motion_loader = _feature_loader(model, fit_dataset, device, workers, seed, "movement")
    static_loader = _feature_loader(model, fit_dataset, device, workers, seed, "static")
    motion_laplace = Laplace(
        model.fc_mv, "classification", subset_of_weights="all", hessian_structure="full"
    )
    static_laplace = Laplace(
        model.fc_st, "classification", subset_of_weights="all", hessian_structure="full"
    )
    motion_laplace.fit(motion_loader)
    static_laplace.fit(static_loader)
    test_loader = make_loader(test_dataset, PAPER_BATCH_SIZE, workers, False, seed + 1)
    motion_probabilities: list[np.ndarray] = []
    static_probabilities: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for raw in test_loader:
            batch = _batch_to_device(raw, device)
            features = model.features(batch["frames"])
            motion = motion_laplace(features, pred_type="glm", link_approx="bridge")
            static = static_laplace(features, pred_type="glm", link_approx="bridge")
            motion_probabilities.append(motion.detach().cpu().numpy())
            static_probabilities.append(static.detach().cpu().numpy())
            for index in range(len(raw["sample_id"])):
                rows.append(
                    {
                        "sample_id": str(raw["sample_id"][index]),
                        "recording_key": str(raw["recording_key"][index]),
                        "gt_segment_id": str(raw["gt_segment_id"][index]),
                        "source_row": int(raw["source_row"][index]),
                        "array_index": int(raw["array_index"][index]),
                        "timestamp_us": int(raw["timestamp_us"][index]),
                        "subject": int(raw["subject"][index]),
                        "sequence": int(raw["sequence"][index]),
                    }
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (
        np.concatenate(motion_probabilities),
        np.concatenate(static_probabilities),
        pd.DataFrame(rows),
        time.perf_counter() - start,
    )


def evaluate_published_baseline(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    profile = str(args.profile)
    preprocess_root = args.preprocess_root.resolve()
    output_root = args.output.resolve() / "published_evs2net" / profile
    manifest, es2_provenance = load_experiment_dataset(args)
    es2_fingerprint = str(es2_provenance["dataset_fingerprint"])
    train_frame, validation_frame, test_frame = split_published_manifest(manifest)
    fit_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    fit_dataset = PublishedES2FrameDataset(preprocess_root, fit_frame, augment_flip=False)
    test_dataset = PublishedES2FrameDataset(preprocess_root, test_frame, augment_flip=False)
    evaluation_frame = test_frame.reset_index(drop=True)
    deterministic_motion_logits: list[np.ndarray] = []
    deterministic_static_logits: list[np.ndarray] = []
    deterministic_motion_probabilities: list[np.ndarray] = []
    deterministic_static_probabilities: list[np.ndarray] = []
    laplace_motion: list[np.ndarray] = []
    laplace_static: list[np.ndarray] = []
    metadata: pd.DataFrame | None = None
    member_rows: list[dict[str, Any]] = []
    for member, seed in enumerate(args.member_seeds, start=1):
        member_dir = output_root / f"member_{member:02d}_seed_{int(seed)}"
        checkpoint_path = member_dir / "finetuned_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing fine-tuned checkpoint: {checkpoint_path}")
        set_seed(int(seed), True)
        model = PublishedEvS2Net(imagenet_pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset_fingerprint") != es2_fingerprint:
            raise RuntimeError(
                f"Fine-tuned checkpoint dataset fingerprint mismatch at {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(test_dataset, PAPER_BATCH_SIZE, int(args.workers), False, int(seed))
        motion_logits, static_logits, _, _, rows, seconds = collect_published_outputs(
            model, test_loader, device, train_mode=False
        )
        if metadata is None:
            metadata = rows
        elif not metadata["sample_id"].equals(rows["sample_id"]):
            raise RuntimeError("Ensemble members produced different test sample order")
        deterministic_motion_logits.append(motion_logits)
        deterministic_static_logits.append(static_logits)
        deterministic_motion_probabilities.append(softmax_numpy(motion_logits))
        deterministic_static_probabilities.append(softmax_numpy(static_logits))
        member_deterministic = evaluate_dual_predictions(
            evaluation_frame,
            softmax_numpy(motion_logits),
            softmax_numpy(static_logits),
        )
        save_evaluation(member_dir / "test_evaluation_deterministic", member_deterministic)
        row: dict[str, Any] = {
            "member": member,
            "seed": int(seed),
            "deterministic_seconds": seconds,
            **{
                f"deterministic_{key}": value
                for key, value in headline_scalars(member_deterministic).items()
            },
        }
        if bool(args.laplace):
            lap_motion, lap_static, lap_rows, lap_seconds = laplace_predict_member(
                model,
                fit_dataset,
                test_dataset,
                device,
                int(args.workers),
                int(seed),
            )
            if not rows["sample_id"].equals(lap_rows["sample_id"]):
                raise RuntimeError("Laplace predictions differ in sample order")
            laplace_motion.append(lap_motion)
            laplace_static.append(lap_static)
            row["laplace_seconds"] = lap_seconds
            member_lap_motion = softmax_numpy(lap_motion) if profile == "released_code" else lap_motion
            member_lap_static = softmax_numpy(lap_static) if profile == "released_code" else lap_static
            member_laplace = evaluate_dual_predictions(
                evaluation_frame, member_lap_motion, member_lap_static
            )
            save_evaluation(member_dir / "test_evaluation_laplace", member_laplace)
            row.update(
                {
                    f"laplace_{key}": value
                    for key, value in headline_scalars(member_laplace).items()
                }
            )
        member_rows.append(row)
    assert metadata is not None
    if not evaluation_frame["sample_id"].equals(metadata["sample_id"]):
        raise RuntimeError("Test manifest and prediction order differ")
    if profile == "released_code":
        deterministic_motion = softmax_numpy(
            np.mean(np.stack(deterministic_motion_logits), axis=0)
        )
        deterministic_static = softmax_numpy(
            np.mean(np.stack(deterministic_static_logits), axis=0)
        )
    else:
        deterministic_motion = np.mean(np.stack(deterministic_motion_probabilities), axis=0)
        deterministic_static = np.mean(np.stack(deterministic_static_probabilities), axis=0)
    deterministic_evaluation = evaluate_dual_predictions(
        evaluation_frame, deterministic_motion, deterministic_static
    )
    deterministic_dir = output_root / "evaluation_deterministic_ensemble"
    save_evaluation(deterministic_dir, deterministic_evaluation)
    final_name = "deterministic_ensemble"
    final_evaluation = deterministic_evaluation
    if bool(args.laplace):
        final_motion = np.mean(np.stack(laplace_motion), axis=0)
        final_static = np.mean(np.stack(laplace_static), axis=0)
        if profile == "released_code":
            # The public authors' evaluator applies softmax again to the
            # already-normalised posterior-predictive ensemble. This is kept
            # only in the explicitly labelled released-code diagnostic.
            final_motion = softmax_numpy(final_motion)
            final_static = softmax_numpy(final_static)
        final_evaluation = evaluate_dual_predictions(
            evaluation_frame, final_motion, final_static
        )
        save_evaluation(output_root / "evaluation_laplace_ensemble", final_evaluation)
        final_name = "laplace_ensemble"
    atomic_csv(output_root / "member_test_metrics.csv", pd.DataFrame(member_rows))
    member_metric_keys = sorted(
        key
        for key in member_rows[0]
        if key.startswith("deterministic_") or key.startswith("laplace_")
        if not key.endswith("_seconds")
    )
    summary = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "method": "published_evs2net",
        "final_variant": final_name,
        "official_test_evaluated": True,
        "member_seeds": [int(value) for value in args.member_seeds],
        "members": member_rows,
        "member_metric_summary": summarize_member_metrics(member_rows, member_metric_keys),
        "headline": final_evaluation["headline"],
        "sota_reference": SOTA_REFERENCE,
        "eventsleep2_dataset": es2_provenance,
        "published_configuration": published_configuration(profile),
        **parameter_counts(PublishedEvS2Net(imagenet_pretrained=False)),
    }
    atomic_json(output_root / "final_summary.json", summary)
    atomic_json(output_root / "completed_successfully.json", {"status": "success", **summary})
    return summary


# ---------------------------------------------------------------------------
# EventSleepFormer on the published split
# ---------------------------------------------------------------------------


@dataclass
class ProposedConfig:
    sequence_length: int = 32
    d_model: int = 256
    temporal_layers: int = 3
    temporal_heads: int = 4
    dropout: float = 0.10
    image_size: int = 224
    dino_model: str = "vit_small_patch14_dinov2.lvd142m"
    backbone_mode: str = "lora"
    lora_rank: int = 8
    lora_blocks: int = 4
    spatial_microbatch: int = 64
    causal: bool = True
    loss: str = "weighted_ce"
    learning_rate: float = 1e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    boundary_weight: float = 0.20
    motion_loss_weight: float = 0.6
    static_loss_weight: float = 0.4
    epochs: int = 50
    patience: int = 10
    batch_size: int = 4
    accumulation_steps: int = 1
    gradient_clip: float = 1.0
    amp: bool = True

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.d_model % self.temporal_heads:
            raise ValueError("d_model must be divisible by temporal_heads")
        if self.loss not in {"ce", "weighted_ce", "balanced_softmax", "focal"}:
            raise ValueError(f"Unsupported loss: {self.loss}")
        if self.backbone_mode not in {"lora", "frozen", "last_block", "full"}:
            raise ValueError(f"Unsupported backbone_mode: {self.backbone_mode}")
        if not math.isclose(self.motion_loss_weight + self.static_loss_weight, 1.0, abs_tol=1e-9):
            raise ValueError("motion_loss_weight + static_loss_weight must equal 1")


def proposed_config_id(config: ProposedConfig) -> str:
    config.validate()
    return stable_hash({
        "protocol": "eventsleep2_published_split",
        "training_revision": PROPOSED_TRAINING_REVISION,
        "config": asdict(config),
    })


def ordered_temporal_claims(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the released row-slice stream used to construct temporal windows.

    A physical frame can occur more than once when two label rows overlap.  The
    source-row key keeps each inclusive row slice intact and prevents a global
    frame-index sort from interleaving conflicting duplicate claims.
    """
    required = {"recording_key", "source_row", "array_index", "timestamp_us"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Temporal samples are missing columns: {missing}")
    return samples.sort_values(
        ["recording_key", "source_row", "array_index"], kind="stable"
    )


def temporal_claim_block_bounds(group: pd.DataFrame) -> list[tuple[int, int]]:
    """Partition one recording wherever the released claim stream jumps."""
    if group.empty:
        return []
    indices = group["array_index"].to_numpy(dtype=np.int64)
    timestamps = group["timestamp_us"].to_numpy(dtype=np.int64)
    starts = [0]
    for position in range(1, len(group)):
        continuous = indices[position] == indices[position - 1] + 1
        local_time = 0 <= timestamps[position] - timestamps[position - 1] <= 30_000_000
        if not (continuous and local_time):
            starts.append(position)
    stops = [*starts[1:], len(group)]
    return list(zip(starts, stops))


class EventSleepFormerWindowDataset:
    def __init__(
        self,
        preprocess_root: Path,
        samples: pd.DataFrame,
        sequence_length: int,
        augment: bool,
        seed: int,
        cache_size: int = 6,
    ) -> None:
        module = require_torch()
        self.module = module
        self.preprocess_root = preprocess_root
        self.sequence_length = int(sequence_length)
        self.target_stride = max(1, self.sequence_length // 2)
        self.context_length = self.sequence_length - self.target_stride
        self.augment = bool(augment)
        self.seed = int(seed)
        self._epoch = module.zeros((), dtype=module.int64).share_memory_()
        frame = samples.reset_index(drop=True).copy()
        frame["manifest_position"] = np.arange(len(frame), dtype=np.int64)
        frame = ordered_temporal_claims(frame)
        self.groups = {
            str(key): group.reset_index(drop=True)
            for key, group in frame.groupby("recording_key", sort=True)
        }
        self.items: list[tuple[str, int, int, int, int]] = []
        for key, group in self.groups.items():
            for block_start, block_stop in temporal_claim_block_bounds(group):
                for target_start in range(int(block_start), int(block_stop), self.target_stride):
                    target_stop = min(target_start + self.target_stride, int(block_stop))
                    start = max(int(block_start), target_start - self.context_length)
                    stop = min(start + self.sequence_length, int(block_stop))
                    if stop == int(block_stop):
                        start = max(int(block_start), stop - self.sequence_length)
                    self.items.append((key, start, stop, target_start, target_stop))
        self.cache_size = max(1, int(cache_size))
        self.arrays: OrderedDict[str, np.ndarray] = OrderedDict()

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return len(self.items)

    def _array(self, relative: str) -> np.ndarray:
        if relative in self.arrays:
            value = self.arrays.pop(relative)
            self.arrays[relative] = value
            return value
        value = np.load(self.preprocess_root / relative, mmap_mode="r", allow_pickle=False)
        self.arrays[relative] = value
        while len(self.arrays) > self.cache_size:
            self.arrays.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        companion = load_companion_module()
        key, start, stop, target_start, target_stop = self.items[index]
        group = self.groups[key].iloc[start:stop].copy()
        relative = str(group["frame_path"].iloc[0])
        indices = group["array_index"].to_numpy(dtype=np.int64)
        frames = np.asarray(self._array(relative)[indices], dtype=np.float32)
        timestamps = group["timestamp_us"].to_numpy(dtype=np.int64)
        movement = group["movement_label"].to_numpy(dtype=np.int64)
        static = group["static_label"].to_numpy(dtype=np.int64)
        positions = group["manifest_position"].to_numpy(dtype=np.int64)
        valid_length = len(group)
        if self.augment:
            rng = companion.augmentation_rng(self.seed, int(self._epoch.item()), index)
            frames = companion.augment_event_sequence(frames, rng)
            if rng.random() < 0.5:
                frames = np.ascontiguousarray(frames[..., ::-1])
                movement = np.asarray([MOVEMENT_FLIP.get(int(value), int(value)) for value in movement])
                static = np.asarray([STATIC_FLIP.get(int(value), int(value)) for value in static])
        descriptors = companion.frame_descriptors(frames, timestamps)
        attention = np.zeros(self.sequence_length, dtype=bool)
        target = np.zeros(self.sequence_length, dtype=bool)
        attention[:valid_length] = True
        global_positions = np.arange(start, stop)
        target[:valid_length] = (global_positions >= target_start) & (global_positions < target_stop)
        if valid_length < self.sequence_length:
            pad = self.sequence_length - valid_length
            frames = np.pad(frames, ((0, pad), (0, 0), (0, 0), (0, 0)))
            timestamps = np.pad(timestamps, (0, pad), mode="edge")
            descriptors = np.pad(descriptors, ((0, pad), (0, 0)))
            movement = np.pad(movement, (0, pad), constant_values=-100)
            static = np.pad(static, (0, pad), constant_values=-100)
            positions = np.pad(positions, (0, pad), constant_values=-1)
        movement_mask = attention & target & (movement >= 0)
        static_mask = attention & target & (static >= 0)
        return {
            "frames": self.module.from_numpy(np.ascontiguousarray(frames)),
            "timestamps": self.module.from_numpy(timestamps.copy()),
            "descriptors": self.module.from_numpy(descriptors.copy()),
            "attention_mask": self.module.from_numpy(attention),
            "movement_labels": self.module.from_numpy(movement.copy()).long(),
            "static_labels": self.module.from_numpy(static.copy()).long(),
            "movement_mask": self.module.from_numpy(movement_mask),
            "static_mask": self.module.from_numpy(static_mask),
            "manifest_positions": self.module.from_numpy(positions.copy()).long(),
        }


class DualHeadEventSleepFormer(nn.Module):
    def __init__(self, config: ProposedConfig, pretrained: bool = True) -> None:
        super().__init__()
        companion = load_companion_module()
        config.validate()
        model_config = companion.ModelConfig(
            method="eventsleepformer",
            num_classes=6,
            clip_level=False,
            descriptor_dim=5,
            d_model=config.d_model,
            temporal_layers=config.temporal_layers,
            temporal_heads=config.temporal_heads,
            dropout=config.dropout,
            sequence_length=config.sequence_length,
            image_size=config.image_size,
            dino_model=config.dino_model,
            pretrained=bool(pretrained),
            backbone_mode=config.backbone_mode,
            lora_rank=config.lora_rank,
            lora_blocks=config.lora_blocks,
            spatial_microbatch=config.spatial_microbatch,
            causal=config.causal,
            variant="full",
        )
        self.config = model_config
        self.proposed_config = copy.deepcopy(config)
        self.spatial = companion.SpatialEncoder(model_config)
        self.temporal = companion.EventAwareTemporalTransformer(self.spatial.output_dim, model_config)
        self.movement_head = nn.Linear(config.d_model, 6)
        self.static_head = nn.Linear(config.d_model, 4)
        self.boundary_head = nn.Linear(config.d_model * 2, 1)

    def forward(
        self,
        frames: Any,
        timestamps: Any,
        descriptors: Any,
        attention_mask: Any,
    ) -> dict[str, Any]:
        batch, length, channels, height, width = frames.shape
        spatial = self.spatial(frames.reshape(batch * length, channels, height, width)).reshape(batch, length, -1)
        features = self.temporal(
            spatial,
            timestamps=timestamps,
            descriptors=descriptors,
            attention_mask=attention_mask,
        )
        boundary = self.boundary_head(torch.cat([features[:, :-1], features[:, 1:]], dim=-1)).squeeze(-1)
        return {
            "movement_logits": self.movement_head(features),
            "static_logits": self.static_head(features),
            "boundary_logits": boundary,
            "features": features,
        }


def proposed_loader(dataset: Any, config: ProposedConfig, workers: int, shuffle: bool, seed: int) -> Any:
    module = require_torch()
    generator = module.Generator()
    generator.manual_seed(seed)
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": module.cuda.is_available(),
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        arguments.update(persistent_workers=True, prefetch_factor=2)
    return module.utils.data.DataLoader(**arguments)


def proposed_counts(samples: pd.DataFrame, device: str) -> tuple[Any, Any]:
    movement = np.bincount(samples["movement_label"].to_numpy(dtype=np.int64), minlength=6)
    static_values = samples.loc[samples["static_label"] >= 0, "static_label"].to_numpy(dtype=np.int64)
    static = np.bincount(static_values, minlength=4)
    if np.any(movement == 0) or np.any(static == 0):
        raise RuntimeError(f"A training class is absent: movement={movement.tolist()}, static={static.tolist()}")
    return (
        torch.tensor(movement, dtype=torch.float32, device=device),
        torch.tensor(static, dtype=torch.float32, device=device),
    )


def proposed_loss(
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    movement_counts: Any,
    static_counts: Any,
    config: ProposedConfig,
) -> tuple[Any, dict[str, float]]:
    companion = load_companion_module()
    motion_mask = batch["movement_mask"].bool()
    static_mask = batch["static_mask"].bool()
    motion = companion.classification_loss(
        output["movement_logits"][motion_mask],
        batch["movement_labels"][motion_mask],
        movement_counts,
        config.loss,
        config.label_smoothing,
    )
    static = companion.classification_loss(
        output["static_logits"][static_mask],
        batch["static_labels"][static_mask],
        static_counts,
        config.loss,
        config.label_smoothing,
    ) if bool(static_mask.any().item()) else motion.new_zeros(())
    total = config.motion_loss_weight * motion + config.static_loss_weight * static
    boundary = motion.new_zeros(())
    labels = batch["movement_labels"]
    # Each target frame contributes exactly one boundary decision against its
    # immediate predecessor (which may be context). This retains transitions
    # at target-chunk edges without double-counting overlapping context.
    valid = (
        motion_mask[:, 1:]
        & batch["attention_mask"][:, :-1].bool()
        & (labels[:, :-1] >= 0)
    )
    if config.boundary_weight > 0 and bool(valid.any().item()):
        target = (labels[:, :-1] != labels[:, 1:]).to(dtype=output["boundary_logits"].dtype)
        positives = target[valid].sum()
        negatives = valid.sum().to(dtype=target.dtype) - positives
        positive_weight = torch.clamp(negatives / positives.clamp_min(1), 1, 10)
        boundary = F.binary_cross_entropy_with_logits(
            output["boundary_logits"][valid], target[valid], pos_weight=positive_weight
        )
        total = total + config.boundary_weight * boundary
    return total, {
        "movement_loss": float(motion.detach().cpu()),
        "static_loss": float(static.detach().cpu()),
        "boundary_loss": float(boundary.detach().cpu()),
    }


def proposed_optimizer(model: Any, config: ProposedConfig) -> Any:
    companion = load_companion_module()
    train_config = companion.TrainConfig(
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        workers=0,
        learning_rate=config.learning_rate,
        backbone_lr_multiplier=config.backbone_lr_multiplier,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        accumulation_steps=config.accumulation_steps,
        gradient_clip=config.gradient_clip,
        amp=config.amp,
        label_smoothing=config.label_smoothing,
        boundary_weight=config.boundary_weight,
        deterministic=True,
    )
    return companion.optimizer_for(model, train_config), train_config


def proposed_optimizer_update(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    config: ProposedConfig,
    pending_batches: int,
) -> dict[str, Any]:
    """Advance the scheduler only when this single optimizer actually steps.

    GradScaler reduces its scale when it skips a step for nonfinite gradients.
    An unchanged or increased scale denotes a successful update. With scaling
    disabled, get_scale() remains one and the optimizer steps normally.
    """
    if not 1 <= pending_batches <= config.accumulation_steps:
        raise ValueError("Invalid number of accumulated supervised batches")
    scaler.unscale_(optimizer)
    # Losses were divided by accumulation_steps. Correct the final short
    # accumulation group to average its actual number of supervised batches.
    if pending_batches != config.accumulation_steps:
        correction = config.accumulation_steps / pending_batches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    norm_value = float(norm.detach().cpu())
    if not scaler.is_enabled() and not math.isfinite(norm_value):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite proposed-model gradient without GradScaler")
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    stepped = scale_after >= scale_before
    if stepped:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "optimizer_updated": stepped,
        "gradient_norm": norm_value,
        "amp_scale": scale_after,
    }


def train_proposed_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: str,
    movement_counts: Any,
    static_counts: Any,
    config: ProposedConfig,
) -> dict[str, float]:
    companion = load_companion_module()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "movement_loss": 0.0, "static_loss": 0.0, "boundary_loss": 0.0}
    batches = pending = 0
    updates = skipped = 0
    finite_norms: list[float] = []

    def update() -> None:
        nonlocal pending, updates, skipped
        result = proposed_optimizer_update(model, optimizer, scheduler, scaler, config, pending)
        updates += int(result["optimizer_updated"])
        skipped += int(not result["optimizer_updated"])
        if math.isfinite(result["gradient_norm"]):
            finite_norms.append(result["gradient_norm"])
        pending = 0

    for raw in loader:
        batch = _batch_to_device(raw, device)
        if not bool(batch["movement_mask"].any().item()):
            continue
        with companion.autocast_context(device, config.amp):
            output = model(
                batch["frames"], batch["timestamps"], batch["descriptors"], batch["attention_mask"]
            )
            loss, parts = proposed_loss(output, batch, movement_counts, static_counts, config)
            scaled = loss / config.accumulation_steps
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Non-finite EventSleepFormer loss")
        scaler.scale(scaled).backward()
        pending += 1
        if pending >= config.accumulation_steps:
            update()
        totals["loss"] += float(loss.detach().cpu())
        for key in parts:
            totals[key] += parts[key]
        batches += 1
    if pending:
        update()
    if not batches:
        raise RuntimeError("No supervised EventSleepFormer batches were available")
    if not updates:
        raise FloatingPointError("Every proposed-model optimizer update was skipped in this epoch")
    return {
        **{key: value / batches for key, value in totals.items()},
        "optimizer_updates": updates,
        "skipped_optimizer_updates": skipped,
        "amp_scale": float(scaler.get_scale()),
        "gradient_norm_mean": float(np.mean(finite_norms)) if finite_norms else 0.0,
    }


def collect_proposed_outputs(
    model: Any,
    loader: Any,
    samples: pd.DataFrame,
    device: str,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    companion = load_companion_module()
    model.eval()
    positions: list[np.ndarray] = []
    movement: list[np.ndarray] = []
    static: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for raw in loader:
            batch = _batch_to_device(raw, device)
            with companion.autocast_context(device, amp):
                output = model(
                    batch["frames"], batch["timestamps"], batch["descriptors"], batch["attention_mask"]
                )
            mask = batch["movement_mask"].detach().cpu().numpy().astype(bool)
            motion_logits = output["movement_logits"].detach().float().cpu().numpy()
            static_logits = output["static_logits"].detach().float().cpu().numpy()
            manifest_positions = raw["manifest_positions"].numpy()
            for item in range(mask.shape[0]):
                selected = mask[item]
                positions.append(manifest_positions[item, selected])
                movement.append(motion_logits[item, selected])
                static.append(static_logits[item, selected])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    position_array = np.concatenate(positions).astype(np.int64)
    movement_array = np.concatenate(movement)
    static_array = np.concatenate(static)
    if len(position_array) != len(samples):
        raise RuntimeError(
            f"EventSleepFormer prediction coverage mismatch: predictions={len(position_array)}, samples={len(samples)}"
        )
    if len(np.unique(position_array)) != len(samples) or set(position_array) != set(range(len(samples))):
        raise RuntimeError("Each supervised EventSleep2 frame must be evaluated exactly once")
    order = np.argsort(position_array, kind="stable")
    metadata = samples.reset_index(drop=True).iloc[position_array[order]].reset_index(drop=True)
    if not np.array_equal(position_array[order], np.arange(len(samples))):
        raise RuntimeError("Prediction manifest positions are not contiguous")
    return movement_array[order], static_array[order], metadata, time.perf_counter() - start


def proposed_validation_metrics(
    samples: pd.DataFrame,
    movement_logits: np.ndarray,
    static_logits: np.ndarray,
) -> tuple[float, dict[str, float]]:
    movement_probabilities = softmax_numpy(movement_logits)
    static_probabilities = softmax_numpy(static_logits)
    movement_labels = samples["movement_label"].to_numpy(dtype=np.int64)
    static_values = samples["static_label"].to_numpy(dtype=np.int64)
    static_mask = static_values >= 0
    movement_metrics, _, _ = confusion_metrics(
        movement_labels, movement_probabilities, MOVEMENT_NAMES, explicit_supported_recall=True
    )
    static_metrics, _, _ = confusion_metrics(
        static_values[static_mask], static_probabilities[static_mask], STATIC_NAMES,
        explicit_supported_recall=True,
    )
    score = PAPER_MOTION_LOSS_WEIGHT * movement_metrics["macro_f1"] + PAPER_STATIC_LOSS_WEIGHT * static_metrics["macro_f1"]
    return score, {
        "joint_macro_f1": score,
        "movement_accuracy": movement_metrics["accuracy"],
        "movement_balanced_accuracy": movement_metrics["balanced_accuracy"],
        "movement_macro_f1": movement_metrics["macro_f1"],
        "static_accuracy": static_metrics["accuracy"],
        "static_balanced_accuracy": static_metrics["balanced_accuracy"],
        "static_macro_f1": static_metrics["macro_f1"],
    }


def proposed_support_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, samples in (("train", train), ("validation", validation)):
        for task, names in (("movement", MOVEMENT_NAMES), ("static", STATIC_NAMES)):
            counts = samples[f"{task}_label"].value_counts()
            for class_id, name in enumerate(names):
                support = int(counts.get(class_id, 0))
                rows.append({"split": split, "task": task, "class_id": class_id,
                             "class_name": name, "support": support})
                if split == "validation" and support == 0:
                    logging.warning(
                        "Validation has zero %s samples for %s; fixed-class macro-F1 "
                        "retains this class with zero F1. Balanced accuracy averages "
                        "recall over supported classes, as in sklearn.", task, name,
                    )
    return pd.DataFrame(rows)


def save_proposed_validation_details(
    run_dir: Path, samples: pd.DataFrame, movement_logits: np.ndarray, static_logits: np.ndarray,
) -> None:
    """Persist the selected validation predictions, without opening test arrays."""
    for task, names, logits in (
        ("movement", MOVEMENT_NAMES, movement_logits), ("static", STATIC_NAMES, static_logits)
    ):
        mask = samples[f"{task}_label"].to_numpy(dtype=np.int64) >= 0
        selected = samples.loc[mask].reset_index(drop=True)
        probabilities = softmax_numpy(logits[mask])
        _, per_class, matrix = confusion_metrics(
            selected[f"{task}_label"].to_numpy(dtype=np.int64), probabilities, names,
            explicit_supported_recall=True,
        )
        atomic_csv(run_dir / f"validation_{task}_per_class.csv", per_class)
        confusion = pd.DataFrame(matrix, columns=[f"pred_{name}" for name in names])
        confusion.insert(0, "true_class", names)
        atomic_csv(run_dir / f"validation_{task}_confusion.csv", confusion)
        atomic_csv(run_dir / f"validation_{task}_predictions.csv",
                   prediction_table(selected, probabilities, task))


def train_proposed_run(
    preprocess_root: Path,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ProposedConfig,
    seed: int,
    device: str,
    workers: int,
    run_dir: Path,
    overwrite: bool,
    dataset_fingerprint: str,
    keep_checkpoint: bool = True,
) -> dict[str, Any]:
    cid = proposed_config_id(config)
    metrics_path = run_dir / "validation_metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if metrics_path.exists() and (checkpoint_path.exists() or not keep_checkpoint) and not overwrite:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("configuration_id") != cid
            or int(existing.get("seed", -1)) != int(seed)
            or existing.get("official_test_evaluated") is not False
            or existing.get("dataset_fingerprint") != dataset_fingerprint
            or existing.get("training_revision") != PROPOSED_TRAINING_REVISION
        ):
            raise RuntimeError(f"Existing proposed run metadata mismatch: {metrics_path}")
        return existing
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Incomplete proposed-model run exists: {run_dir}; inspect it or use --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(run_dir / "class_support.csv", proposed_support_table(train_frame, validation_frame))
    set_seed(seed, True)
    train_dataset = EventSleepFormerWindowDataset(
        preprocess_root, train_frame, config.sequence_length, True, seed
    )
    validation_dataset = EventSleepFormerWindowDataset(
        preprocess_root, validation_frame, config.sequence_length, False, seed
    )
    train_loader = proposed_loader(train_dataset, config, workers, True, seed)
    validation_loader = proposed_loader(validation_dataset, config, workers, False, seed + 1)
    movement_counts, static_counts = proposed_counts(train_frame, device)
    model = DualHeadEventSleepFormer(config, pretrained=True).to(device)
    optimizer, train_config = proposed_optimizer(model, config)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.accumulation_steps))
    companion = load_companion_module()
    scheduler = companion.scheduler_for(optimizer, steps_per_epoch, train_config)
    amp_enabled = config.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        train_metrics = train_proposed_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            movement_counts,
            static_counts,
            config,
        )
        movement_logits, static_logits, metadata, validation_seconds = collect_proposed_outputs(
            model, validation_loader, validation_frame, device, config.amp
        )
        score, validation_metrics = proposed_validation_metrics(metadata, movement_logits, static_logits)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "validation_seconds": validation_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
        logging.info(
            "EventSleepFormer config=%s seed=%d epoch=%d/%d loss=%.4f val-joint-macro-F1=%.4f "
            "lr=%.8g updates=%d amp-skips=%d amp-scale=%.0f",
            cid, seed, epoch, config.epochs, train_metrics["loss"], score,
            optimizer.param_groups[0]["lr"], train_metrics["optimizer_updates"],
            train_metrics["skipped_optimizer_updates"], train_metrics["amp_scale"],
        )
        if score > best_score + 1e-6:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            torch.save(
                {
                    "pipeline_version": VERSION,
                    "configuration_id": cid,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_epoch": epoch,
                    "best_validation_joint_macro_f1": score,
                    "dataset_fingerprint": dataset_fingerprint,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    movement_logits, static_logits, metadata, inference_seconds = collect_proposed_outputs(
        model, validation_loader, validation_frame, device, config.amp
    )
    score, values = proposed_validation_metrics(metadata, movement_logits, static_logits)
    save_proposed_validation_details(run_dir, metadata, movement_logits, static_logits)
    metrics = {
        "pipeline_version": VERSION,
        "training_revision": PROPOSED_TRAINING_REVISION,
        "configuration_id": cid,
        "seed": seed,
        "evaluation_scope": "validation_only",
        "official_test_evaluated": False,
        "test_dataset_instantiated": False,
        "dataset_fingerprint": dataset_fingerprint,
        "best_epoch": best_epoch,
        "best_validation_joint_macro_f1": best_score,
        "training_seconds": training_seconds,
        "optimizer_updates": sum(int(row["train_optimizer_updates"]) for row in history),
        "skipped_optimizer_updates": sum(int(row["train_skipped_optimizer_updates"]) for row in history),
        "validation_inference_seconds": inference_seconds,
        **values,
        **parameter_counts(model),
    }
    atomic_json(metrics_path, metrics)
    atomic_json(
        run_dir / "run_configuration.json",
        {
            "created_utc": utc_now(),
            "pipeline_version": VERSION,
            "configuration_id": cid,
            "config": asdict(config),
            "seed": seed,
            "protocol": "published_fixed_sequence_split",
            "evaluation_scope": "validation_only",
            "official_test_evaluated": False,
            "dataset_fingerprint": dataset_fingerprint,
            "train_recordings": [recording_key(*value) for value in PUBLISHED_TRAIN_RECORDINGS],
            "validation_recordings": [recording_key(*value) for value in PUBLISHED_VALIDATION_RECORDINGS],
            "test_recordings_opened": [],
        },
    )
    atomic_json(run_dir / "completed_successfully.json", {"status": "success", "configuration_id": cid})
    if not keep_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    return metrics


def curated_candidates(plan: str, epochs: int, patience: int) -> list[ProposedConfig]:
    base = ProposedConfig(epochs=epochs, patience=patience)
    candidates: list[ProposedConfig] = []

    def add(**changes: Any) -> None:
        candidate = copy.deepcopy(base)
        for key, value in changes.items():
            setattr(candidate, key, value)
        candidate.validate()
        candidates.append(candidate)

    # Stage A: the EventSleep2 starting configuration and loss/LR neighbours.
    add()
    add(loss="balanced_softmax")
    add(loss="focal")
    add(learning_rate=5e-5)
    add(learning_rate=2e-4)
    add(label_smoothing=0.0, boundary_weight=0.1)
    # Stage B: temporal scale and capacity neighbours selected for full nights.
    add(sequence_length=16, temporal_layers=2, warmup_epochs=2)
    add(sequence_length=64, temporal_layers=4, warmup_epochs=4, accumulation_steps=2)
    if plan == "full":
        add(d_model=192, temporal_layers=2, dropout=0.05)
        add(d_model=384, temporal_layers=4, temporal_heads=8, dropout=0.15, accumulation_steps=2)
        add(lora_rank=16, lora_blocks=6, learning_rate=5e-5)
        add(lora_rank=4, lora_blocks=2, dropout=0.05)
        add(backbone_mode="last_block", learning_rate=5e-5, backbone_lr_multiplier=0.05)
        add(motion_loss_weight=0.7, static_loss_weight=0.3)
        add(motion_loss_weight=0.5, static_loss_weight=0.5)
    unique: dict[str, ProposedConfig] = {}
    for candidate in candidates:
        unique[proposed_config_id(candidate)] = candidate
    return list(unique.values())


def load_candidate_file(path: Path) -> list[ProposedConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "candidates" in payload:
        rows = payload["candidates"]
    elif isinstance(payload, dict) and "config" in payload:
        rows = [payload["config"]]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("Candidate file must contain a list or {'candidates': [...]} object")
    candidates = [ProposedConfig(**row) for row in rows]
    for candidate in candidates:
        candidate.validate()
    return candidates


def tune_proposed(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    profile = str(args.profile)
    preprocess_root = args.preprocess_root.resolve()
    manifest, es2_provenance = load_experiment_dataset(args)
    es2_fingerprint = str(es2_provenance["dataset_fingerprint"])
    train_frame, validation_frame, test_frame = split_published_manifest(manifest)
    # Test metadata may be present in the cache manifest, but no test Dataset is
    # constructed and no test frame array is opened in this command.
    del test_frame
    candidates = (
        load_candidate_file(args.candidate_file.resolve())
        if args.candidate_file is not None
        else curated_candidates(str(args.search_plan), int(args.epochs), int(args.patience))
    )
    if len(candidates) > int(args.max_configurations):
        raise ValueError(
            f"Search has {len(candidates)} configurations, exceeding --max-configurations={args.max_configurations}"
        )
    output = args.output.resolve() / "eventsleepformer_tuning" / profile
    rows: list[dict[str, Any]] = []
    payloads: dict[str, ProposedConfig] = {}
    for candidate in candidates:
        cid = proposed_config_id(candidate)
        payloads[cid] = candidate
        for seed in args.seeds:
            run_dir = output / "runs" / f"cfg-{cid}_seed-{int(seed)}"
            metrics = train_proposed_run(
                preprocess_root,
                train_frame,
                validation_frame,
                candidate,
                int(seed),
                device,
                int(args.workers),
                run_dir,
                bool(args.overwrite),
                es2_fingerprint,
                keep_checkpoint=bool(args.keep_tuning_checkpoints),
            )
            rows.append({**metrics, **asdict(candidate)})
            atomic_csv(output / "tuning_progress.csv", pd.DataFrame(rows))
    runs = pd.DataFrame(rows)
    if runs.empty or runs["official_test_evaluated"].astype(bool).any():
        raise RuntimeError("Tuning leakage guard failed or no tuning runs completed")
    summary = (
        runs.groupby("configuration_id", as_index=False)
        .agg(
            validation_joint_macro_f1_mean=("joint_macro_f1", "mean"),
            validation_joint_macro_f1_std=("joint_macro_f1", "std"),
            movement_accuracy_mean=("movement_accuracy", "mean"),
            static_accuracy_mean=("static_accuracy", "mean"),
            completed_runs=("seed", "size"),
        )
        .sort_values(
            ["validation_joint_macro_f1_mean", "movement_accuracy_mean", "static_accuracy_mean"],
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    atomic_csv(output / "tuning_runs.csv", runs)
    atomic_csv(output / "tuning_summary.csv", summary)
    best_id = str(summary.iloc[0]["configuration_id"])
    locked = {
        "selected_utc": utc_now(),
        "pipeline_version": VERSION,
        "configuration_id": best_id,
        "config": asdict(payloads[best_id]),
        "selection_source": "validation_only",
        "selection_metric": "0.6*movement_macro_f1 + 0.4*static_macro_f1",
        "validation_joint_macro_f1_mean": float(summary.iloc[0]["validation_joint_macro_f1_mean"]),
        "official_test_evaluated_during_selection": False,
        "dataset_fingerprint": es2_fingerprint,
        "profile": profile,
        "protocol": "published_fixed_sequence_split",
        "annotation_protocol": es2_provenance.get("annotation_protocol", "released_indices"),
        "eventsleep2_dataset": es2_provenance,
    }
    atomic_json(output / "best_configuration.json", locked)
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "candidate_count": len(candidates),
        "completed_runs": len(runs),
        "best_configuration_id": best_id,
        "best_validation_joint_macro_f1": locked["validation_joint_macro_f1_mean"],
        "official_test_evaluated": False,
        "test_dataset_instantiated": False,
        "eventsleep2_dataset": es2_provenance,
    }
    atomic_json(output / "tuning_completed.json", report)
    return report


def load_locked_proposed(
    path: Path,
    expected_profile: str,
    expected_dataset_fingerprint: str | None = None,
) -> tuple[str, ProposedConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_evaluated_during_selection") is not False:
        raise RuntimeError("Locked configuration does not prove validation-only selection")
    if payload.get("profile") != expected_profile:
        raise RuntimeError(
            f"Locked profile {payload.get('profile')!r} differs from requested {expected_profile!r}"
        )
    if (
        expected_dataset_fingerprint is not None
        and payload.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise RuntimeError(
            "Locked EventSleepFormer configuration was selected on a different "
            "EventSleep2 dataset fingerprint"
        )
    config = ProposedConfig(**payload["config"])
    observed = proposed_config_id(config)
    if observed != str(payload.get("configuration_id")):
        raise RuntimeError("Locked EventSleepFormer configuration hash mismatch")
    return observed, config, payload


def final_proposed(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    profile = str(args.profile)
    preprocess_root = args.preprocess_root.resolve()
    manifest, es2_provenance = load_experiment_dataset(args)
    es2_fingerprint = str(es2_provenance["dataset_fingerprint"])
    cid, config, lock_payload = load_locked_proposed(
        args.locked_config.resolve(), profile, es2_fingerprint
    )
    train_frame, validation_frame, test_frame = split_published_manifest(manifest)
    output = args.output.resolve() / "eventsleepformer_final_published" / profile / f"cfg-{cid}"
    member_motion: list[np.ndarray] = []
    member_static: list[np.ndarray] = []
    individual_rows: list[dict[str, Any]] = []
    reference_metadata: pd.DataFrame | None = None
    for member, seed in enumerate(args.seeds, start=1):
        run_dir = output / "runs" / f"member-{member:02d}_seed-{int(seed)}"
        train_proposed_run(
            preprocess_root,
            train_frame,
            validation_frame,
            config,
            int(seed),
            device,
            int(args.workers),
            run_dir,
            bool(args.overwrite),
            es2_fingerprint,
            keep_checkpoint=True,
        )
        checkpoint_path = run_dir / "best_model.pt"
        model = DualHeadEventSleepFormer(config, pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("dataset_fingerprint") != es2_fingerprint:
            raise RuntimeError(
                f"EventSleepFormer checkpoint dataset fingerprint mismatch at {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_dataset = EventSleepFormerWindowDataset(
            preprocess_root, test_frame, config.sequence_length, False, int(seed)
        )
        test_loader = proposed_loader(test_dataset, config, int(args.workers), False, int(seed) + 1)
        motion_logits, static_logits, metadata, seconds = collect_proposed_outputs(
            model, test_loader, test_frame, device, config.amp
        )
        if reference_metadata is None:
            reference_metadata = metadata
        elif not reference_metadata["sample_id"].equals(metadata["sample_id"]):
            raise RuntimeError("EventSleepFormer final members produced different test sample order")
        motion_probabilities = softmax_numpy(motion_logits)
        static_probabilities = softmax_numpy(static_logits)
        member_motion.append(motion_probabilities)
        member_static.append(static_probabilities)
        individual = evaluate_dual_predictions(metadata, motion_probabilities, static_probabilities)
        save_evaluation(run_dir / "test_evaluation", individual)
        individual_rows.append(
            {
                "member": member,
                "seed": int(seed),
                "inference_seconds": seconds,
                **headline_scalars(individual),
            }
        )
        atomic_json(
            run_dir / "final_test_access.json",
            {
                "official_test_evaluated": True,
                "locked_configuration_id": cid,
                "locked_configuration_path": str(args.locked_config.resolve()),
            },
        )
    assert reference_metadata is not None
    ensemble = evaluate_dual_predictions(
        reference_metadata,
        np.mean(np.stack(member_motion), axis=0),
        np.mean(np.stack(member_static), axis=0),
    )
    save_evaluation(output / "ensemble_evaluation", ensemble)
    atomic_csv(output / "individual_test_metrics.csv", pd.DataFrame(individual_rows))
    individual_metric_keys = sorted(
        key for key in individual_rows[0] if key not in {"member", "seed", "inference_seconds"}
    )
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "method": "eventsleepformer_dual_head",
        "profile": profile,
        "protocol": "published_fixed_sequence_split",
        "configuration_id": cid,
        "locked_configuration": lock_payload,
        "seeds": [int(value) for value in args.seeds],
        "official_test_evaluated": True,
        "headline": ensemble["headline"],
        "individual_seed_summary": summarize_member_metrics(
            individual_rows, individual_metric_keys
        ),
        "sota_reference": SOTA_REFERENCE,
        "eventsleep2_dataset": es2_provenance,
    }
    atomic_json(output / "final_summary.json", report)
    atomic_json(output / "completed_successfully.json", {"status": "success", **report})
    return report


# ---------------------------------------------------------------------------
# Audits, smoke checks, and CLI
# ---------------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "timestamp_dataset", None) is not None:
        return timestamp_adapter().preflight(sys.modules[__name__], args)
    preprocess_root = args.preprocess_root.resolve()
    profile = str(args.profile)
    manifest = load_published_manifest(preprocess_root, profile)
    dataset_provenance = published_dataset_provenance(preprocess_root, profile)
    release_inventory = dataset_provenance["release_inventory"]
    train, validation, test = split_published_manifest(manifest)
    cache_root = published_manifest_path(preprocess_root, profile).parent
    marker = json.loads((cache_root / "completed_successfully.json").read_text(encoding="utf-8"))
    recordings_path = cache_root / "recordings.csv"
    if not recordings_path.is_file():
        raise FileNotFoundError(f"Missing published recording inventory: {recordings_path}")
    recordings = pd.read_csv(recordings_path, low_memory=False)
    release_recordings_path = cache_root / "release_inventory_recordings.csv"
    release_overlaps_path = cache_root / "release_inventory_overlaps.csv"
    if not release_recordings_path.is_file():
        raise FileNotFoundError(
            f"Missing release-specific recording audit: {release_recordings_path}"
        )
    if not release_overlaps_path.is_file():
        raise FileNotFoundError(
            f"Missing release-specific overlap audit: {release_overlaps_path}"
        )
    release_recordings = pd.read_csv(release_recordings_path, low_memory=False)
    release_overlaps = pd.read_csv(release_overlaps_path, low_memory=False)
    recording_columns = {
        "recording_key", "frame_path", "timestamp_path", "frames",
        "released_label_index_extent", "trailing_frame_margin", "alignment_status",
        "raw_sha256", "semantic_signature", "supervised_samples",
        "supervised_static_samples", "multiply_claimed_index_positions",
        "annotation_overlap_extra_claims", "supervised_overlap_positions",
        "supervised_overlap_extra_claims",
        "conflicting_movement_overlap_positions",
        "conflicting_static_overlap_positions", "overlapping_row_pairs",
    }
    missing_recording_columns = sorted(recording_columns - set(recordings.columns))
    if missing_recording_columns:
        raise ValueError(
            f"Published recording inventory is missing columns: {missing_recording_columns}"
        )
    missing_release_recording_columns = sorted(
        {
            "recording_key",
            "released_label_index_extent",
            "supervised_samples",
            "supervised_static_samples",
            "multiply_claimed_index_positions",
            "annotation_overlap_extra_claims",
            "supervised_overlap_positions",
            "supervised_overlap_extra_claims",
            "conflicting_movement_overlap_positions",
            "conflicting_static_overlap_positions",
            "overlapping_row_pairs",
        }
        - set(release_recordings.columns)
    )
    if missing_release_recording_columns:
        raise ValueError(
            "Release-specific recording audit is missing columns: "
            f"{missing_release_recording_columns}"
        )
    checked: list[dict[str, Any]] = []
    invalid: list[str] = []
    warnings = list(release_inventory.get("warnings", []))
    release_recordings_digest = file_sha256(release_recordings_path)
    release_overlaps_digest = file_sha256(release_overlaps_path)
    if marker.get("release_inventory_recordings_sha256") != release_recordings_digest:
        invalid.append("release-specific recording audit hash disagrees with completion marker")
    if marker.get("release_inventory_overlaps_sha256") != release_overlaps_digest:
        invalid.append("release-specific overlap audit hash disagrees with completion marker")
    segment_conflicts: list[str] = []
    if recordings["recording_key"].astype(str).duplicated().any():
        invalid.append("recording inventory contains duplicate recording keys")
    if release_recordings["recording_key"].astype(str).duplicated().any():
        invalid.append("release-specific recording audit contains duplicate recording keys")
    release_by_key = release_recordings.set_index("recording_key")
    recordings_by_key = recordings.set_index("recording_key")
    if set(release_by_key.index.astype(str)) != set(recordings_by_key.index.astype(str)):
        invalid.append("release-specific and rendered recording inventories use different keys")
    else:
        inventory_match_columns = (
            "released_label_index_extent",
            "supervised_samples",
            "supervised_static_samples",
            "multiply_claimed_index_positions",
            "annotation_overlap_extra_claims",
            "supervised_overlap_positions",
            "supervised_overlap_extra_claims",
            "conflicting_movement_overlap_positions",
            "conflicting_static_overlap_positions",
            "overlapping_row_pairs",
        )
        for key in recordings_by_key.index.astype(str):
            for column in inventory_match_columns:
                if int(recordings_by_key.loc[key, column]) != int(
                    release_by_key.loc[key, column]
                ):
                    invalid.append(
                        f"recording audit mismatch for {key}/{column}: "
                        f"rendered={recordings_by_key.loc[key, column]}, "
                        f"release={release_by_key.loc[key, column]}"
                    )
    for (recording, segment), group in manifest.groupby(
        ["recording_key", "gt_segment_id"], sort=False
    ):
        if group["movement_label"].astype(int).nunique() != 1:
            segment_conflicts.append(f"{recording}/{segment}: movement")
        applicable = group[group["static_label"].astype(int) >= 0]
        if not applicable.empty and applicable["static_label"].astype(int).nunique() != 1:
            segment_conflicts.append(f"{recording}/{segment}: static")
    invalid.extend(f"conflicting GT segment: {value}" for value in segment_conflicts)
    claim_order_valid = True
    claim_order_errors: list[str] = []
    for recording, group in manifest.groupby("recording_key", sort=False):
        source_rows = group["source_row"].to_numpy(dtype=np.int64)
        if len(source_rows) > 1 and np.any(np.diff(source_rows) < 0):
            claim_order_valid = False
            claim_order_errors.append(f"{recording}: source rows are not in CSV order")
        for source_row, claim in group.groupby("source_row", sort=False):
            indices = claim["array_index"].to_numpy(dtype=np.int64)
            if len(indices) > 1 and not np.all(np.diff(indices) == 1):
                claim_order_valid = False
                claim_order_errors.append(
                    f"{recording}/source_row={source_row}: frame slice is not contiguous"
                )
    invalid.extend(f"invalid released claim order: {value}" for value in claim_order_errors)
    rows_to_check = recordings.copy()
    if not bool(args.full_array_check):
        rows_to_check = rows_to_check.iloc[: min(len(rows_to_check), 3)].copy()
    checked_frame_total = 0
    for row in rows_to_check.itertuples(index=False):
        relative = str(row.frame_path)
        path = preprocess_root / relative
        timestamp_path = preprocess_root / str(row.timestamp_path)
        metadata_path = path.with_suffix(".json")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            claimed = manifest[
                manifest["frame_path"].astype(str) == relative
            ]["array_index"].astype(int)
            claimed_valid = (
                not claimed.empty
                and int(claimed.min()) >= 0
                and int(claimed.max()) < len(array)
            )
            recorded_frames = int(row.frames)
            released_extent = int(row.released_label_index_extent)
            alignment_valid = (
                str(row.alignment_status).startswith("released_indices_valid_")
                and metadata.get("alignment_audit", {}).get("all_released_indices_valid") is True
            )
            valid = bool(
                array.ndim == 4
                and array.shape[1:] == (2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH)
                and str(array.dtype) == "float32"
                and str(timestamps.dtype) == "int64"
                and len(array) == recorded_frames
                and len(timestamps) == recorded_frames
                and released_extent <= recorded_frames
                and int(row.trailing_frame_margin) == recorded_frames - released_extent
                and not np.any(np.diff(timestamps) < 0)
                and claimed_valid
                and alignment_valid
            )
            checked_frame_total += int(len(array))
            checked.append(
                {
                    "recording_key": str(row.recording_key),
                    "path": str(path),
                    "timestamp_path": str(timestamp_path),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "claimed_index_range": (
                        [int(claimed.min()), int(claimed.max())] if not claimed.empty else None
                    ),
                    "released_label_index_extent": released_extent,
                    "trailing_frame_margin": recorded_frames - released_extent,
                    "alignment_status": str(row.alignment_status),
                    "valid": valid,
                }
            )
            if not valid:
                invalid.append(str(path))
        except Exception as exc:
            invalid.append(f"{path}: {type(exc).__name__}: {exc}")
    expected_counts = {
        "train": [recording_key(*value) for value in PUBLISHED_TRAIN_RECORDINGS],
        "validation": [recording_key(*value) for value in PUBLISHED_VALIDATION_RECORDINGS],
        "test": [recording_key(*value) for value in PUBLISHED_TEST_RECORDINGS],
    }
    observed_counts = {
        "train": sorted(train["recording_key"].unique().tolist()),
        "validation": sorted(validation["recording_key"].unique().tolist()),
        "test": sorted(test["recording_key"].unique().tolist()),
    }
    expected_recording_keys = sorted(recording_key(*value) for value in PUBLISHED_ALL_RECORDINGS)
    observed_recording_keys = sorted(recordings["recording_key"].astype(str).tolist())
    if observed_recording_keys != expected_recording_keys:
        invalid.append(
            f"recording inventory mismatch: observed={observed_recording_keys}, "
            f"expected={expected_recording_keys}"
        )
    sample_ids_unique = not manifest["sample_id"].astype(str).duplicated().any()
    if not sample_ids_unique:
        invalid.append("published sample IDs are not unique")
    claim_sizes = manifest.groupby(["recording_key", "array_index"], sort=False).size()
    overlapping_claim_sizes = claim_sizes[claim_sizes > 1]
    multiply_claimed_positions = int(len(overlapping_claim_sizes))
    overlap_extra_sample_claims = int((overlapping_claim_sizes - 1).sum())
    physical_frame_claim_duplicate_rows = int(overlapping_claim_sizes.sum())
    expected_overlap_positions = int(
        release_inventory.get("protocol_supervised_overlap_positions", -1)
    )
    expected_extra_claims = int(
        release_inventory.get("protocol_supervised_overlap_extra_claims", -1)
    )
    if multiply_claimed_positions != expected_overlap_positions:
        invalid.append(
            f"manifest multiply-claimed positions {multiply_claimed_positions} != "
            f"release inventory {expected_overlap_positions}"
        )
    if overlap_extra_sample_claims != expected_extra_claims:
        invalid.append(
            f"manifest overlap extra claims {overlap_extra_sample_claims} != "
            f"release inventory {expected_extra_claims}"
        )
    if int(marker.get("physical_frame_claim_duplicate_rows", -1)) != (
        physical_frame_claim_duplicate_rows
    ):
        invalid.append("completion marker duplicate-row count disagrees with samples.csv")
    if int(marker.get("multiply_claimed_physical_frame_positions", -1)) != (
        multiply_claimed_positions
    ):
        invalid.append("completion marker overlap-position count disagrees with samples.csv")
    if int(marker.get("overlap_extra_sample_claims", -1)) != overlap_extra_sample_claims:
        invalid.append("completion marker overlap-extra-claim count disagrees with samples.csv")
    expected_pair_count = int(release_inventory.get("protocol_overlapping_row_pairs", -1))
    if len(release_overlaps) != expected_pair_count:
        invalid.append(
            f"overlap audit row-pair count {len(release_overlaps)} != "
            f"release inventory {expected_pair_count}"
        )
    if multiply_claimed_positions:
        warnings.append(
            f"Preserved {multiply_claimed_positions} multiply-claimed physical positions as "
            f"{overlap_extra_sample_claims} additional source-row sample claims"
        )
    recorded_rendered_total = int(recordings["frames"].astype(int).sum())
    released_index_extent_total = int(
        recordings["released_label_index_extent"].astype(int).sum()
    )
    recorded_supervised_total = int(recordings["supervised_samples"].astype(int).sum())
    recorded_static_total = int(
        recordings["supervised_static_samples"].astype(int).sum()
    )
    recorded_overlap_positions = int(
        recordings["supervised_overlap_positions"].astype(int).sum()
    )
    recorded_overlap_extra_claims = int(
        recordings["supervised_overlap_extra_claims"].astype(int).sum()
    )
    if len(manifest) != recorded_supervised_total:
        invalid.append(
            f"manifest sample count {len(manifest)} != recording inventory supervised total "
            f"{recorded_supervised_total}"
        )
    if int((manifest["static_label"].astype(int) >= 0).sum()) != recorded_static_total:
        invalid.append(
            "manifest static-supervision count disagrees with recording inventory"
        )
    if int(marker.get("supervised_samples", -1)) != len(manifest):
        invalid.append("completion marker supervised-sample count disagrees with samples.csv")
    if recorded_overlap_positions != multiply_claimed_positions:
        invalid.append("recording inventory overlap-position total disagrees with samples.csv")
    if recorded_overlap_extra_claims != overlap_extra_sample_claims:
        invalid.append("recording inventory overlap-extra-claim total disagrees with samples.csv")
    inventory_extent_total = int(
        release_inventory.get("protocol_released_label_index_extent_total", -1)
    )
    if released_index_extent_total != inventory_extent_total:
        invalid.append(
            f"recording label-index extent total {released_index_extent_total} != "
            f"selected-label inventory total {inventory_extent_total}"
        )
    if int(marker.get("rendered_frame_total", -1)) != recorded_rendered_total:
        invalid.append("completion marker rendered-frame total disagrees with recordings.csv")
    if int(marker.get("released_label_index_extent_total", -1)) != released_index_extent_total:
        invalid.append("completion marker label-index extent disagrees with recordings.csv")
    observed_fingerprint_payload = {
        "cache_version": PUBLISHED_CACHE_VERSION,
        "profile": profile,
        "labels_sha256": marker.get("labels_sha256"),
        "release_inventory_recordings_sha256": release_recordings_digest,
        "release_inventory_overlaps_sha256": release_overlaps_digest,
        "paper_split": published_configuration(profile)["split"],
        "label_application": {
            "order": "original_csv_source_row",
            "interval": "inclusive_FrameIni_through_FrameEnd",
            "overlap_policy": "preserve_each_source_row_claim",
        },
        "recordings": [
            {
                "recording_key": str(row.recording_key),
                "raw_sha256": str(row.raw_sha256),
                "semantic_signature": str(row.semantic_signature),
                "frames": int(row.frames),
                "released_label_index_extent": int(row.released_label_index_extent),
                "supervised_samples": int(row.supervised_samples),
                "supervised_static_samples": int(row.supervised_static_samples),
                "multiply_claimed_index_positions": int(
                    row.multiply_claimed_index_positions
                ),
                "annotation_overlap_extra_claims": int(
                    row.annotation_overlap_extra_claims
                ),
                "supervised_overlap_positions": int(row.supervised_overlap_positions),
                "supervised_overlap_extra_claims": int(
                    row.supervised_overlap_extra_claims
                ),
                "conflicting_movement_overlap_positions": int(
                    row.conflicting_movement_overlap_positions
                ),
                "conflicting_static_overlap_positions": int(
                    row.conflicting_static_overlap_positions
                ),
                "overlapping_row_pairs": int(row.overlapping_row_pairs),
            }
            for row in recordings.itertuples(index=False)
        ],
    }
    if marker.get("dataset_fingerprint_payload") != observed_fingerprint_payload:
        invalid.append(
            "dataset fingerprint payload disagrees with the current recording inventory"
        )
    elif stable_hash(observed_fingerprint_payload, 64) != dataset_provenance[
        "dataset_fingerprint"
    ]:
        invalid.append("recomputed dataset fingerprint disagrees with completion marker")
    alignment_path = cache_root / "alignment_audit.json"
    if not alignment_path.is_file():
        invalid.append(f"missing alignment audit: {alignment_path}")
        alignment_report: dict[str, Any] | None = None
    else:
        alignment_report = json.loads(alignment_path.read_text(encoding="utf-8"))
        if len(alignment_report.get("recordings", [])) != len(recordings):
            invalid.append("alignment audit recording count disagrees with recordings.csv")
    if bool(args.full_array_check):
        if len(checked) != len(recordings):
            invalid.append(f"checked {len(checked)} arrays but inventory contains {len(recordings)}")
        if checked_frame_total != recorded_rendered_total:
            invalid.append(
                f"checked frame total {checked_frame_total} != recorded render total "
                f"{recorded_rendered_total}"
            )
    split_valid = all(
        sorted(expected_counts[key]) == observed_counts[key] for key in expected_counts
    )
    if not split_valid:
        invalid.append(
            f"fixed paper split mismatch: expected={expected_counts}, observed={observed_counts}"
        )
    labels_path = Path(str(marker.get("labels_path", "")))
    if not labels_path.is_file():
        invalid.append(f"selected label file no longer exists: {labels_path}")
    elif file_sha256(labels_path) != marker.get("labels_sha256"):
        invalid.append("selected label file hash changed after cache preparation")
    report = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "passed": not invalid,
        "samples": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "test_static": int((test["static_label"] >= 0).sum()),
        },
        "expected_recordings": expected_counts,
        "observed_recordings": observed_counts,
        "arrays_checked": checked,
        "checked_frame_total": checked_frame_total,
        "recorded_rendered_frame_total": recorded_rendered_total,
        "released_label_index_extent_total": released_index_extent_total,
        "release_inventory": release_inventory,
        "legacy_v112_reference_label_index_extent_total": (
            LEGACY_V112_REFERENCE_LABEL_INDEX_EXTENT
        ),
        "legacy_reference_inventory_is_validation_requirement": False,
        "dataset_provenance": dataset_provenance,
        "frame_inventory_interpretation": (
            "rendered-frame total may exceed the released label-index extent; every released "
            "index must be in range"
        ),
        "sample_ids_unique": sample_ids_unique,
        "claim_order_valid": claim_order_valid,
        "claim_order_errors": claim_order_errors,
        "physical_frame_claim_duplicate_rows": physical_frame_claim_duplicate_rows,
        "multiply_claimed_physical_frame_positions": multiply_claimed_positions,
        "overlap_extra_sample_claims": overlap_extra_sample_claims,
        "overlap_audit": release_overlaps.to_dict(orient="records"),
        "overlap_interpretation": (
            "Each original CSV row contributes its full inclusive frame slice. Overlapping "
            "physical indices remain distinct supervised sample claims, matching the released "
            "GetLabelsFullSequencev21 loader."
        ),
        "gt_segment_conflicts": segment_conflicts,
        "invalid_arrays": invalid,
        "warnings": warnings,
        "alignment_audit": alignment_report,
        "timestamp_drift_policy": (
            "non-fatal when released indices remain valid; report drift and do not use labels "
            "to realign the frame stream"
        ),
        "paper_comparison_policy": (
            "architecture, preprocessing parameters, training hyperparameters, metric definitions, "
            "and the fixed recording split remain paper-locked; release-specific totals are derived "
            "from the SHA-256-identified label file and must be disclosed with results"
        ),
        "published_configuration": published_configuration(profile),
        "paper_code_discrepancies": {
            "loss": "paper says standard CE; released code uses fixed class-weighted CE",
            "metric": "paper defines top-1 sample accuracy; released plotting also labels mean class recall as Accuracy",
            "validation": "released code evaluates validation with model still in train mode and augments validation",
            "optimizer": "paper does not state weight decay; released code uses Adam weight_decay=0.01",
            "augmentation": "paper does not state horizontal flipping; released code duplicates horizontally flipped frames",
            "eventsleep1_static_mask": "released base-training code excludes static class 0 via targets_st > 0 after labels were shifted to 0..3",
            "clip_grouping": "released code groups contiguous equal labels; paper describes explicit GT clips",
            "seeds": "paper states five ensemble members but does not publish their independent seeds",
            "frame_origin": (
                "released no-GT converter assumes zero-relative timestamps; public AEDAT4 files "
                "contain absolute timestamps, so the equivalent boundary origin is first raw "
                "event + 150 ms"
            ),
            "released_index_timestamp_alignment": (
                "released FrameIni/FrameEnd indices are applied directly; their timestamp fields "
                "show measurable drift in the public release and are audit-only"
            ),
        },
    }
    atomic_json(args.output.resolve() / "published_preflight.json", report)
    if not report["passed"]:
        raise RuntimeError(
            f"EventSleep2 published-protocol preflight failed: {invalid or observed_counts}"
        )
    return report


def model_check(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    set_seed(20260904, True)
    baseline = PublishedEvS2Net(imagenet_pretrained=bool(args.imagenet_pretrained)).to(device)
    baseline.eval()
    frame = torch.zeros((2, 2, PAPER_OUTPUT_HEIGHT, PAPER_OUTPUT_WIDTH), device=device)
    with torch.inference_mode():
        motion, static = baseline(frame)
    official_checkpoint_report: dict[str, Any] | None = None
    if args.official_checkpoint is not None:
        checkpoint_path = args.official_checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        baseline.load_state_dict(state, strict=True)
        with torch.inference_mode():
            checkpoint_motion, checkpoint_static = baseline(frame)
        official_checkpoint_report = {
            "path": str(checkpoint_path),
            "strict_state_dict_load": True,
            "motion_shape": list(checkpoint_motion.shape),
            "static_shape": list(checkpoint_static.shape),
        }
    proposed_config = ProposedConfig(
        dino_model=str(args.dino_model),
        backbone_mode=str(args.backbone_mode),
        sequence_length=int(args.sequence_length),
        d_model=int(args.d_model),
        temporal_layers=int(args.temporal_layers),
        temporal_heads=int(args.temporal_heads),
        lora_rank=int(args.lora_rank),
        lora_blocks=int(args.lora_blocks),
    )
    proposed = DualHeadEventSleepFormer(proposed_config, pretrained=bool(args.pretrained)).to(device)
    proposed.eval()
    synthetic = torch.zeros((1, proposed_config.sequence_length, 2, 32, 40), device=device)
    timestamps = torch.arange(proposed_config.sequence_length, device=device).view(1, -1) * PAPER_FRAME_INTERVAL_US
    descriptors = torch.zeros((1, proposed_config.sequence_length, 5), device=device)
    attention = torch.ones((1, proposed_config.sequence_length), dtype=torch.bool, device=device)
    with torch.inference_mode():
        output = proposed(synthetic, timestamps, descriptors, attention)
    passed = (
        tuple(motion.shape) == (2, 6)
        and tuple(static.shape) == (2, 4)
        and tuple(output["movement_logits"].shape) == (1, proposed_config.sequence_length, 6)
        and tuple(output["static_logits"].shape) == (1, proposed_config.sequence_length, 4)
        and bool(torch.isfinite(output["movement_logits"]).all().item())
    )
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "device": device,
        "baseline_motion_shape": list(motion.shape),
        "baseline_static_shape": list(static.shape),
        "proposed_motion_shape": list(output["movement_logits"].shape),
        "proposed_static_shape": list(output["static_logits"].shape),
        "baseline_parameters": parameter_counts(baseline),
        "proposed_parameters": parameter_counts(proposed),
        "official_checkpoint": official_checkpoint_report,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError("Model component check failed")
    atomic_json(args.output.resolve() / "eventsleep2_model_check.json", report)
    return report


def synthetic_smoke_test(output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20260904)
    synthetic_packets = [
        np.asarray([[0, 0, 1_000_000, 0], [1, 1, 1_050_000, 1]], dtype=np.int64),
        np.asarray([[2, 2, 1_160_000, 0]], dtype=np.int64),
        np.asarray([[3, 3, 1_310_000, 1]], dtype=np.int64),
        np.asarray([[4, 4, 2_000_000, 0]], dtype=np.int64),
    ]
    synthetic_anchors: list[int] = []

    def collect_anchor(_: int, events: np.ndarray) -> None:
        synthetic_anchors.append(int(events[:, 2].max()))

    stream_check = _stream_published_packets(
        iter(synthetic_packets), "synthetic_absolute_timestamps", "smoke", collect_anchor
    )
    synthetic_labels = pd.DataFrame(
        [{"FrameIni": 0, "FrameEnd": 2, "TSInit": 1_050_000, "TSEnd": 1_310_000}]
    )
    alignment_check = _alignment_audit(
        synthetic_labels, np.asarray(synthetic_anchors, dtype=np.int64)
    )
    frame_origin_stream_check = bool(
        stream_check["raw_initial_timestamp_us"] == 1_000_000
        and stream_check["frames"] == 3
        and synthetic_anchors == [1_050_000, 1_160_000, 1_310_000]
        and alignment_check["all_released_indices_valid"]
        and alignment_check["exactly_timestamp_aligned_rows"] == 1
    )
    inventory_rows: list[dict[str, Any]] = []
    for source_row, (subject, sequence) in enumerate(PUBLISHED_ALL_RECORDINGS):
        movement = source_row % 6
        inventory_rows.append(
            {
                "source_row": source_row,
                "Subject": subject,
                "Sequence": sequence,
                "FrameIni": 0,
                "FrameEnd": 2,
                "Label_mv": movement,
                "Label_st": 6 if movement in (2, 3) else np.nan,
                "nFrames": 3,
                "TSInit": 1_000_000,
                "TSEnd": 1_300_000,
            }
        )
    synthetic_inventory, _, _ = derive_release_inventory(pd.DataFrame(inventory_rows))
    adaptive_inventory_check = bool(
        synthetic_inventory["structurally_compatible"]
        and not synthetic_inventory["legacy_v112_reference_comparison"]["exact_match"]
        and synthetic_inventory["protocol_released_label_index_extent_total"] == 30
    )
    overlap_inventory_rows = list(inventory_rows)
    next_source_row = len(overlap_inventory_rows)
    for subject, sequence, start, end, movement, static in (
        (2, 1, 799, 818, 3, 7),
        (2, 1, 818, 828, 5, np.nan),
        (2, 1, 862, 882, 2, 6),
        (2, 1, 876, 923, 3, 7),
        (1, 4, 209, 261, 2, 6),
        (1, 4, 235, 250, 0, np.nan),
    ):
        overlap_inventory_rows.append(
            {
                "source_row": next_source_row,
                "Subject": subject,
                "Sequence": sequence,
                "FrameIni": start,
                "FrameEnd": end,
                "Label_mv": movement,
                "Label_st": static,
                "nFrames": end - start + 1,
                "TSInit": 1_000_000 + start * PAPER_FRAME_INTERVAL_US,
                "TSEnd": 1_000_000 + end * PAPER_FRAME_INTERVAL_US,
            }
        )
        next_source_row += 1
    overlap_inventory, _, overlap_pairs = derive_release_inventory(
        pd.DataFrame(overlap_inventory_rows)
    )
    released_overlap_semantics_check = bool(
        overlap_inventory["structurally_compatible"]
        and overlap_inventory["protocol_multiply_claimed_index_positions"] == 24
        and overlap_inventory["protocol_supervised_overlap_positions"] == 24
        and overlap_inventory["protocol_supervised_overlap_extra_claims"] == 24
        and overlap_inventory["protocol_conflicting_movement_overlap_positions"] == 24
        and overlap_inventory["protocol_conflicting_static_overlap_positions"] == 7
        and overlap_inventory["protocol_overlapping_row_pairs"] == 3
        and len(overlap_pairs) == 3
    )
    temporal_overlap_claims = pd.DataFrame(
        [
            {
                "recording_key": "subject02_seq01",
                "source_row": source_row,
                "array_index": frame_index,
                "timestamp_us": frame_index * PAPER_FRAME_INTERVAL_US,
            }
            for source_row, start, end in (
                (618, 799, 818),
                (619, 818, 828),
                (623, 862, 882),
                (624, 876, 923),
            )
            for frame_index in range(start, end + 1)
        ]
    )
    ordered_overlap_claims = ordered_temporal_claims(temporal_overlap_claims)
    temporal_blocks = temporal_claim_block_bounds(ordered_overlap_claims)
    temporal_overlap_isolation_check = bool(
        len(temporal_blocks) == 4
        and ordered_overlap_claims["source_row"].is_monotonic_increasing
        and all(
            np.all(
                np.diff(
                    ordered_overlap_claims.iloc[start:stop]["array_index"].to_numpy(
                        dtype=np.int64
                    )
                )
                == 1
            )
            for start, stop in temporal_blocks
        )
    )
    compatible_cache_payload = {
        "algorithm": "eventsleep2_authors_full_recording_stream_v2",
        "profile": "paper_declared",
        "source_name": "synthetic.aedat4",
    }
    wrapped_cache_payload = {"pipeline_version": VERSION, **compatible_cache_payload}
    semantic_cache_signature_check = bool(
        _cache_semantic_payload(wrapped_cache_payload)
        == _cache_semantic_payload(compatible_cache_payload)
        and stable_hash(wrapped_cache_payload, 64)
        != stable_hash(compatible_cache_payload, 64)
    )
    samples: list[dict[str, Any]] = []
    position = 0
    for subject, sequence in PUBLISHED_ALL_RECORDINGS:
        key = recording_key(subject, sequence)
        for segment, movement in enumerate(range(6)):
            static = (subject + sequence + segment) % 4 if movement in (2, 3) else -100
            for frame_index in range(3):
                samples.append(
                    {
                        "sample_id": f"{key}_{position}",
                        "recording_key": key,
                        "subject": subject,
                        "sequence": sequence,
                        "source_row": segment,
                        "gt_segment_id": f"{key}_{segment}",
                        "array_index": position,
                        "timestamp_us": position * PAPER_FRAME_INTERVAL_US,
                        "movement_label": movement,
                        "static_label": static,
                        "frame_path": f"frames/{key}.npy",
                    }
                )
                position += 1
    manifest = pd.DataFrame(samples)
    train, validation, test = split_published_manifest(manifest)
    movement_logits = rng.normal(size=(len(test), 6))
    static_logits = rng.normal(size=(len(test), 4))
    movement_logits[np.arange(len(test)), test["movement_label"].to_numpy(dtype=int)] += 2
    static_mask = test["static_label"].to_numpy(dtype=int) >= 0
    static_logits[np.flatnonzero(static_mask), test.loc[static_mask, "static_label"].to_numpy(dtype=int)] += 2
    evaluation = evaluate_dual_predictions(test, softmax_numpy(movement_logits), softmax_numpy(static_logits))
    base = ProposedConfig()
    cid = proposed_config_id(base)
    lock = {
        "configuration_id": cid,
        "config": asdict(base),
        "official_test_evaluated_during_selection": False,
        "dataset_fingerprint": "synthetic-dataset-fingerprint",
        "profile": "paper_declared",
    }
    lock_path = output / "synthetic_lock.json"
    atomic_json(lock_path, lock)
    observed, _, _ = load_locked_proposed(
        lock_path, "paper_declared", "synthetic-dataset-fingerprint"
    )
    representation = published_configuration("paper_declared")["representation"]
    decoder_recorded = representation["decoder"]["version"] == AUTHORS_AEDAT_VERSION
    preprocessing_environment_recorded = representation["authors_preprocessing_environment"] == {
        "python": AUTHORS_PYTHON_VERSION,
        "dependencies": AUTHORS_PREPROCESS_DEPENDENCIES,
    }
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "configuration_hash_round_trip": observed == cid,
        "authors_aedat_version": AUTHORS_AEDAT_VERSION,
        "decoder_configuration_recorded": decoder_recorded,
        "preprocessing_environment_recorded": preprocessing_environment_recorded,
        "full_recording_origin_stream_check": frame_origin_stream_check,
        "adaptive_release_inventory_check": adaptive_inventory_check,
        "released_overlap_semantics_check": released_overlap_semantics_check,
        "temporal_overlap_isolation_check": temporal_overlap_isolation_check,
        "semantic_cache_signature_check": semantic_cache_signature_check,
        "synthetic_overlap_inventory": overlap_inventory,
        "synthetic_overlap_pairs": overlap_pairs.to_dict(orient="records"),
        "synthetic_temporal_blocks": [list(value) for value in temporal_blocks],
        "synthetic_stream_audit": stream_check,
        "synthetic_alignment_audit": alignment_check,
        "movement_frame_accuracy": evaluation["headline"]["motion_per_frame_accuracy"],
        "static_frame_accuracy": evaluation["headline"]["static_per_frame_accuracy"],
        "passed": (
            observed == cid
            and decoder_recorded
            and preprocessing_environment_recorded
            and frame_origin_stream_check
            and adaptive_inventory_check
            and released_overlap_semantics_check
            and temporal_overlap_isolation_check
            and semantic_cache_signature_check
            and len(train) > 0
            and len(validation) > 0
            and len(test) > 0
        ),
    }
    atomic_json(output / "smoke_test_completed.json", report)
    return report


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PUBLISHED_PROFILES, default="paper_declared")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timestamp-dataset", type=Path, default=None,
                        help="Explicit adapted timestamp dataset directory; default uses released indices")


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EventSleep2 published-protocol reproduction and EventSleepFormer tuning"
    )
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-published", help="Build 320x240 frames using declared EventSleep2 preprocessing parameters")
    add_common(prepare)
    prepare.add_argument("--eventsleep2-root", type=Path, required=True)
    prepare.add_argument("--eventsleep2-labels", type=Path, default=None)
    prepare.add_argument("--overwrite-cache", action="store_true")
    prepare.add_argument("--recordings-limit", type=int, default=None, help="Smoke/pilot only")

    timestamp_prepare = commands.add_parser("prepare-timestamp-aligned", help="Create adapted timestamp claims using existing frame caches")
    add_common(timestamp_prepare)
    timestamp_prepare.add_argument("--eventsleep2-labels", type=Path, default=None)

    check = commands.add_parser("preflight", help="Verify the fixed split and published frame cache")
    add_common(check)
    check.add_argument("--full-array-check", action="store_true")

    baseline = commands.add_parser("baseline-train", help="Train the paper EventSleep2-net baseline")
    add_common(baseline)
    add_device(baseline)
    baseline.add_argument("--stage", choices=("base", "finetune", "all"), default="all")
    baseline.add_argument("--eventsleep1-root", type=Path, required=True)
    baseline.add_argument("--eventsleep1-labels", type=Path, default=None)
    baseline.add_argument("--member-seeds", nargs="+", type=int, default=[9, 19, 29, 39, 49])
    baseline.add_argument("--cache-workers", type=int, default=2)
    baseline.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    baseline.add_argument("--overwrite-cache", action="store_true")
    baseline.add_argument("--overwrite-runs", action="store_true")
    baseline.add_argument("--allow-nonpaper-ensemble", action="store_true")

    baseline_eval = commands.add_parser("baseline-evaluate", help="Evaluate the five-member EventSleep2-net ensemble")
    add_common(baseline_eval)
    add_device(baseline_eval)
    baseline_eval.add_argument("--member-seeds", nargs="+", type=int, default=[9, 19, 29, 39, 49])
    baseline_eval.add_argument("--laplace", action=argparse.BooleanOptionalAction, default=True)
    baseline_eval.add_argument("--allow-nonpaper-ensemble", action="store_true")

    tune = commands.add_parser("tune-proposed", help="Validation-only EventSleepFormer tuning")
    add_common(tune)
    add_device(tune)
    tune.add_argument("--search-plan", choices=("quick", "full"), default="quick")
    tune.add_argument("--candidate-file", type=Path, default=None)
    tune.add_argument("--epochs", type=int, default=50)
    tune.add_argument("--patience", type=int, default=10)
    tune.add_argument("--seeds", nargs="+", type=int, default=[13])
    tune.add_argument("--max-configurations", type=int, default=20)
    tune.add_argument("--keep-tuning-checkpoints", action="store_true")
    tune.add_argument("--overwrite", action="store_true")

    final = commands.add_parser("final-proposed", help="Run locked EventSleepFormer evaluation on the official test split")
    add_common(final)
    add_device(final)
    final.add_argument("--locked-config", type=Path, required=True)
    final.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    final.add_argument("--allow-nonpaper-ensemble", action="store_true")
    final.add_argument("--overwrite", action="store_true")

    component = commands.add_parser("model-check", help="Run baseline and proposed model forward checks")
    component.add_argument("--output", type=Path, required=True)
    component.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    component.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=False)
    component.add_argument("--official-checkpoint", type=Path, default=None)
    component.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    component.add_argument("--dino-model", default="tiny_event_cnn")
    component.add_argument("--backbone-mode", choices=("lora", "frozen", "last_block", "full"), default="frozen")
    component.add_argument("--sequence-length", type=int, default=4)
    component.add_argument("--d-model", type=int, default=64)
    component.add_argument("--temporal-layers", type=int, default=1)
    component.add_argument("--temporal-heads", type=int, default=4)
    component.add_argument("--lora-rank", type=int, default=4)
    component.add_argument("--lora-blocks", type=int, default=2)
    component.add_argument("--verbose", action="store_true")

    smoke = commands.add_parser("smoke-test", help="Run data-only protocol and metric checks")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    configure_logging(output / "logs" / f"{args.command}.log", bool(getattr(args, "verbose", False)))
    logging.info("EventSleep2 experiment pipeline v%s", VERSION)
    try:
        if args.command in {"baseline-train", "baseline-evaluate", "final-proposed"}:
            seeds = list(args.member_seeds) if hasattr(args, "member_seeds") else list(args.seeds)
            if len(seeds) != PAPER_ENSEMBLE_SIZE and not bool(args.allow_nonpaper_ensemble):
                raise ValueError(
                    f"The paper uses exactly {PAPER_ENSEMBLE_SIZE} ensemble members; received {len(seeds)}. "
                    "Use --allow-nonpaper-ensemble only for a pilot."
                )
        if args.command in {"prepare-published", "prepare-timestamp-aligned"} and args.timestamp_dataset is not None:
            raise ValueError("Preparation takes --output; --timestamp-dataset selects an existing dataset for training or preflight")
        if args.command == "prepare-timestamp-aligned":
            report = timestamp_adapter().create_bundle(
                sys.modules[__name__], args.preprocess_root.resolve(), str(args.profile),
                args.eventsleep2_labels, args.output.resolve())
        elif args.command == "prepare-published":
            manifest = build_published_es2_cache(
                args.eventsleep2_root.resolve(),
                args.preprocess_root.resolve(),
                None if args.eventsleep2_labels is None else args.eventsleep2_labels.resolve(),
                str(args.profile),
                bool(args.overwrite_cache),
                args.recordings_limit,
            )
            cache_marker = json.loads(
                (
                    published_manifest_path(args.preprocess_root.resolve(), str(args.profile)).parent
                    / "completed_successfully.json"
                ).read_text(encoding="utf-8")
            )
            report = {
                "completed_utc": utc_now(),
                "pipeline_version": VERSION,
                "profile": args.profile,
                "samples": len(manifest),
                "partial": args.recordings_limit is not None,
                "rendered_frame_total": int(cache_marker["rendered_frame_total"]),
                "released_label_index_extent_total": int(
                    cache_marker["released_label_index_extent_total"]
                ),
                "all_released_indices_valid": bool(
                    cache_marker["all_released_indices_valid"]
                ),
                "dataset_fingerprint": str(cache_marker["dataset_fingerprint"]),
                "labels_sha256": str(cache_marker["labels_sha256"]),
                "release_inventory": cache_marker["release_inventory"],
                "multiply_claimed_physical_frame_positions": int(
                    cache_marker["multiply_claimed_physical_frame_positions"]
                ),
                "overlap_extra_sample_claims": int(
                    cache_marker["overlap_extra_sample_claims"]
                ),
                "physical_frame_claim_duplicate_rows": int(
                    cache_marker["physical_frame_claim_duplicate_rows"]
                ),
                "cache_version": PUBLISHED_CACHE_VERSION,
                "aedat_version": AUTHORS_AEDAT_VERSION,
                "preprocessing_dependencies": AUTHORS_PREPROCESS_DEPENDENCIES,
                "python_version": AUTHORS_PYTHON_VERSION,
            }
            atomic_json(output / "prepare_published_completed.json", report)
        elif args.command == "preflight":
            report = preflight(args)
        elif args.command == "baseline-train":
            report = train_published_baseline(args)
        elif args.command == "baseline-evaluate":
            report = evaluate_published_baseline(args)
        elif args.command == "tune-proposed":
            report = tune_proposed(args)
        elif args.command == "final-proposed":
            report = final_proposed(args)
        elif args.command == "model-check":
            report = model_check(args)
        elif args.command == "smoke-test":
            report = synthetic_smoke_test(output)
        else:
            parser.error(f"Unsupported command: {args.command}")
        logging.info("Command completed successfully: %s", report)
        return 0
    except Exception as exc:
        logging.error("Command failed: %s", exc)
        logging.debug("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
