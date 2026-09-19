#!/usr/bin/env python3
"""EventSleepFormer paper runtime.

This module contains the audited model, training, evaluation, and supporting
experiment utilities used to produce the final paper results. Public users
should normally invoke the reproducibility wrappers in ``scripts/``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


VERSION = "1.0.6"
STAGE_NAMES = {
    0: "HeadMove",
    1: "Hands2FaceHead",
    2: "RollLeft",
    3: "RollRight",
    4: "LegsShake",
    5: "ArmsShake",
    6: "LieLeft",
    7: "LieRight",
    8: "LieUp",
    9: "LieDown",
}
ES1_CLASSES = list(range(10))
ES2_MOVEMENT_CLASSES = list(range(6))
ES2_POSTURE_CLASSES = list(range(6, 10))
DEFAULT_SEEDS = [13, 37, 73, 101, 137]
ES1_VALIDATION_POLICIES = ("grouped_fold", "matched_resnete")
AUGMENTATION_POLICY = "deterministic_epoch_sample_v1"
DETERMINISM_POLICY = "strict_math_sdp_v1"
EVALUATION_SCOPES = ("test", "validation_only")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESS_ROOT = PROJECT_ROOT.parent / "data" / "processed" / "eventsleep1"
MAX_SEQUENCE_GAP_US = 30_000_000
MAX_LOCAL_GAP_US = 1_000_000
MAX_TIME_BIAS_GAP_SECONDS = 30.0
DEFAULT_METHODS = [
    "resnet18@ce",
    "resnet18@balanced_softmax",
    "mobilenet_v3@balanced_softmax",
    "cnn_tcn@balanced_softmax",
    "vit_scratch@balanced_softmax",
    "dinov2_linear@balanced_softmax",
    "dinov2_temporal@balanced_softmax",
    "eventsleepformer@balanced_softmax",
]

# Published ResNet-E protocol from the EventSleep authors' released code.
# These constants are intentionally isolated from the EventSleepFormer defaults:
# changing a generic --epochs/--batch-size flag must never silently change the
# reproduction baseline.
RESNETE_TRAIN_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
RESNETE_VALIDATION_SUBJECTS = (11,)
RESNETE_TEST_SUBJECTS = (9, 12, 13, 14)
RESNETE_CONFIGURATIONS = (1, 2, 3)
RESNETE_CHUNK_US = 150_000
RESNETE_HISTORY_US = 512_000
RESNETE_K = 1
RESNETE_STEP_PIXELS = 2
RESNETE_SENSOR_HEIGHT = 480
RESNETE_SENSOR_WIDTH = 640
RESNETE_OUTPUT_HEIGHT = 180
RESNETE_OUTPUT_WIDTH = 250
RESNETE_BATCH_SIZE = 32
RESNETE_EPOCHS = 10
RESNETE_LEARNING_RATE = 1e-3
RESNETE_WEIGHT_DECAY = 0.01
RESNETE_FREQUENCIES = (
    0.0759,
    0.1512,
    0.1576,
    0.0749,
    0.0820,
    0.0868,
    0.0920,
    0.0503,
    0.1845,
    0.0448,
)
RESNETE_FLIPPED_LABELS = {2: 3, 3: 2, 6: 7, 7: 6}
RESNETE_PROFILES = ("paper_intended", "authors_code_exact")
RESNETE_SOURCE_URLS = {
    "paper": "https://arxiv.org/abs/2404.01801",
    "repository": "https://github.com/ropertunizar/EventSleep",
    "training": "https://github.com/ropertunizar/EventSleep/blob/main/train_ResNet-E.py",
    "testing": "https://github.com/ropertunizar/EventSleep/blob/main/test_ResNet-E.py",
    "preprocessing": "https://github.com/ropertunizar/EventSleep/blob/main/events_to_frames.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def configure_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def software_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "timm",
    ]:
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            packages[name] = f"unavailable: {type(exc).__name__}: {exc}"
    environment: dict[str, Any] = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        import torch

        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            environment["gpu_name"] = torch.cuda.get_device_name(0)
            environment["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return environment


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and configure the CUDA attention backend deterministically.

    The audited release disables Flash and memory-efficient SDPA when deterministic=True and
    forces PyTorch's math SDPA path.  This avoids the non-deterministic Flash
    Attention backward warning observed in v1.0.5.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    if deterministic:
        # Required by deterministic CUDA GEMM on supported CUDA versions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            with contextlib.suppress(Exception):
                torch.backends.cuda.matmul.allow_tf32 = False
            with contextlib.suppress(Exception):
                torch.backends.cudnn.allow_tf32 = False
            # Explicitly avoid the non-deterministic fused SDPA kernels.
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_flash_sdp(False)
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True, warn_only=False)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(False)
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_flash_sdp(True)
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            with contextlib.suppress(Exception):
                torch.backends.cuda.enable_math_sdp(True)
    except ImportError:
        pass


def require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def contiguous_blocks(
    frame_indices: np.ndarray,
    timestamps_us: np.ndarray,
    max_gap_us: int = MAX_SEQUENCE_GAP_US,
) -> list[tuple[int, int]]:
    """Return half-open blocks that never cross index or timestamp gaps."""
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    timestamps = np.asarray(timestamps_us, dtype=np.int64).reshape(-1)
    if len(indices) != len(timestamps):
        raise ValueError("frame_indices and timestamps_us must have equal length")
    if len(indices) == 0:
        return []
    index_step = np.diff(indices)
    time_step = np.diff(timestamps)
    breaks = np.flatnonzero(
        (index_step != 1) | (time_step < 0) | (time_step > int(max_gap_us))
    ) + 1
    boundaries = np.concatenate(([0], breaks, [len(indices)]))
    return [
        (int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if stop > start
    ]


def parse_method_spec(spec: str) -> tuple[str, str]:
    if "@" in spec:
        method, loss = spec.split("@", 1)
    else:
        method = spec
        loss = "balanced_softmax" if method not in {"resnet18", "mobilenet_v3", "vit_scratch"} else "ce"
    allowed_methods = {
        "resnet18",
        "mobilenet_v3",
        "cnn_tcn",
        "vit_scratch",
        "dinov2_linear",
        "dinov2_temporal",
        "eventsleepformer",
    }
    allowed_losses = {"ce", "weighted_ce", "balanced_softmax", "focal"}
    if method not in allowed_methods:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(allowed_methods)}")
    if loss not in allowed_losses:
        raise ValueError(f"Unknown loss {loss!r}; choose from {sorted(allowed_losses)}")
    return method, loss


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return requested


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)


# ---------------------------------------------------------------------------
# Preprocessed data validation and EventSleep1 temporal cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataPaths:
    preprocess_root: Path

    @property
    def es1_manifest(self) -> Path:
        return self.preprocess_root / "eventsleep1" / "manifests" / "all_samples.csv"

    @property
    def es2_manifest(self) -> Path:
        return self.preprocess_root / "eventsleep2" / "manifests" / "all_frames.csv"

    @property
    def completion_marker(self) -> Path:
        return self.preprocess_root / "completed_successfully.json"

    @property
    def es1_sequence_manifest(self) -> Path:
        return self.preprocess_root / "eventsleep1" / "temporal_sequences" / "manifest.csv"

    @property
    def es2_descriptor_dir(self) -> Path:
        return self.preprocess_root / "eventsleep2" / "descriptors"

    def resnete_root(self, profile: str) -> Path:
        if profile not in RESNETE_PROFILES:
            raise ValueError(f"Unknown ResNet-E profile: {profile}")
        return self.preprocess_root / "eventsleep1" / "resnet_e_published" / profile

    def resnete_manifest(self, profile: str) -> Path:
        return self.resnete_root(profile) / "manifest.csv"


def validate_preprocessed(paths: DataPaths, full_array_check: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not paths.completion_marker.exists():
        errors.append(f"Missing verification marker: {paths.completion_marker}")
    else:
        marker = json.loads(paths.completion_marker.read_text(encoding="utf-8"))
        if marker.get("verification") != "PASSED":
            errors.append("Preprocessing verification marker does not say PASSED")
        if marker.get("datasets_kept_separate") is not True:
            errors.append("Preprocessing marker does not confirm separate datasets")

    if not paths.es1_manifest.exists():
        errors.append(f"Missing EventSleep1 manifest: {paths.es1_manifest}")
        es1 = pd.DataFrame()
    else:
        es1 = pd.read_csv(paths.es1_manifest, low_memory=False)
        require_columns(
            es1,
            ["sample_id", "split", "subject", "label", "frame_path", "source_path"],
            "EventSleep1 manifest",
        )
        if es1["sample_id"].duplicated().any():
            errors.append("EventSleep1 sample_id is not unique")
        train_subjects = set(es1.loc[es1["split"] == "TRAIN", "subject"].astype(int))
        test_subjects = set(es1.loc[es1["split"] == "TEST", "subject"].astype(int))
        if train_subjects & test_subjects:
            errors.append(f"EventSleep1 subject leakage: {sorted(train_subjects & test_subjects)}")
        if len(es1) != 995:
            warnings.append(f"EventSleep1 release normally has 995 clips; found {len(es1)}")

    if not paths.es2_manifest.exists():
        errors.append(f"Missing EventSleep2 manifest: {paths.es2_manifest}")
        es2 = pd.DataFrame()
    else:
        es2 = pd.read_csv(paths.es2_manifest, low_memory=False)
        require_columns(
            es2,
            [
                "frame_key",
                "recording_key",
                "subject",
                "sequence",
                "array_index",
                "frame_path",
                "anchor_timestamp_us",
                "supervised_movement",
                "movement_label",
                "supervised_posture",
                "posture_label",
            ],
            "EventSleep2 manifest",
        )
        if es2["frame_key"].duplicated().any():
            errors.append("EventSleep2 frame_key is not unique")
        movement = bool_series(es2["supervised_movement"])
        if movement.sum() != 17569:
            warnings.append(f"Expected 17,569 supervised movement frames; found {int(movement.sum())}")
        if sorted(es2["subject"].astype(int).unique().tolist()) != [1, 2, 3]:
            errors.append("EventSleep2 subjects are not exactly [1, 2, 3]")

    checked_arrays = 0
    if full_array_check and not es1.empty:
        for relpath in es1["frame_path"]:
            path = paths.preprocess_root / str(relpath)
            if not path.exists():
                errors.append(f"Missing EventSleep1 frame: {path}")
            checked_arrays += 1
    if full_array_check and not es2.empty:
        for relpath in es2["frame_path"].drop_duplicates():
            path = paths.preprocess_root / str(relpath)
            if not path.exists():
                errors.append(f"Missing EventSleep2 frame array: {path}")
            checked_arrays += 1

    report = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "preprocess_root": str(paths.preprocess_root.resolve()),
        "eventsleep1_samples": int(len(es1)),
        "eventsleep2_frames": int(len(es2)),
        "eventsleep2_movement_supervised": int(bool_series(es2["supervised_movement"]).sum()) if len(es2) else 0,
        "eventsleep2_posture_supervised": int(bool_series(es2["supervised_posture"]).sum()) if len(es2) else 0,
        "arrays_checked": checked_arrays,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return report


def _find_event_field(names: Sequence[str] | None, candidates: Sequence[str]) -> str:
    if not names:
        raise ValueError("Raw EventSleep1 clip is not a structured event array")
    lookup = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing event field {candidates}; found {names}")


def _structured_events(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = events.dtype.names
    x = np.asarray(events[_find_event_field(names, ["x"])], dtype=np.int32).reshape(-1)
    y = np.asarray(events[_find_event_field(names, ["y"])], dtype=np.int32).reshape(-1)
    t = np.asarray(events[_find_event_field(names, ["t", "timestamp", "ts"])], dtype=np.int64).reshape(-1)
    p = np.asarray(events[_find_event_field(names, ["p", "on", "polarity"])], dtype=np.int8).reshape(-1)
    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError("Event columns have inconsistent lengths")
    if np.any((p != 0) & (p != 1)):
        raise ValueError(f"Polarity is not binary: {np.unique(p).tolist()}")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    return x, y, t, p


def _update_latest(
    latest: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    roi: tuple[int, int, int, int],
    height: int,
    width: int,
) -> int:
    if len(t) == 0:
        return 0
    x0, y0, x1, y1 = roi
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (x >= 0) & (x < 640) & (y >= 0) & (y < 480)
    if not np.any(keep):
        return 0
    xv, yv, tv, pv = x[keep] - x0, y[keep] - y0, t[keep], p[keep]
    mx = np.minimum((xv * width) // (x1 - x0), width - 1)
    my = np.minimum((yv * height) // (y1 - y0), height - 1)
    pixel = (my * width + mx).astype(np.int64, copy=False)
    indices = pv.astype(np.int64, copy=False) * height * width + pixel
    np.maximum.at(latest, indices, tv)
    return int(keep.sum())


def _surface(latest: np.ndarray, anchor: int, history_us: int, height: int, width: int) -> np.ndarray:
    lower = anchor - history_us
    delta = np.zeros_like(latest, dtype=np.int64)
    valid = latest > lower
    delta[valid] = np.clip(np.minimum(latest[valid], anchor) - lower, 0, history_us)
    return (delta.astype(np.float32) / float(history_us)).astype(np.float16).reshape(2, height, width)


def frame_descriptors(frames: np.ndarray, timestamps: np.ndarray | None = None) -> np.ndarray:
    """Return stable, label-free descriptors for [T,2,H,W] time surfaces."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(f"Expected [T,2,H,W], found {values.shape}")
    t_count, _, height, width = values.shape
    active = (values > 0).mean(axis=(1, 2, 3))
    neg = values[:, 0].sum(axis=(1, 2))
    pos = values[:, 1].sum(axis=(1, 2))
    polarity = (pos - neg) / (pos + neg + 1e-6)
    recency = values.mean(axis=(1, 2, 3))
    magnitude = values.sum(axis=1)
    mass = magnitude.sum(axis=(1, 2)) + 1e-6
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :]
    cx = (magnitude * xx).sum(axis=(1, 2)) / mass
    cy = (magnitude * yy).sum(axis=(1, 2)) / mass
    spread = (
        (magnitude * (xx - cx[:, None, None]) ** 2).sum(axis=(1, 2))
        + (magnitude * (yy - cy[:, None, None]) ** 2).sum(axis=(1, 2))
    ) / mass
    change = np.zeros(t_count, dtype=np.float32)
    if t_count > 1:
        change[1:] = np.abs(values[1:] - values[:-1]).mean(axis=(1, 2, 3))
    return np.stack([active, polarity, recency, spread, change], axis=1).astype(np.float32)


def _build_es1_sequence_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(task["source_path"]))
    output = Path(str(task["output_path"]))
    seq_len = int(task["sequence_length"])
    history_us = int(task["history_us"])
    height = int(task["height"])
    width = int(task["width"])
    overwrite = bool(task["overwrite"])
    expected_shape = (seq_len, 2, height, width)
    if output.exists() and not overwrite:
        try:
            cached = np.load(output, allow_pickle=False)
            if tuple(cached["frames"].shape) == expected_shape and tuple(cached["timestamps_us"].shape) == (seq_len,):
                return {"sequence_path": str(task["relative_output"]), "status": "reused"}
        except Exception:
            pass
        raise RuntimeError(f"Stale or invalid sequence cache exists: {output}; use --overwrite")

    events = np.load(source, mmap_mode="r", allow_pickle=False)
    x, y, t, p = _structured_events(events)
    if len(t) == 0:
        raise ValueError(f"No events in {source}")
    anchors = np.rint(np.linspace(int(t.min()), int(t.max()), seq_len)).astype(np.int64)
    roi = tuple(int(task[name]) for name in ["roi_x0", "roi_y0", "roi_x1", "roi_y1"])
    latest = np.full(2 * height * width, -10**18, dtype=np.int64)
    frames = np.empty(expected_shape, dtype=np.float16)
    event_counts = np.zeros(seq_len, dtype=np.int64)
    cursor = 0
    previous = 0
    for index, anchor in enumerate(anchors):
        cursor = int(np.searchsorted(t, anchor, side="right"))
        event_counts[index] = cursor - previous
        _update_latest(latest, x[previous:cursor], y[previous:cursor], t[previous:cursor], p[previous:cursor], roi, height, width)
        frames[index] = _surface(latest, int(anchor), history_us, height, width)
        previous = cursor
    descriptors = frame_descriptors(frames, anchors)
    atomic_npz(
        output,
        frames=frames,
        timestamps_us=anchors,
        descriptors=descriptors,
        events_since_previous_anchor=event_counts,
    )
    return {"sequence_path": str(task["relative_output"]), "status": "created"}


def build_es1_sequences(
    paths: DataPaths,
    sequence_length: int,
    history_us: int,
    workers: int,
    overwrite: bool,
    height: int = 120,
    width: int = 160,
) -> pd.DataFrame:
    import concurrent.futures

    manifest = pd.read_csv(paths.es1_manifest, low_memory=False)
    require_columns(
        manifest,
        ["sample_id", "source_path", "roi_x0", "roi_y0", "roi_x1", "roi_y1"],
        "EventSleep1 manifest",
    )
    cache_root = paths.preprocess_root / "eventsleep1" / "temporal_sequences"
    tasks: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        relative = Path("eventsleep1") / "temporal_sequences" / "arrays" / f"{row['sample_id']}.npz"
        tasks.append(
            {
                **row,
                "output_path": str(paths.preprocess_root / relative),
                "relative_output": relative.as_posix(),
                "sequence_length": sequence_length,
                "history_us": history_us,
                "height": height,
                "width": width,
                "overwrite": overwrite,
            }
        )
    logging.info("Building EventSleep1 temporal cache: %d clips, %d frames/clip", len(tasks), sequence_length)
    results: list[dict[str, Any]] = [dict() for _ in tasks]
    executor_class = concurrent.futures.ProcessPoolExecutor if workers > 1 else None
    if executor_class is None:
        for index, task in enumerate(tasks):
            results[index] = _build_es1_sequence_worker(task)
            if (index + 1) % 25 == 0 or index + 1 == len(tasks):
                logging.info("EventSleep1 temporal clips: %d/%d", index + 1, len(tasks))
    else:
        with executor_class(max_workers=workers) as executor:
            future_map = {executor.submit(_build_es1_sequence_worker, task): index for index, task in enumerate(tasks)}
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                results[index] = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    logging.info("EventSleep1 temporal clips: %d/%d", completed, len(tasks))
    output_manifest = manifest.copy()
    output_manifest["sequence_path"] = [item["sequence_path"] for item in results]
    output_manifest["sequence_length"] = sequence_length
    output_manifest["history_us"] = history_us
    atomic_csv(paths.es1_sequence_manifest, output_manifest)
    atomic_json(
        cache_root / "completed_successfully.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "clips": len(output_manifest),
            "sequence_length": sequence_length,
            "history_us": history_us,
            "datasets_kept_separate": True,
        },
    )
    return output_manifest


# ---------------------------------------------------------------------------
# Published ResNet-E preprocessing reproduction (EventSleep1 only)
# ---------------------------------------------------------------------------


def resnete_profile_settings(profile: str) -> dict[str, Any]:
    """Return immutable choices for a documented ResNet-E reproduction.

    ``paper_intended`` is the controlled baseline used in our main comparison:
    subject-specific bed crops, a horizontal left/right flip on training data,
    validation in evaluation mode, and explicit clip IDs during aggregation.

    ``authors_code_exact`` is a diagnostic that mirrors three observable quirks
    in the released scripts: the final subject variable controls every crop,
    the height axis is flipped while left/right labels are swapped, and the
    augmented validation pass runs with BatchNorm in training mode.  It must
    never be presented as the leakage-safe primary result.
    """
    if profile not in RESNETE_PROFILES:
        raise ValueError(f"Unknown ResNet-E profile {profile!r}; choose from {RESNETE_PROFILES}")
    if profile == "paper_intended":
        return {
            "profile": profile,
            "event_pixel_selection": "last_event_per_pixel_and_window",
            "crop_policy": "subject_specific_day1_day2",
            "flip_axis": "width_horizontal",
            "augment_training": True,
            "augment_validation": False,
            "validation_batchnorm_mode": "eval",
            "primary_clip_grouping": "explicit_sample_id",
            "scientific_role": "primary_controlled_baseline",
        }
    return {
        "profile": profile,
        "event_pixel_selection": "released_events_to_frame_function",
        "crop_policy": "released_loop_variable_day2_for_all_clips",
        "flip_axis": "height_as_released",
        "augment_training": True,
        "augment_validation": True,
        "validation_batchnorm_mode": "train_as_released",
        "primary_clip_grouping": "explicit_sample_id_with_label_run_metric_also_reported",
        "scientific_role": "reproducibility_diagnostic_not_primary",
    }


def _resnete_crop_for_subject(subject: int, profile: str) -> tuple[int, int, int, int]:
    settings = resnete_profile_settings(profile)
    crop_subject = 10 if settings["crop_policy"].startswith("released_loop_variable") else int(subject)
    if crop_subject in {1, 2, 3, 4}:
        return (95, 60, 595, 420)
    return (95, 80, 595, 440)


def _resnete_deduplicate_events(total_events: np.ndarray, minimum_time_us: int) -> np.ndarray:
    """Mirror the authors' reverse/bin/unique event suppression."""
    if len(total_events) == 0:
        return total_events
    reversed_events = np.asarray(total_events[::-1], dtype=np.int64).copy()
    original_time = reversed_events[:, 2].copy()
    reversed_events[:, 2] -= np.mod(reversed_events[:, 2], int(minimum_time_us))
    _, indices = np.unique(reversed_events, return_index=True, axis=0)
    selected = reversed_events[indices].copy()
    selected[:, 2] = original_time[indices]
    return selected


def _resnete_events_to_frame(
    events: np.ndarray,
    unique_coordinates: np.ndarray,
    unique_starts: np.ndarray,
    height: int,
    width: int,
    k: int,
    authors_code_exact: bool,
) -> np.ndarray:
    """Create one polarity FIFO update, optionally matching released code."""
    # Keep absolute microsecond timestamps in float64 until the per-window
    # origin is subtracted.  AEDAT clocks can be around 1e15; float32 would
    # destroy the 150-ms differences that define the representation.
    output = np.zeros(height * width * k, dtype=np.float64)
    if len(unique_coordinates) == 0:
        return output.reshape(height, width, k)
    if authors_code_exact:
        # This is a literal, guarded implementation of the active
        # events_to_frame() function in the released repository.  It is kept
        # only for the diagnostic profile; the main profile uses the intended
        # last-k-events behavior below.
        true_indices: list[np.ndarray] = []
        k_indices: list[np.ndarray] = []
        previous = -1
        for start in unique_starts.astype(int):
            current = 1 + np.arange(max(previous, start - k, 0), start, dtype=np.int64)
            true_indices.append(current)
            k_indices.append(np.arange(k - len(current), k, dtype=np.int64))
            previous = int(start)
        selected_indices = np.concatenate(true_indices) if true_indices else np.empty(0, dtype=np.int64)
        selected_k = np.concatenate(k_indices) if k_indices else np.empty(0, dtype=np.int64)
        if len(selected_indices):
            selected = events[selected_indices]
            coordinates = np.column_stack(
                [selected[:, 1].astype(int), selected[:, 0].astype(int), selected_k]
            )
            flat = np.ravel_multi_index(coordinates.T, (height, width, k))
            output[flat] = selected[:, 2].astype(np.float64)
        return output.reshape(height, width, k)

    stops = np.concatenate([unique_starts[1:], np.array([len(events)], dtype=np.int64)])
    for coordinate, start, stop in zip(unique_coordinates, unique_starts, stops):
        values = events[int(start) : int(stop), 2]
        chosen = values[-k:]
        y, x = int(coordinate[1]), int(coordinate[0])
        output_values = np.zeros(k, dtype=np.float64)
        output_values[-len(chosen) :] = chosen.astype(np.float64)
        base = (y * width + x) * k
        output[base : base + k] = output_values
    return output.reshape(height, width, k)


def _resnete_window_representation(
    events: np.ndarray,
    height: int,
    width: int,
    k: int,
    authors_code_exact: bool,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    for polarity in (1, 0):
        selected = events[events[:, 3] == polarity]
        if len(selected):
            selected = selected[np.lexsort((selected[:, 2], selected[:, 1], selected[:, 0]))]
            coordinates, starts = np.unique(selected[:, :2], return_index=True, axis=0)
        else:
            coordinates = np.empty((0, 2), dtype=np.int64)
            starts = np.empty(0, dtype=np.int64)
        outputs.append(
            _resnete_events_to_frame(
                selected,
                coordinates,
                starts,
                height,
                width,
                k,
                authors_code_exact,
            )
        )
    positive, negative = outputs
    return positive, negative


def _resnete_process_event_stream(
    total_events: np.ndarray,
    profile: str,
    height: int = RESNETE_SENSOR_HEIGHT,
    width: int = RESNETE_SENSOR_WIDTH,
    chunk_us: int = RESNETE_CHUNK_US,
    k: int = RESNETE_K,
    history_us: int = RESNETE_HISTORY_US,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the 150-ms, two-polarity time surfaces used by ResNet-E."""
    if len(total_events) == 0:
        raise ValueError("Cannot build ResNet-E frames from an empty event clip")
    settings = resnete_profile_settings(profile)
    events = _resnete_deduplicate_events(
        np.asarray(total_events, dtype=np.int64),
        max(1, int(chunk_us // k)),
    )
    if len(events) == 0:
        raise ValueError("All events were removed by ResNet-E temporal suppression")
    time_window_ends = np.arange(
        int(events[:, 2].max()),
        int(events[:, 2].min()),
        -int(chunk_us),
        dtype=np.int64,
    )[::-1]
    if len(time_window_ends) == 0:
        time_window_ends = np.array([int(events[:, 2].max())], dtype=np.int64)
    positive_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)
    negative_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)
    previous_end: float | int = -np.inf
    frames: list[np.ndarray] = []
    anchors: list[int] = []
    code_exact = profile == "authors_code_exact"
    for current_end in time_window_ends:
        selected = events[(events[:, 2] > previous_end) & (events[:, 2] <= int(current_end))]
        positive, negative = _resnete_window_representation(
            selected,
            height,
            width,
            k,
            authors_code_exact=code_exact,
        )
        if selected.size == 0 or (float(positive.sum()) == 0.0 and float(negative.sum()) == 0.0):
            continue
        positive_fifo = np.sort(
            np.concatenate([positive_fifo, positive.astype(np.float64)], axis=2), axis=2
        )[:, :, -k:]
        negative_fifo = np.sort(
            np.concatenate([negative_fifo, negative.astype(np.float64)], axis=2), axis=2
        )[:, :, -k:]
        frames.append(np.stack([negative_fifo, positive_fifo], axis=-1))
        anchors.append(int(current_end))
        previous_end = int(current_end)
    if not frames:
        raise ValueError("ResNet-E frame construction produced no non-empty windows")
    frame_array = np.stack(frames).astype(np.float64, copy=False)
    anchor_array = np.asarray(anchors, dtype=np.int64)
    difference = (int(history_us) - anchor_array).astype(np.float64)[:, None, None, None, None]
    frame_array = frame_array + difference
    frame_array[frame_array < 0] = 0
    frame_array /= float(history_us)
    if not np.isfinite(frame_array).all():
        raise FloatingPointError("Non-finite values in ResNet-E time surfaces")
    return frame_array, anchor_array


def _build_resnete_clip_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(task["source_path"]))
    frame_path = Path(str(task["frame_output_path"]))
    timestamp_path = Path(str(task["timestamp_output_path"]))
    metadata_path = Path(str(task["metadata_output_path"]))
    profile = str(task["profile"])
    overwrite = bool(task["overwrite"])
    stat = source.stat()
    signature_payload = {
        "algorithm": "eventsleepformer-resnete-v1",
        "profile": profile,
        "source_path": str(source.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "chunk_us": RESNETE_CHUNK_US,
        "history_us": RESNETE_HISTORY_US,
        "k": RESNETE_K,
        "step_pixels": RESNETE_STEP_PIXELS,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if frame_path.exists() and timestamp_path.exists() and metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            raise RuntimeError(f"Stale ResNet-E cache exists for {source}; use --overwrite-cache")
        frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
        if (
            frames.ndim == 4
            and tuple(frames.shape[1:]) == (2, RESNETE_OUTPUT_HEIGHT, RESNETE_OUTPUT_WIDTH)
            and len(frames) == len(timestamps)
            and str(frames.dtype) == "float16"
        ):
            return {
                "resnete_frame_path": str(task["relative_frame_path"]),
                "resnete_timestamp_path": str(task["relative_timestamp_path"]),
                "resnete_frame_count": int(len(frames)),
                "cache_status": "reused",
            }
        raise RuntimeError(f"Invalid ResNet-E cache exists for {source}; use --overwrite-cache")

    raw = np.load(source, mmap_mode="r", allow_pickle=False)
    x, y, timestamp, polarity = _structured_events(raw)
    total_events = np.column_stack([x, y, timestamp, polarity]).astype(np.int64, copy=False)
    full_frames, anchors = _resnete_process_event_stream(total_events, profile)
    x0, y0, x1, y1 = _resnete_crop_for_subject(int(task["subject"]), profile)
    cropped = full_frames[
        :,
        y0:y1:RESNETE_STEP_PIXELS,
        x0:x1:RESNETE_STEP_PIXELS,
        :,
        :,
    ]
    cropped = cropped.reshape(len(cropped), RESNETE_OUTPUT_HEIGHT, RESNETE_OUTPUT_WIDTH, 2)
    cropped = cropped.transpose(0, 3, 1, 2).astype(np.float16)
    if tuple(cropped.shape[1:]) != (2, RESNETE_OUTPUT_HEIGHT, RESNETE_OUTPUT_WIDTH):
        raise RuntimeError(f"Unexpected ResNet-E crop shape for {source}: {cropped.shape}")
    atomic_npy(frame_path, cropped)
    atomic_npy(timestamp_path, anchors.astype(np.int64))
    atomic_json(
        metadata_path,
        {
            "created_utc": utc_now(),
            "pipeline_version": VERSION,
            "signature": signature,
            "signature_payload": signature_payload,
            "frames": int(len(cropped)),
            "shape": list(cropped.shape),
            "crop": [x0, y0, x1, y1],
            "profile_settings": resnete_profile_settings(profile),
        },
    )
    return {
        "resnete_frame_path": str(task["relative_frame_path"]),
        "resnete_timestamp_path": str(task["relative_timestamp_path"]),
        "resnete_frame_count": int(len(cropped)),
        "cache_status": "created",
    }


def build_published_resnete_cache(
    paths: DataPaths,
    profile: str,
    workers: int,
    overwrite: bool,
) -> pd.DataFrame:
    import concurrent.futures

    settings = resnete_profile_settings(profile)
    source_manifest = pd.read_csv(paths.es1_manifest, low_memory=False)
    require_columns(
        source_manifest,
        ["sample_id", "source_path", "split", "subject", "configuration", "label"],
        "EventSleep1 manifest",
    )
    subjects = set(source_manifest["subject"].astype(int))
    expected_subjects = set(RESNETE_TRAIN_SUBJECTS) | set(RESNETE_VALIDATION_SUBJECTS) | set(RESNETE_TEST_SUBJECTS)
    if subjects != expected_subjects:
        raise RuntimeError(
            f"Published ResNet-E split requires subjects {sorted(expected_subjects)}, found {sorted(subjects)}"
        )
    configurations = set(source_manifest["configuration"].astype(int))
    if configurations != set(RESNETE_CONFIGURATIONS):
        raise RuntimeError(
            f"Published ResNet-E protocol requires configurations {RESNETE_CONFIGURATIONS}, found {sorted(configurations)}"
        )
    cache_root = paths.resnete_root(profile)
    tasks: list[dict[str, Any]] = []
    for row in source_manifest.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        relative_frame = Path("eventsleep1") / "resnet_e_published" / profile / "arrays" / f"{sample_id}.npy"
        relative_timestamp = Path("eventsleep1") / "resnet_e_published" / profile / "timestamps" / f"{sample_id}.npy"
        metadata = cache_root / "metadata" / f"{sample_id}.json"
        tasks.append(
            {
                **row,
                "profile": profile,
                "frame_output_path": str(paths.preprocess_root / relative_frame),
                "timestamp_output_path": str(paths.preprocess_root / relative_timestamp),
                "metadata_output_path": str(metadata),
                "relative_frame_path": relative_frame.as_posix(),
                "relative_timestamp_path": relative_timestamp.as_posix(),
                "overwrite": overwrite,
            }
        )
    logging.info(
        "Building published ResNet-E cache (%s): %d clips with %d worker(s)",
        profile,
        len(tasks),
        workers,
    )
    results: list[dict[str, Any]] = [dict() for _ in tasks]
    if workers <= 1:
        for index, task in enumerate(tasks):
            results[index] = _build_resnete_clip_worker(task)
            if (index + 1) % 25 == 0 or index + 1 == len(tasks):
                logging.info("ResNet-E cache clips: %d/%d", index + 1, len(tasks))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_build_resnete_clip_worker, task): index
                for index, task in enumerate(tasks)
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                results[index] = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    logging.info("ResNet-E cache clips: %d/%d", completed, len(tasks))
    output = source_manifest.copy()
    for column in ["resnete_frame_path", "resnete_timestamp_path", "resnete_frame_count", "cache_status"]:
        output[column] = [result[column] for result in results]
    output["resnete_profile"] = profile
    output["resnete_chunk_us"] = RESNETE_CHUNK_US
    output["resnete_history_us"] = RESNETE_HISTORY_US
    output["resnete_k"] = RESNETE_K
    output["resnete_height"] = RESNETE_OUTPUT_HEIGHT
    output["resnete_width"] = RESNETE_OUTPUT_WIDTH
    atomic_csv(paths.resnete_manifest(profile), output)
    atomic_json(
        cache_root / "completed_successfully.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "profile": profile,
            "profile_settings": settings,
            "clips": int(len(output)),
            "frames": int(output["resnete_frame_count"].sum()),
            "paper_reported_clips": 1016,
            "current_release_clips": int(len(output)),
            "exact_paper_data_cardinality": bool(len(output) == 1016),
            "published_constants": published_resnete_configuration(),
            "source_urls": RESNETE_SOURCE_URLS,
        },
    )
    return output


def verify_published_resnete_cache(
    paths: DataPaths,
    profile: str,
    full_array_check: bool = False,
) -> dict[str, Any]:
    manifest_path = paths.resnete_manifest(profile)
    marker_path = paths.resnete_root(profile) / "completed_successfully.json"
    if not manifest_path.exists() or not marker_path.exists():
        raise FileNotFoundError(
            f"Missing ResNet-E cache for profile {profile}; run resnet-e --stage prepare first"
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    require_columns(
        manifest,
        [
            "sample_id",
            "subject",
            "configuration",
            "label",
            "resnete_frame_path",
            "resnete_timestamp_path",
            "resnete_frame_count",
            "resnete_profile",
        ],
        "ResNet-E cache manifest",
    )
    if set(manifest["resnete_profile"].astype(str)) != {profile}:
        raise RuntimeError("ResNet-E cache manifest mixes profiles")
    checks = manifest if full_array_check else manifest.iloc[np.linspace(0, len(manifest) - 1, min(20, len(manifest)), dtype=int)]
    invalid: list[str] = []
    for row in checks.itertuples(index=False):
        frame_path = paths.preprocess_root / str(row.resnete_frame_path)
        timestamp_path = paths.preprocess_root / str(row.resnete_timestamp_path)
        try:
            frames = np.load(frame_path, mmap_mode="r", allow_pickle=False)
            timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
            if (
                tuple(frames.shape) != (
                    int(row.resnete_frame_count),
                    2,
                    RESNETE_OUTPUT_HEIGHT,
                    RESNETE_OUTPUT_WIDTH,
                )
                or len(timestamps) != len(frames)
                or str(frames.dtype) != "float16"
                or np.any(np.diff(timestamps) < 0)
            ):
                invalid.append(str(row.sample_id))
        except Exception as exc:
            invalid.append(f"{row.sample_id}: {type(exc).__name__}: {exc}")
    train_subjects = sorted(
        set(manifest[manifest["subject"].astype(int).isin(RESNETE_TRAIN_SUBJECTS)]["subject"].astype(int))
    )
    validation_subjects = sorted(
        set(manifest[manifest["subject"].astype(int).isin(RESNETE_VALIDATION_SUBJECTS)]["subject"].astype(int))
    )
    test_subjects = sorted(
        set(manifest[manifest["subject"].astype(int).isin(RESNETE_TEST_SUBJECTS)]["subject"].astype(int))
    )
    if train_subjects != list(RESNETE_TRAIN_SUBJECTS):
        invalid.append(f"train subjects: {train_subjects}")
    if validation_subjects != list(RESNETE_VALIDATION_SUBJECTS):
        invalid.append(f"validation subjects: {validation_subjects}")
    if test_subjects != list(RESNETE_TEST_SUBJECTS):
        invalid.append(f"test subjects: {test_subjects}")
    report = {
        "verified_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "profile_settings": resnete_profile_settings(profile),
        "clips": int(len(manifest)),
        "frames": int(pd.to_numeric(manifest["resnete_frame_count"], errors="raise").sum()),
        "arrays_checked": int(len(checks)),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "errors": invalid,
        "passed": not invalid,
    }
    if invalid:
        raise RuntimeError("ResNet-E cache verification failed:\n- " + "\n- ".join(invalid))
    return report


def build_es2_descriptors(paths: DataPaths, overwrite: bool = False) -> pd.DataFrame:
    manifest = pd.read_csv(paths.es2_manifest, low_memory=False)
    require_columns(manifest, ["recording_key", "frame_path", "array_index", "anchor_timestamp_us"], "EventSleep2 manifest")
    output_rows: list[dict[str, Any]] = []
    for index, (recording, group) in enumerate(manifest.groupby("recording_key", sort=True), start=1):
        group = group.sort_values("array_index", kind="stable")
        source = paths.preprocess_root / str(group["frame_path"].iloc[0])
        output = paths.es2_descriptor_dir / f"{recording}.npy"
        if output.exists() and not overwrite:
            descriptors = np.load(output, mmap_mode="r", allow_pickle=False)
            if tuple(descriptors.shape) != (len(group), 5):
                raise RuntimeError(f"Invalid descriptor cache {output}; use --overwrite")
        else:
            frames = np.load(source, mmap_mode="r", allow_pickle=False)
            timestamps = group["anchor_timestamp_us"].to_numpy(dtype=np.int64)
            descriptors = np.empty((len(frames), 5), dtype=np.float32)
            chunk_size = 256
            for start in range(0, len(frames), chunk_size):
                stop = min(start + chunk_size, len(frames))
                extended_start = max(0, start - 1)
                local = frame_descriptors(frames[extended_start:stop], timestamps[extended_start:stop])
                descriptors[start:stop] = local[start - extended_start :]
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, descriptors, allow_pickle=False)
            os.replace(temporary, output)
        output_rows.append({"recording_key": recording, "descriptor_path": str(output.relative_to(paths.preprocess_root)), "frames": len(group)})
        logging.info("EventSleep2 descriptors: %d/%d", index, manifest["recording_key"].nunique())
    table = pd.DataFrame(output_rows)
    atomic_csv(paths.es2_descriptor_dir / "manifest.csv", table)
    atomic_json(
        paths.es2_descriptor_dir / "completed_successfully.json",
        {"completed_utc": utc_now(), "pipeline_version": VERSION, "recordings": len(table), "descriptor_dimensions": 5},
    )
    return table


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------


def augmentation_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    """Deterministic RNG that changes by epoch and sample index.

    This keeps augmentations exactly reproducible for a fixed experiment seed
    while preventing a training sample from receiving the same transform on
    every epoch.  SeedSequence avoids fragile hand-written integer arithmetic.
    """
    if int(epoch) < 0:
        raise ValueError("augmentation epoch must be non-negative")
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(index), 0xE51E])
    return np.random.default_rng(sequence)


def _load_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install requirements_eventsleep.txt "
            "inside the CUDA-enabled Docker environment."
        ) from exc


class EventSleep1Dataset:
    def __init__(
        self,
        preprocess_root: Path,
        manifest: pd.DataFrame,
        augment: bool,
        seed: int,
    ) -> None:
        torch = _load_torch()
        self.torch = torch
        self.preprocess_root = preprocess_root
        self.frame = manifest.reset_index(drop=True).copy()
        self.augment = bool(augment)
        self.seed = int(seed)
        # Shared CPU tensor lets persistent DataLoader workers observe epoch
        # updates from the parent process without sacrificing reproducibility.
        self._augmentation_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self._augmentation_epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = self.preprocess_root / str(row["sequence_path"])
        with np.load(path, allow_pickle=False) as payload:
            frames = np.asarray(payload["frames"], dtype=np.float32)
            timestamps = np.asarray(payload["timestamps_us"], dtype=np.int64)
            descriptors = np.asarray(payload["descriptors"], dtype=np.float32)
        if self.augment:
            rng = augmentation_rng(self.seed, int(self._augmentation_epoch.item()), index)
            frames = augment_event_sequence(frames, rng)
            descriptors = frame_descriptors(frames, timestamps)
        return {
            "frames": self.torch.from_numpy(frames.copy()),
            "timestamps": self.torch.from_numpy(timestamps.copy()),
            "descriptors": self.torch.from_numpy(descriptors.copy()),
            "attention_mask": self.torch.ones(len(frames), dtype=self.torch.bool),
            "labels": self.torch.tensor(int(row["label"]), dtype=self.torch.long),
            "loss_mask": self.torch.tensor(True, dtype=self.torch.bool),
            "sample_id": str(row["sample_id"]),
            "subject": int(row["subject"]),
            "recording_key": f"subject{int(row['subject']):02d}_config{int(row['configuration'])}",
            "frame_indices": self.torch.arange(len(frames), dtype=self.torch.long),
        }


class PublishedResNetEFrameDataset:
    """Frame-level EventSleep1 dataset for the published ResNet-E protocol."""

    def __init__(
        self,
        preprocess_root: Path,
        manifest: pd.DataFrame,
        profile: str,
        augment: bool,
        mmap_cache_size: int = 12,
    ) -> None:
        torch_module = _load_torch()
        self.torch = torch_module
        self.preprocess_root = preprocess_root
        self.frame = manifest.reset_index(drop=True).copy()
        self.profile = profile
        self.settings = resnete_profile_settings(profile)
        self.augment = bool(augment)
        self.mmap_cache_size = max(1, int(mmap_cache_size))
        self._arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        self._timestamps: OrderedDict[str, np.ndarray] = OrderedDict()
        self.items: list[tuple[int, int, bool]] = []
        for row_index, row in self.frame.iterrows():
            count = int(row["resnete_frame_count"])
            if count <= 0:
                raise ValueError(f"ResNet-E clip has no frames: {row['sample_id']}")
            self.items.extend((int(row_index), frame_index, False) for frame_index in range(count))
            if self.augment:
                self.items.extend((int(row_index), frame_index, True) for frame_index in range(count))

    def __len__(self) -> int:
        return len(self.items)

    def _mmap(self, relative: str, cache: OrderedDict[str, np.ndarray]) -> np.ndarray:
        key = str(relative)
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        value = np.load(self.preprocess_root / key, mmap_mode="r", allow_pickle=False)
        cache[key] = value
        while len(cache) > self.mmap_cache_size:
            cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, frame_index, augmented = self.items[index]
        row = self.frame.iloc[row_index]
        frames = self._mmap(str(row["resnete_frame_path"]), self._arrays)
        timestamps = self._mmap(str(row["resnete_timestamp_path"]), self._timestamps)
        frame = np.asarray(frames[frame_index], dtype=np.float32)
        label = int(row["label"])
        if augmented:
            if self.settings["flip_axis"] == "width_horizontal":
                frame = frame[..., ::-1]
            else:
                frame = frame[..., ::-1, :]
            label = int(RESNETE_FLIPPED_LABELS.get(label, label))
        frame = np.ascontiguousarray(frame)
        subject = int(row["subject"])
        configuration = int(row["configuration"])
        return {
            "frames": self.torch.from_numpy(frame),
            "labels": self.torch.tensor(label, dtype=self.torch.long),
            "sample_id": str(row["sample_id"]),
            "subject": subject,
            "configuration": configuration,
            "recording_key": f"subject{subject:02d}_config{configuration}",
            "frame_index": int(frame_index),
            "timestamp_us": int(timestamps[frame_index]),
            "augmented": bool(augmented),
        }


class EventSleep2SequenceDataset:
    def __init__(
        self,
        preprocess_root: Path,
        manifest: pd.DataFrame,
        task: str,
        sequence_length: int,
        augment: bool,
        seed: int,
        include_recordings: set[str] | None = None,
    ) -> None:
        torch = _load_torch()
        self.torch = torch
        self.preprocess_root = preprocess_root
        self.task = task
        self.sequence_length = int(sequence_length)
        self.target_stride = max(1, self.sequence_length // 2)
        self.context_length = self.sequence_length - self.target_stride
        self.augment = bool(augment)
        self.seed = int(seed)
        self._augmentation_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        frame = manifest.copy()
        if include_recordings is not None:
            frame = frame[frame["recording_key"].astype(str).isin(include_recordings)].copy()
        frame = frame.sort_values(["recording_key", "array_index"], kind="stable")
        self.groups: dict[str, pd.DataFrame] = {
            str(key): group.reset_index(drop=True)
            for key, group in frame.groupby("recording_key", sort=True)
        }
        self.items: list[tuple[str, int, int, int, int]] = []
        for key, group in self.groups.items():
            frame_indices = group["array_index"].to_numpy(dtype=np.int64)
            timestamps = group["anchor_timestamp_us"].to_numpy(dtype=np.int64)
            for segment_start, segment_stop in contiguous_blocks(frame_indices, timestamps):
                for target_start in range(segment_start, segment_stop, self.target_stride):
                    target_stop = min(target_start + self.target_stride, segment_stop)
                    start = max(segment_start, target_start - self.context_length)
                    stop = min(start + self.sequence_length, segment_stop)
                    if stop == segment_stop:
                        start = max(segment_start, stop - self.sequence_length)
                    self.items.append((key, start, stop, target_start, target_stop))

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self._augmentation_epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key, start, stop, target_start, target_stop = self.items[index]
        group = self.groups[key].iloc[start:stop].copy()
        frame_path = self.preprocess_root / str(group["frame_path"].iloc[0])
        array = np.load(frame_path, mmap_mode="r", allow_pickle=False)
        array_indices = group["array_index"].to_numpy(dtype=np.int64)
        frames = np.asarray(array[array_indices], dtype=np.float32)
        timestamps = group["anchor_timestamp_us"].to_numpy(dtype=np.int64)
        descriptor_path = self.preprocess_root / "eventsleep2" / "descriptors" / f"{key}.npy"
        if descriptor_path.exists():
            all_descriptors = np.load(descriptor_path, mmap_mode="r", allow_pickle=False)
            descriptors = np.asarray(all_descriptors[array_indices], dtype=np.float32)
        else:
            descriptors = frame_descriptors(frames, timestamps)

        if self.task == "movement":
            supervised = bool_series(group["supervised_movement"]).to_numpy(dtype=bool)
            raw_labels = pd.to_numeric(group["movement_label"], errors="coerce").fillna(-100).to_numpy(dtype=np.int64)
            labels = np.where(supervised, raw_labels, -100)
        elif self.task == "posture":
            supervised = bool_series(group["supervised_posture"]).to_numpy(dtype=bool)
            raw_labels = pd.to_numeric(group["posture_label"], errors="coerce").fillna(-100).to_numpy(dtype=np.int64)
            labels = np.where(supervised, raw_labels - 6, -100)
        else:
            raise ValueError(f"Unsupported EventSleep2 task: {self.task}")

        valid_length = len(frames)
        if valid_length < self.sequence_length:
            pad = self.sequence_length - valid_length
            frames = np.pad(frames, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="constant")
            timestamps = np.pad(timestamps, (0, pad), mode="edge")
            descriptors = np.pad(descriptors, ((0, pad), (0, 0)), mode="constant")
            labels = np.pad(labels, (0, pad), constant_values=-100)
            array_indices = np.pad(array_indices, (0, pad), constant_values=-1)
        attention = np.zeros(self.sequence_length, dtype=bool)
        attention[:valid_length] = True
        target_mask = np.zeros(self.sequence_length, dtype=bool)
        global_positions = np.arange(start, stop)
        target_mask[:valid_length] = (global_positions >= target_start) & (global_positions < target_stop)
        if self.augment:
            rng = augmentation_rng(self.seed, int(self._augmentation_epoch.item()), index)
            frames[:valid_length] = augment_event_sequence(frames[:valid_length], rng)
            descriptors[:valid_length] = frame_descriptors(frames[:valid_length], timestamps[:valid_length])
        return {
            "frames": self.torch.from_numpy(frames.copy()),
            "timestamps": self.torch.from_numpy(timestamps.copy()),
            "descriptors": self.torch.from_numpy(descriptors.copy()),
            "attention_mask": self.torch.from_numpy(attention),
            "labels": self.torch.from_numpy(labels.copy()).long(),
            "loss_mask": self.torch.from_numpy((labels != -100) & attention & target_mask),
            "sample_id": key,
            "subject": int(group["subject"].iloc[0]),
            "recording_key": key,
            "frame_indices": self.torch.from_numpy(array_indices.copy()).long(),
        }


def augment_event_sequence(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Event-safe augmentations; never flip left/right labels or swap polarity."""
    output = np.asarray(frames, dtype=np.float32).copy()
    if rng.random() < 0.5:
        height, width = output.shape[-2:]
        shift_y = int(rng.integers(-max(1, height // 30), max(1, height // 30) + 1))
        shift_x = int(rng.integers(-max(1, width // 30), max(1, width // 30) + 1))
        output = np.roll(output, shift=(shift_y, shift_x), axis=(-2, -1))
        if shift_y > 0:
            output[..., :shift_y, :] = 0
        elif shift_y < 0:
            output[..., shift_y:, :] = 0
        if shift_x > 0:
            output[..., :, :shift_x] = 0
        elif shift_x < 0:
            output[..., :, shift_x:] = 0
    if rng.random() < 0.4:
        scale = float(rng.uniform(0.85, 1.15))
        output = np.clip(output * scale, 0.0, 1.0)
    if rng.random() < 0.3:
        height, width = output.shape[-2:]
        mask_height = max(1, int(height * rng.uniform(0.03, 0.10)))
        mask_width = max(1, int(width * rng.uniform(0.03, 0.10)))
        y0 = int(rng.integers(0, max(1, height - mask_height + 1)))
        x0 = int(rng.integers(0, max(1, width - mask_width + 1)))
        output[..., y0 : y0 + mask_height, x0 : x0 + mask_width] = 0.0
    return output


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _load_torch()
    tensor_keys = ["frames", "timestamps", "descriptors", "attention_mask", "labels", "loss_mask", "frame_indices"]
    batch: dict[str, Any] = {}
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    for key in ["sample_id", "subject", "recording_key"]:
        batch[key] = [sample[key] for sample in samples]
    return batch


def es1_split(
    manifest: pd.DataFrame,
    validation_fold: int,
    validation_policy: str = "grouped_fold",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_policy not in ES1_VALIDATION_POLICIES:
        raise ValueError(
            f"Unknown EventSleep1 validation policy {validation_policy!r}; "
            f"choose from {ES1_VALIDATION_POLICIES}"
        )

    official_train = manifest[manifest["split"] == "TRAIN"].copy()
    test = manifest[manifest["split"] == "TEST"].copy()

    if validation_policy == "matched_resnete":
        validation_subjects = set(RESNETE_VALIDATION_SUBJECTS)
        available_train_subjects = set(official_train["subject"].astype(int).unique())
        missing = validation_subjects - available_train_subjects
        if missing:
            raise RuntimeError(
                f"Matched ResNet-E validation requires subject(s) {sorted(missing)} "
                "inside the official EventSleep1 TRAIN split"
            )
    else:
        fold_path = manifest.attrs.get("subject_folds_path")
        if fold_path is not None and Path(str(fold_path)).exists():
            folds = pd.read_csv(fold_path)
            validation_subjects = set(
                folds.loc[
                    (folds["official_split"] == "TRAIN")
                    & (folds["validation_fold_within_official_train"] == validation_fold),
                    "subject",
                ].astype(int)
            )
        else:
            train_subjects = sorted(official_train["subject"].astype(int).unique())
            validation_subjects = {
                subject for index, subject in enumerate(train_subjects) if index % 5 == validation_fold
            }

    validation = official_train[official_train["subject"].astype(int).isin(validation_subjects)].copy()
    train = official_train[~official_train["subject"].astype(int).isin(validation_subjects)].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            f"Invalid EventSleep1 split for validation policy {validation_policy!r}: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}"
        )
    if set(train["subject"]) & set(validation["subject"]):
        raise RuntimeError("EventSleep1 train/validation subject leakage")
    if set(train["subject"]) & set(test["subject"]) or set(validation["subject"]) & set(test["subject"]):
        raise RuntimeError("EventSleep1 official test subject leakage")

    if validation_policy == "matched_resnete":
        observed_train = set(train["subject"].astype(int).unique())
        observed_validation = set(validation["subject"].astype(int).unique())
        observed_test = set(test["subject"].astype(int).unique())
        if observed_train != set(RESNETE_TRAIN_SUBJECTS):
            raise RuntimeError(
                "Matched EventSleepFormer training subjects do not equal the ResNet-E training subjects: "
                f"observed={sorted(observed_train)}, expected={list(RESNETE_TRAIN_SUBJECTS)}"
            )
        if observed_validation != set(RESNETE_VALIDATION_SUBJECTS):
            raise RuntimeError(
                "Matched EventSleepFormer validation subjects do not equal the ResNet-E validation subjects: "
                f"observed={sorted(observed_validation)}, expected={list(RESNETE_VALIDATION_SUBJECTS)}"
            )
        if observed_test != set(RESNETE_TEST_SUBJECTS):
            raise RuntimeError(
                "Matched EventSleepFormer test subjects do not equal the ResNet-E test subjects: "
                f"observed={sorted(observed_test)}, expected={list(RESNETE_TEST_SUBJECTS)}"
            )
    return train, validation, test


def es2_loso_recordings(
    manifest: pd.DataFrame,
    held_out_subject: int,
) -> tuple[set[str], set[str], set[str]]:
    test = set(manifest.loc[manifest["subject"].astype(int) == held_out_subject, "recording_key"].astype(str))
    source = manifest[manifest["subject"].astype(int) != held_out_subject].copy()
    counts = source.groupby("subject")["recording_key"].nunique().sort_values(ascending=False)
    if counts.empty:
        raise RuntimeError("No source subjects remain for LOSO")
    validation_subject = int(counts.index[0])
    candidates = sorted(source.loc[source["subject"].astype(int) == validation_subject, "recording_key"].astype(str).unique())
    validation = {candidates[-1]}
    train = set(source["recording_key"].astype(str).unique()) - validation
    if not train or not validation or not test:
        raise RuntimeError(f"Invalid LOSO partition for held-out subject {held_out_subject}")
    if train & validation or train & test or validation & test:
        raise RuntimeError("EventSleep2 recording leakage")
    return train, validation, test


def class_counts_from_dataset(dataset: Any, num_classes: int, clip_level: bool) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    if clip_level:
        labels = pd.to_numeric(dataset.frame["label"], errors="raise").to_numpy(dtype=np.int64)
        counts += np.bincount(labels, minlength=num_classes)
    else:
        for group in dataset.groups.values():
            if dataset.task == "movement":
                mask = bool_series(group["supervised_movement"])
                labels = pd.to_numeric(group.loc[mask, "movement_label"], errors="raise").to_numpy(dtype=np.int64)
            else:
                mask = bool_series(group["supervised_posture"])
                labels = pd.to_numeric(group.loc[mask, "posture_label"], errors="raise").to_numpy(dtype=np.int64) - 6
            counts += np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        logging.warning("One or more training classes have zero samples: %s", counts.tolist())
    return counts


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # data-only preflight remains usable without torch
    torch = None  # type: ignore

    class _NNPlaceholder:
        Module = object

    nn = _NNPlaceholder()  # type: ignore
    F = None  # type: ignore


@dataclass
class ModelConfig:
    method: str
    num_classes: int
    clip_level: bool
    descriptor_dim: int = 5
    d_model: int = 256
    temporal_layers: int = 3
    temporal_heads: int = 4
    dropout: float = 0.1
    sequence_length: int = 32
    image_size: int = 224
    dino_model: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    backbone_mode: str = "auto"
    lora_rank: int = 8
    lora_blocks: int = 4
    spatial_microbatch: int = 64
    causal: bool = True
    variant: str = "full"


class PublishedResNetE(nn.Module):
    """Architecture used by the published EventSleep ResNet-E baseline.

    The ImageNet ResNet18 keeps its original 1000-way fully connected layer.
    The authors replace the first convolution with a randomly initialized
    two-channel convolution and append a Xavier-initialized 1000-to-10 layer.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 10) -> None:
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as exc:
            raise RuntimeError("torchvision is required for published ResNet-E") from exc
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(
            2,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.pretrained = backbone
        self.fc = nn.Linear(1000, int(num_classes))
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pretrained(frames))


class EventAdapter(nn.Module):
    """Stable 2-polarity to RGB-like mapping plus a learnable residual."""

    def __init__(self) -> None:
        super().__init__()
        self.residual = nn.Conv2d(2, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        negative, positive = frames[:, 0:1], frames[:, 1:2]
        base = torch.cat([negative, positive, 0.5 * (negative + positive)], dim=1)
        rgb = torch.clamp(base + self.residual(frames), 0.0, 1.0)
        return (rgb - self.image_mean) / self.image_std


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.scale = float(alpha if alpha is not None else rank) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b)
        return self.base(value) + self.scale * update


def inject_vit_lora(backbone: nn.Module, rank: int, last_blocks: int) -> int:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("LoRA requested, but the selected backbone has no ViT blocks")
    selected = list(blocks)[-int(last_blocks) :]
    replacements = 0
    for block in selected:
        attention = getattr(block, "attn", None)
        if attention is None:
            continue
        for name in ["qkv", "proj"]:
            layer = getattr(attention, name, None)
            if isinstance(layer, nn.Linear):
                setattr(attention, name, LoRALinear(layer, rank=rank, dropout=0.05))
                replacements += 1
    if replacements == 0:
        raise ValueError("No compatible ViT attention projections were found for LoRA")
    return replacements


class TinyEventCNN(nn.Module):
    def __init__(self, output_dim: int = 192) -> None:
        super().__init__()
        channels = [2, 32, 64, 128, output_dim]
        blocks = []
        for left, right in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(left, right, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(8, right), right),
                    nn.GELU(),
                ]
            )
        self.network = nn.Sequential(*blocks)
        self.output_dim = output_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.network(frames)
        return features.mean(dim=(-2, -1))


class SpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.method = config.method
        self.microbatch = int(config.spatial_microbatch)
        self.image_size = int(config.image_size)
        self.adapter: nn.Module
        self.backbone: nn.Module
        self.pretrained = bool(config.pretrained)
        if config.dino_model == "tiny_event_cnn" or config.method == "cnn_tcn":
            self.adapter = nn.Identity()
            self.backbone = TinyEventCNN(output_dim=192)
            self.output_dim = 192
            self.image_size = 0
            return

        self.adapter = EventAdapter()
        if config.method in {"resnet18", "mobilenet_v3"}:
            try:
                import torchvision.models as tvm
            except ImportError as exc:
                raise RuntimeError("torchvision is required for CNN baselines") from exc
            if config.method == "resnet18":
                weights = tvm.ResNet18_Weights.DEFAULT if config.pretrained else None
                model = tvm.resnet18(weights=weights)
                self.output_dim = int(model.fc.in_features)
                model.fc = nn.Identity()
            else:
                weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
                model = tvm.mobilenet_v3_small(weights=weights)
                self.output_dim = int(model.classifier[0].in_features)
                model.classifier = nn.Identity()
            self.backbone = model
            return

        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("timm is required for ViT/DINOv2 methods") from exc
        if config.method == "vit_scratch":
            name = "vit_tiny_patch16_224"
            self.backbone = timm.create_model(
                name,
                pretrained=False,
                num_classes=0,
                global_pool="token",
                img_size=config.image_size,
            )
            self.output_dim = int(self.backbone.num_features)
            return

        self.backbone = timm.create_model(
            config.dino_model,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="token",
            img_size=config.image_size,
        )
        self.output_dim = int(self.backbone.num_features)
        mode = config.backbone_mode
        if mode == "auto":
            mode = "lora" if config.method == "eventsleepformer" else "frozen"
        if config.variant == "frozen_no_lora":
            mode = "frozen"
        if mode not in {"frozen", "lora", "last_block", "full"}:
            raise ValueError(f"Unsupported backbone mode: {mode}")
        if mode in {"frozen", "lora", "last_block"}:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        if mode == "lora":
            count = inject_vit_lora(self.backbone, config.lora_rank, config.lora_blocks)
            logging.info("Inserted LoRA into %d DINOv2 attention projections", count)
        elif mode == "last_block":
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise ValueError("last_block mode requires a ViT backbone")
            for parameter in blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        if self.image_size:
            frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            frames = self.adapter(frames)
        return self.backbone(frames)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if len(frames) <= self.microbatch:
            return self._encode(frames)
        outputs = []
        for start in range(0, len(frames), self.microbatch):
            outputs.append(self._encode(frames[start : start + self.microbatch]))
        return torch.cat(outputs, dim=0)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.depthwise = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels * 2, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        output = self.depthwise(value)
        if self.padding:
            output = output[..., : -self.padding]
        left, gate = self.pointwise(output).chunk(2, dim=1)
        output = left * torch.sigmoid(gate)
        return self.norm(residual + self.dropout(output))


class TemporalTCN(nn.Module):
    def __init__(self, input_dim: int, d_model: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TemporalConvBlock(d_model, 2**index, dropout) for index in range(layers)])

    def forward(self, value: torch.Tensor, attention_mask: torch.Tensor, **_: Any) -> torch.Tensor:
        output = self.project(value).transpose(1, 2)
        for block in self.blocks:
            output = block(output)
        output = output.transpose(1, 2)
        return output * attention_mask.unsqueeze(-1)


class StandardTemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, config.sequence_length, config.d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.temporal_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.temporal_layers, enable_nested_tensor=False)
        self.causal = bool(config.causal)

    def forward(
        self,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        length = value.shape[1]
        output = self.project(value) + self.position[:, :length]
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=value.device), diagonal=1
            )
        output = self.encoder(output, mask=causal_mask, src_key_padding_mask=~attention_mask)
        return output * attention_mask.unsqueeze(-1)


class EventAwareAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, use_time: bool, use_density: bool, causal: bool) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by temporal_heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.time_decay_raw = nn.Parameter(torch.zeros(heads))
        self.density_gain = nn.Parameter(torch.zeros(heads))
        self.use_time = use_time
        self.use_density = use_density
        self.causal = causal

    def forward(
        self,
        value: torch.Tensor,
        timestamps: torch.Tensor,
        density_score: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dimension = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, content = qkv.unbind(0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if self.use_time:
            # Keep timestamps in int64 until after subtraction. Casting raw
            # microsecond timestamps to FP16 first overflows above 65,504 and
            # turns finite gaps into NaNs under CUDA autocast. Subtracting a
            # per-sequence origin also makes this safe for real AEDAT clocks,
            # whose absolute timestamps can be around 1e15 microseconds.
            relative_us = timestamps.to(dtype=torch.int64) - timestamps[:, :1].to(dtype=torch.int64)
            pairwise_us = torch.abs(relative_us[:, :, None] - relative_us[:, None, :])
            pairwise_seconds = (pairwise_us.to(dtype=torch.float32) / 1_000_000.0).clamp(
                max=MAX_TIME_BIAS_GAP_SECONDS
            )
            time_distance = torch.log1p(pairwise_seconds).to(dtype=scores.dtype)
            decay = F.softplus(self.time_decay_raw).view(1, self.heads, 1, 1)
            scores = scores - decay * time_distance[:, None]
        if self.use_density:
            gain = self.density_gain.view(1, self.heads, 1, 1)
            scores = scores + gain * density_score[:, None, None, :]
        scores = scores.masked_fill(~attention_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if self.causal:
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=value.device), diagonal=1)
            scores = scores.masked_fill(causal[None, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, content).transpose(1, 2).reshape(batch, length, dimension)
        output = self.projection(output)
        return output * attention_mask.unsqueeze(-1)


class EventAwareBlock(nn.Module):
    def __init__(self, config: ModelConfig, use_time: bool, use_density: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = EventAwareAttention(
            config.d_model,
            config.temporal_heads,
            config.dropout,
            use_time,
            use_density,
            config.causal,
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, value: torch.Tensor, timestamps: torch.Tensor, density_score: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), timestamps, density_score, attention_mask)
        value = value + self.feedforward(self.norm2(value))
        return value * attention_mask.unsqueeze(-1)


class EventAwareTemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.project = nn.Linear(input_dim, config.d_model)
        self.density_projection = nn.Sequential(
            nn.LayerNorm(config.descriptor_dim),
            nn.Linear(config.descriptor_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.density_score = nn.Sequential(
            nn.LayerNorm(config.descriptor_dim),
            nn.Linear(config.descriptor_dim, 1),
            nn.Tanh(),
        )
        use_time = config.variant not in {"no_time"}
        use_density = config.variant not in {"no_density"}
        self.use_density = use_density
        self.blocks = nn.ModuleList(
            [EventAwareBlock(config, use_time=use_time, use_density=use_density) for _ in range(config.temporal_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        value: torch.Tensor,
        timestamps: torch.Tensor,
        descriptors: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.project(value)
        if self.use_density:
            output = output + self.density_projection(descriptors)
            score = self.density_score(descriptors).squeeze(-1)
        else:
            score = torch.zeros_like(timestamps, dtype=output.dtype)
        for block in self.blocks:
            output = block(output, timestamps, score, attention_mask)
        return self.norm(output) * attention_mask.unsqueeze(-1)


class MaskedAttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("btd,d->bt", value, self.query) / math.sqrt(value.shape[-1])
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("bt,btd->bd", weights, value)


class SleepActivityModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial = SpatialEncoder(config)
        spatial_dim = self.spatial.output_dim
        self.temporal_kind = "none"
        if config.method == "cnn_tcn":
            self.temporal = TemporalTCN(spatial_dim, config.d_model, config.temporal_layers, config.dropout)
            feature_dim = config.d_model
            self.temporal_kind = "tcn"
        elif config.method == "dinov2_temporal" or config.variant == "standard_attention":
            self.temporal = StandardTemporalTransformer(spatial_dim, config)
            feature_dim = config.d_model
            self.temporal_kind = "standard"
        elif config.method == "eventsleepformer" and config.variant != "no_temporal":
            self.temporal = EventAwareTemporalTransformer(spatial_dim, config)
            feature_dim = config.d_model
            self.temporal_kind = "event_aware"
        else:
            self.temporal = nn.Identity()
            feature_dim = spatial_dim
        self.classifier = nn.Linear(feature_dim, config.num_classes)
        self.pool = MaskedAttentionPool(feature_dim)
        self.boundary_head = nn.Linear(feature_dim * 2, 1) if config.method == "eventsleepformer" and config.variant != "no_boundary" else None

    def forward(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        descriptors: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        batch, length, channels, height, width = frames.shape
        spatial = self.spatial(frames.reshape(batch * length, channels, height, width)).reshape(batch, length, -1)
        if self.temporal_kind == "event_aware":
            features = self.temporal(spatial, timestamps=timestamps, descriptors=descriptors, attention_mask=attention_mask)
        elif self.temporal_kind in {"standard", "tcn"}:
            features = self.temporal(spatial, attention_mask=attention_mask)
        else:
            features = spatial * attention_mask.unsqueeze(-1)
        token_logits = self.classifier(features)
        if self.config.clip_level:
            if self.temporal_kind == "none":
                valid = attention_mask.unsqueeze(-1)
                clip_logits = (token_logits * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            else:
                clip_logits = self.classifier(self.pool(features, attention_mask))
            logits = clip_logits
        else:
            logits = token_logits
        if self.boundary_head is not None and length > 1:
            pair = torch.cat([features[:, :-1], features[:, 1:]], dim=-1)
            boundary_logits = self.boundary_head(pair).squeeze(-1)
        else:
            boundary_logits = None
        return {"logits": logits, "token_logits": token_logits, "boundary_logits": boundary_logits, "features": features}


def model_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    loss_name: str,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    safe_counts = counts.to(device=logits.device, dtype=logits.dtype).clamp_min(1.0)
    if loss_name == "ce":
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if loss_name == "weighted_ce":
        weights = safe_counts.sum() / (len(safe_counts) * safe_counts)
        weights = weights / weights.mean()
        return F.cross_entropy(logits, labels, weight=weights, label_smoothing=label_smoothing)
    if loss_name == "balanced_softmax":
        adjusted = logits + torch.log(safe_counts).unsqueeze(0)
        return F.cross_entropy(adjusted, labels, label_smoothing=label_smoothing)
    if loss_name == "focal":
        losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
        probability = torch.exp(-losses)
        return (((1.0 - probability) ** 2.0) * losses).mean()
    raise ValueError(f"Unsupported loss: {loss_name}")


def total_model_loss(
    output: Mapping[str, torch.Tensor | None],
    batch: Mapping[str, Any],
    counts: torch.Tensor,
    loss_name: str,
    clip_level: bool,
    boundary_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    labels = batch["labels"]
    loss_mask = batch["loss_mask"].bool()
    if clip_level:
        class_loss = classification_loss(logits[loss_mask], labels[loss_mask], counts, loss_name, label_smoothing)
    else:
        class_loss = classification_loss(logits[loss_mask], labels[loss_mask], counts, loss_name, label_smoothing)
    boundary_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    boundary_logits = output.get("boundary_logits")
    if (
        not clip_level
        and boundary_weight > 0
        and isinstance(boundary_logits, torch.Tensor)
        and boundary_logits.numel()
    ):
        valid = loss_mask[:, :-1] & loss_mask[:, 1:]
        if "frame_indices" in batch:
            valid = valid & ((batch["frame_indices"][:, 1:] - batch["frame_indices"][:, :-1]) == 1)
        if "timestamps" in batch:
            gaps = batch["timestamps"][:, 1:] - batch["timestamps"][:, :-1]
            valid = valid & (gaps >= 0) & (gaps <= MAX_LOCAL_GAP_US)
        if valid.any():
            target = (labels[:, :-1] != labels[:, 1:]).to(dtype=boundary_logits.dtype)
            positives = target[valid].sum()
            negatives = valid.sum().to(dtype=boundary_logits.dtype) - positives
            positive_weight = torch.clamp(negatives / positives.clamp_min(1.0), min=1.0, max=10.0)
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits[valid], target[valid], pos_weight=positive_weight
            )
    total = class_loss + float(boundary_weight) * boundary_loss
    return total, {
        "classification_loss": float(class_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
    }


def numpy_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        probabilities = numpy_softmax(logits, temperature)
        chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.log(chosen).mean())

    result = minimize_scalar(objective, bounds=(math.log(0.25), math.log(8.0)), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.0


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    order = np.argsort(confidence)
    groups = np.array_split(order, min(bins, len(order)))
    error = 0.0
    for indices in groups:
        if len(indices) == 0:
            continue
        error += (len(indices) / len(labels)) * abs(float(correct[indices].mean()) - float(confidence[indices].mean()))
    return float(error)


def classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    temperature: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = numpy_softmax(logits, temperature)
    prediction = probabilities.argmax(axis=1)
    class_ids = np.arange(len(class_names), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=class_ids, zero_division=0
    )
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(len(class_names), dtype=np.float64)[labels]
    metrics = {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "worst_class_f1": float(f1.min()),
        "kappa": float(cohen_kappa_score(labels, prediction, labels=class_ids)),
        "nll": float(-np.log(chosen).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probabilities, labels),
        "temperature": float(temperature),
    }
    per_class = pd.DataFrame(
        {
            "class_id": class_ids,
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    confusion = confusion_matrix(labels, prediction, labels=class_ids)
    return metrics, per_class, confusion


def _segments(values: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            segments.append((int(values[start]), start, index))
            start = index
    segments.append((int(values[start]), start, len(values)))
    return segments


def _levenshtein(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def segmental_scores(predictions: pd.DataFrame, thresholds: Sequence[float] = (0.10, 0.25, 0.50)) -> dict[str, float]:
    totals = {threshold: [0, 0, 0] for threshold in thresholds}  # tp, fp, fn
    edit_scores: list[float] = []
    boundary_tp = boundary_fp = boundary_fn = 0
    for _, recording in predictions.groupby("recording_key", sort=False):
        recording = recording.sort_values("frame_index", kind="stable").reset_index(drop=True)
        frame_indices = recording["frame_index"].to_numpy(dtype=np.int64)
        timestamps = recording["timestamp_us"].to_numpy(dtype=np.int64)
        for block_start, block_stop in contiguous_blocks(
            frame_indices,
            timestamps,
            max_gap_us=MAX_LOCAL_GAP_US,
        ):
            block = np.arange(block_start, block_stop)
            group = recording.iloc[block]
            truth = group["y_true"].to_numpy(dtype=np.int64)
            pred = group["y_pred"].to_numpy(dtype=np.int64)
            true_segments = _segments(truth)
            pred_segments = _segments(pred)
            true_labels = [segment[0] for segment in true_segments]
            pred_labels = [segment[0] for segment in pred_segments]
            denominator = max(len(true_labels), len(pred_labels), 1)
            edit_scores.append(100.0 * (1.0 - _levenshtein(true_labels, pred_labels) / denominator))
            for threshold in thresholds:
                matched: set[int] = set()
                tp = fp = 0
                for predicted_label, predicted_start, predicted_stop in pred_segments:
                    best_iou = 0.0
                    best_index = -1
                    for index, (true_label, true_start, true_stop) in enumerate(true_segments):
                        if index in matched or true_label != predicted_label:
                            continue
                        intersection = max(0, min(predicted_stop, true_stop) - max(predicted_start, true_start))
                        union = max(predicted_stop, true_stop) - min(predicted_start, true_start)
                        iou = intersection / max(union, 1)
                        if iou > best_iou:
                            best_iou, best_index = iou, index
                    if best_iou >= threshold and best_index >= 0:
                        tp += 1
                        matched.add(best_index)
                    else:
                        fp += 1
                fn = len(true_segments) - len(matched)
                totals[threshold][0] += tp
                totals[threshold][1] += fp
                totals[threshold][2] += fn
            true_boundaries = np.flatnonzero(truth[1:] != truth[:-1]) + 1
            pred_boundaries = np.flatnonzero(pred[1:] != pred[:-1]) + 1
            matched_boundaries: set[int] = set()
            for boundary in pred_boundaries:
                candidates = [
                    (abs(int(boundary) - int(target)), index)
                    for index, target in enumerate(true_boundaries)
                    if index not in matched_boundaries and abs(int(boundary) - int(target)) <= 1
                ]
                if candidates:
                    _, best = min(candidates)
                    matched_boundaries.add(best)
                    boundary_tp += 1
                else:
                    boundary_fp += 1
            boundary_fn += len(true_boundaries) - len(matched_boundaries)
    result = {"edit_score": float(np.mean(edit_scores)) if edit_scores else float("nan")}
    for threshold, (tp, fp, fn) in totals.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        result[f"segment_f1_at_{int(threshold * 100)}"] = 2 * precision * recall / max(precision + recall, 1e-12)
    precision = boundary_tp / max(boundary_tp + boundary_fp, 1)
    recall = boundary_tp / max(boundary_tp + boundary_fn, 1)
    result["boundary_f1_tolerance_1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    return result


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 50
    patience: int = 10
    batch_size: int = 4
    workers: int = 4
    learning_rate: float = 3e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    accumulation_steps: int = 1
    gradient_clip: float = 1.0
    amp: bool = True
    label_smoothing: float = 0.05
    boundary_weight: float = 0.2
    deterministic: bool = True


def loader_for(dataset: Any, batch_size: int, workers: int, shuffle: bool, seed: int) -> Any:
    torch = _load_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_samples,
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(**kwargs)


def move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def optimizer_for(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    backbone_parameters: list[nn.Parameter] = []
    other_parameters: list[nn.Parameter] = []
    model_config = getattr(model, "config", None)
    lower_pretrained_backbone = bool(
        model_config is not None
        and (
            model_config.method in {"resnet18", "mobilenet_v3"}
            or model_config.backbone_mode in {"last_block", "full"}
        )
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_base_backbone = name.startswith("spatial.backbone.") and "lora_" not in name
        if lower_pretrained_backbone and is_base_backbone:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = [{"params": other_parameters, "lr": config.learning_rate}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.learning_rate * config.backbone_lr_multiplier})
    return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)


def scheduler_for(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, config.epochs * steps_per_epoch)
    warmup = min(total - 1, max(0, config.warmup_epochs * steps_per_epoch))

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def autocast_context(device: str, enabled: bool) -> Any:
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: str,
    counts: torch.Tensor,
    loss_name: str,
    clip_level: bool,
    train_config: TrainConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_class = total_boundary = 0.0
    updates = batches = pending = 0

    def optimizer_update() -> None:
        nonlocal updates, pending
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        updates += 1
        pending = 0

    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device)
        if int(batch["loss_mask"].sum().item()) == 0:
            continue
        with autocast_context(device, train_config.amp):
            output = model(
                batch["frames"],
                batch["timestamps"],
                batch["descriptors"],
                batch["attention_mask"],
            )
            loss, parts = total_model_loss(
                output,
                batch,
                counts,
                loss_name,
                clip_level,
                train_config.boundary_weight,
                train_config.label_smoothing,
            )
            scaled_loss = loss / train_config.accumulation_steps
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: {float(loss.detach().cpu())}; "
                f"class counts={counts.detach().cpu().tolist()}"
            )
        scaler.scale(scaled_loss).backward()
        pending += 1
        if pending >= train_config.accumulation_steps:
            optimizer_update()
        total_loss += float(loss.detach().cpu())
        total_class += parts["classification_loss"]
        total_boundary += parts["boundary_loss"]
        batches += 1
    if pending:
        optimizer_update()
    if batches == 0:
        raise RuntimeError("No supervised training batches were available")
    return {
        "loss": total_loss / batches,
        "classification_loss": total_class / batches,
        "boundary_loss": total_boundary / batches,
        "optimizer_updates": updates,
    }


def collect_predictions(
    model: nn.Module,
    loader: Any,
    device: str,
    clip_level: bool,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with autocast_context(device, amp):
                output = model(
                    batch["frames"],
                    batch["timestamps"],
                    batch["descriptors"],
                    batch["attention_mask"],
                )
            logits = output["logits"]
            assert isinstance(logits, torch.Tensor)
            if clip_level:
                values = logits.float().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_logits.append(values)
                all_labels.append(labels)
                for item in range(len(labels)):
                    rows.append(
                        {
                            "sample_id": raw_batch["sample_id"][item],
                            "subject": int(raw_batch["subject"][item]),
                            "recording_key": raw_batch["recording_key"][item],
                            "frame_index": -1,
                            "timestamp_us": int(raw_batch["timestamps"][item, -1]),
                            "y_true": int(labels[item]),
                        }
                    )
            else:
                values = logits.float().cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                mask = batch["loss_mask"].cpu().numpy().astype(bool)
                frame_indices = raw_batch["frame_indices"].numpy()
                timestamps = raw_batch["timestamps"].numpy()
                for item in range(values.shape[0]):
                    selected = mask[item]
                    all_logits.append(values[item, selected])
                    all_labels.append(labels[item, selected])
                    positions = np.flatnonzero(selected)
                    for position in positions:
                        rows.append(
                            {
                                "sample_id": f"{raw_batch['recording_key'][item]}_frame{int(frame_indices[item, position]):06d}",
                                "subject": int(raw_batch["subject"][item]),
                                "recording_key": raw_batch["recording_key"][item],
                                "frame_index": int(frame_indices[item, position]),
                                "timestamp_us": int(timestamps[item, position]),
                                "y_true": int(labels[item, position]),
                            }
                        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    if not all_labels:
        raise RuntimeError("No supervised predictions were collected")
    logits_array = np.concatenate(all_logits, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    metadata = pd.DataFrame(rows)
    if len(metadata) != len(labels_array):
        raise RuntimeError("Prediction metadata and logits have inconsistent lengths")
    return logits_array, labels_array, metadata, elapsed


def attach_predictions(
    metadata: pd.DataFrame,
    logits: np.ndarray,
    temperature: float,
    class_names: Sequence[str],
) -> pd.DataFrame:
    probabilities = numpy_softmax(logits, temperature)
    prediction = probabilities.argmax(axis=1)
    output = metadata.copy()
    output["y_pred"] = prediction
    output["y_true_name"] = [class_names[int(value)] for value in output["y_true"]]
    output["y_pred_name"] = [class_names[int(value)] for value in prediction]
    output["confidence"] = probabilities.max(axis=1)
    for index, name in enumerate(class_names):
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        output[f"prob_{index}_{safe_name}"] = probabilities[:, index]
        output[f"logit_{index}_{safe_name}"] = logits[:, index]
    return output


def per_unit_metrics(predictions: pd.DataFrame, class_names: Sequence[str]) -> pd.DataFrame:
    logit_columns = [column for column in predictions if column.startswith("logit_")]
    rows: list[dict[str, Any]] = []
    for (subject, recording), group in predictions.groupby(["subject", "recording_key"], sort=True):
        logits = group[logit_columns].to_numpy(dtype=np.float64)
        labels = group["y_true"].to_numpy(dtype=np.int64)
        metrics, _, _ = classification_metrics(labels, logits, class_names, temperature=1.0)
        rows.append({"subject": subject, "recording_key": recording, **metrics})
    return pd.DataFrame(rows)


def build_datasets_for_run(
    paths: DataPaths,
    dataset_name: str,
    task: str,
    held_out_subject: int | None,
    validation_fold: int,
    es1_validation_policy: str,
    sequence_length: int,
    seed: int,
    include_test: bool = True,
) -> tuple[Any, Any, Any | None, bool, list[str], dict[str, Any]]:
    if dataset_name == "eventsleep1":
        if not paths.es1_sequence_manifest.exists():
            raise FileNotFoundError(
                f"Missing temporal cache manifest {paths.es1_sequence_manifest}. Run prepare first."
            )
        manifest = pd.read_csv(paths.es1_sequence_manifest, low_memory=False)
        if "sequence_length" in manifest.columns:
            cached_lengths = sorted(pd.to_numeric(manifest["sequence_length"], errors="raise").astype(int).unique().tolist())
            if cached_lengths != [int(sequence_length)]:
                raise ValueError(
                    f"EventSleep1 temporal cache length is {cached_lengths}, but the model requested {sequence_length}. "
                    "Use the same --sequence-length or rebuild the temporal cache with --overwrite."
                )
        manifest.attrs["subject_folds_path"] = str(
            paths.preprocess_root / "eventsleep1" / "manifests" / "subject_folds.csv"
        )
        train_frame, validation_frame, test_frame = es1_split(manifest, validation_fold, es1_validation_policy)
        train = EventSleep1Dataset(paths.preprocess_root, train_frame, augment=True, seed=seed)
        validation = EventSleep1Dataset(paths.preprocess_root, validation_frame, augment=False, seed=seed)
        test = EventSleep1Dataset(paths.preprocess_root, test_frame, augment=False, seed=seed) if include_test else None
        class_names = [STAGE_NAMES[index] for index in ES1_CLASSES]
        split = {
            "protocol": "official_subject_independent_train_test",
            "validation_policy": es1_validation_policy,
            "validation_fold": validation_fold if es1_validation_policy == "grouped_fold" else None,
            "matched_resnete_split": es1_validation_policy == "matched_resnete",
            "train_subjects": sorted(train_frame["subject"].astype(int).unique().tolist()),
            "validation_subjects": sorted(validation_frame["subject"].astype(int).unique().tolist()),
            "test_subjects": sorted(test_frame["subject"].astype(int).unique().tolist()),
        }
        return train, validation, test, True, class_names, split

    if dataset_name != "eventsleep2":
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if held_out_subject is None:
        raise ValueError("EventSleep2 requires a held-out subject")
    manifest = pd.read_csv(paths.es2_manifest, low_memory=False)
    train_keys, validation_keys, test_keys = es2_loso_recordings(manifest, held_out_subject)
    train = EventSleep2SequenceDataset(paths.preprocess_root, manifest, task, sequence_length, True, seed, train_keys)
    validation = EventSleep2SequenceDataset(paths.preprocess_root, manifest, task, sequence_length, False, seed, validation_keys)
    test = EventSleep2SequenceDataset(paths.preprocess_root, manifest, task, sequence_length, False, seed, test_keys) if include_test else None
    original_ids = ES2_MOVEMENT_CLASSES if task == "movement" else ES2_POSTURE_CLASSES
    class_names = [STAGE_NAMES[index] for index in original_ids]
    split = {
        "protocol": "leave_one_subject_out",
        "validation_policy": "recording_holdout_within_non_test_subjects",
        "held_out_subject": held_out_subject,
        "train_recordings": sorted(train_keys),
        "validation_recordings": sorted(validation_keys),
        "test_recordings": sorted(test_keys),
    }
    return train, validation, test, False, class_names, split


def configuration_id(
    model_config: ModelConfig,
    train_config: TrainConfig,
    method_spec: str,
) -> str:
    """Stable short hash for a tunable configuration, excluding seed/fold/task-derived fields."""
    model_payload = asdict(model_config)
    # These fields are assigned from the dataset/task inside run_single_experiment
    # and must not change the identity of the hyperparameter configuration.
    for key in ("method", "num_classes", "clip_level"):
        model_payload.pop(key, None)
    payload = {
        "method_spec": method_spec,
        "model": model_payload,
        "training": asdict(train_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def run_single_experiment(
    paths: DataPaths,
    output_root: Path,
    dataset_name: str,
    task: str,
    method_spec: str,
    seed: int,
    held_out_subject: int | None,
    validation_fold: int,
    es1_validation_policy: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: str,
    overwrite: bool,
    evaluation_scope: str = "test",
    keep_checkpoint: bool = True,
) -> dict[str, Any]:
    if evaluation_scope not in EVALUATION_SCOPES:
        raise ValueError(f"Unknown evaluation scope {evaluation_scope!r}; choose from {EVALUATION_SCOPES}")
    method, loss_name = parse_method_spec(method_spec)
    if dataset_name == "eventsleep1":
        if es1_validation_policy == "grouped_fold":
            validation_tag = f"grouped_fold{validation_fold}"
        else:
            validation_tag = es1_validation_policy
        fold_name = f"official_test_val-{validation_tag}"
    else:
        fold_name = f"held_out_subject{held_out_subject:02d}"
    variant_suffix = "" if model_config.variant == "full" else f"_variant-{model_config.variant}"
    version_tag = VERSION.replace(".", "_")
    scope_tag = "test" if evaluation_scope == "test" else "validation-only"
    config_hash = configuration_id(model_config, train_config, method_spec)
    run_name = (
        f"{dataset_name}_{task}_{fold_name}_{method}_{loss_name}{variant_suffix}"
        f"_eval-{scope_tag}_cfg-{config_hash}_pipe-{version_tag}_seed-{seed}"
    )
    run_dir = output_root / "runs" / run_name
    completion = run_dir / "completed_successfully.json"
    metrics_path = run_dir / ("metrics.json" if evaluation_scope == "test" else "validation_metrics.json")
    if completion.exists() and metrics_path.exists() and not overwrite:
        logging.info("Reusing completed run: %s", run_name)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise RuntimeError(f"Incomplete run exists: {run_dir}; inspect it or use --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed, train_config.deterministic)
    include_test = evaluation_scope == "test"
    train_dataset, validation_dataset, test_dataset, clip_level, class_names, split = build_datasets_for_run(
        paths,
        dataset_name,
        task,
        held_out_subject,
        validation_fold,
        es1_validation_policy,
        model_config.sequence_length,
        seed,
        include_test=include_test,
    )
    num_classes = len(class_names)
    model_config = copy.deepcopy(model_config)
    model_config.method = method
    model_config.num_classes = num_classes
    model_config.clip_level = clip_level
    # Recompute after method/task normalization; remains seed/fold independent.
    config_hash = configuration_id(model_config, train_config, method_spec)
    split = dict(split)
    split["evaluation_scope"] = evaluation_scope
    split["test_dataset_instantiated"] = bool(include_test)
    atomic_json(run_dir / "split.json", split)
    counts_numpy = class_counts_from_dataset(train_dataset, num_classes, clip_level)
    atomic_csv(
        run_dir / "training_class_counts.csv",
        pd.DataFrame({"class_id": range(num_classes), "class_name": class_names, "count": counts_numpy}),
    )
    counts = torch.tensor(counts_numpy, dtype=torch.float32, device=device)
    model = SleepActivityModel(model_config).to(device)
    parameters = model_parameter_counts(model)
    train_loader = loader_for(train_dataset, train_config.batch_size, train_config.workers, True, seed)
    validation_loader = loader_for(validation_dataset, train_config.batch_size, train_config.workers, False, seed)
    test_loader = None
    if include_test:
        if test_dataset is None:
            raise RuntimeError("Test evaluation requested but test dataset was not instantiated")
        test_loader = loader_for(test_dataset, train_config.batch_size, train_config.workers, False, seed)
    optimizer = optimizer_for(model, train_config)
    update_steps = max(1, math.ceil(len(train_loader) / train_config.accumulation_steps))
    scheduler = scheduler_for(optimizer, update_steps, train_config)
    amp_enabled = train_config.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best_model.pt"
    training_start = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, train_config.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            counts,
            loss_name,
            clip_level,
            train_config,
        )
        validation_logits, validation_labels, _, validation_seconds = collect_predictions(
            model, validation_loader, device, clip_level, train_config.amp
        )
        validation_metrics, _, _ = classification_metrics(validation_labels, validation_logits, class_names)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "validation_seconds": validation_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
        score = float(validation_metrics["macro_f1"])
        logging.info(
            "%s | epoch %d/%d | loss %.4f | val macro-F1 %.4f",
            run_name,
            epoch,
            train_config.epochs,
            train_metrics["loss"],
            score,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "pipeline_version": VERSION,
                    "configuration_id": config_hash,
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "class_names": class_names,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= train_config.patience:
            logging.info("Early stopping %s at epoch %d", run_name, epoch)
            break

    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    if evaluation_scope == "validation_only":
        eval_logits, eval_labels, metadata, inference_seconds = collect_predictions(
            model, validation_loader, device, clip_level, train_config.amp
        )
        # Do not temperature-fit and score on the same validation set.  Primary
        # tuning metrics (macro-F1/balanced accuracy) are unaffected by this.
        temperature = 1.0
        metrics, per_class, confusion = classification_metrics(
            eval_labels, eval_logits, class_names, temperature=temperature
        )
        prediction_name = "validation_predictions.csv"
        per_class_name = "validation_per_class_metrics.csv"
        confusion_name = "validation_confusion_matrix.csv"
        per_unit_name = "validation_per_unit_metrics.csv"
        evaluated_split = "validation"
    else:
        validation_logits, validation_labels, _, _ = collect_predictions(
            model, validation_loader, device, clip_level, train_config.amp
        )
        temperature = fit_temperature(validation_logits, validation_labels)
        if test_loader is None:
            raise RuntimeError("Internal error: missing test loader")
        eval_logits, eval_labels, metadata, inference_seconds = collect_predictions(
            model, test_loader, device, clip_level, train_config.amp
        )
        metrics, per_class, confusion = classification_metrics(
            eval_labels, eval_logits, class_names, temperature
        )
        prediction_name = "predictions.csv"
        per_class_name = "per_class_metrics.csv"
        confusion_name = "confusion_matrix.csv"
        per_unit_name = "per_unit_metrics.csv"
        evaluated_split = "test"

    predictions = attach_predictions(metadata, eval_logits, temperature, class_names)
    predictions.insert(0, "dataset", dataset_name)
    predictions.insert(1, "task", task)
    predictions.insert(2, "method", method)
    predictions.insert(3, "loss", loss_name)
    predictions.insert(4, "variant", model_config.variant)
    predictions.insert(5, "seed", seed)
    predictions.insert(6, "fold", fold_name)
    predictions.insert(7, "evaluated_split", evaluated_split)
    if not clip_level:
        metrics.update(segmental_scores(predictions))
    metrics.update(
        {
            "pipeline_version": VERSION,
            "configuration_id": config_hash,
            "dataset": dataset_name,
            "task": task,
            "method": method,
            "loss": loss_name,
            "variant": model_config.variant,
            "seed": seed,
            "fold": fold_name,
            "evaluation_scope": evaluation_scope,
            "evaluated_split": evaluated_split,
            "official_test_evaluated": bool(evaluation_scope == "test"),
            "validation_policy": str(split.get("validation_policy", "unknown")),
            "augmentation_policy": AUGMENTATION_POLICY,
            "determinism_policy": DETERMINISM_POLICY if train_config.deterministic else "disabled",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score,
            "temperature": float(temperature),
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "inference_ms_per_prediction": 1000.0 * inference_seconds / len(eval_labels),
            **parameters,
        }
    )
    if device.startswith("cuda"):
        metrics["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
    per_class.insert(0, "dataset", dataset_name)
    per_class.insert(1, "task", task)
    per_class.insert(2, "method", method)
    per_class.insert(3, "loss", loss_name)
    per_class.insert(4, "variant", model_config.variant)
    per_class.insert(5, "seed", seed)
    per_class.insert(6, "fold", fold_name)
    per_class.insert(7, "evaluated_split", evaluated_split)
    atomic_json(metrics_path, metrics)
    atomic_csv(run_dir / per_class_name, per_class)
    atomic_csv(run_dir / prediction_name, predictions)
    atomic_csv(run_dir / per_unit_name, per_unit_metrics(predictions, class_names))
    atomic_csv(
        run_dir / confusion_name,
        pd.DataFrame(confusion, index=class_names, columns=class_names).reset_index(names="true_class"),
    )
    atomic_json(
        run_dir / "run_configuration.json",
        {
            "created_utc": utc_now(),
            "pipeline_version": VERSION,
            "configuration_id": config_hash,
            "evaluation_scope": evaluation_scope,
            "official_test_evaluated": bool(evaluation_scope == "test"),
            "data": asdict(paths),
            "model": asdict(model_config),
            "training": asdict(train_config),
            "method_spec": method_spec,
            "split": split,
            "augmentation_policy": AUGMENTATION_POLICY,
            "determinism_policy": DETERMINISM_POLICY if train_config.deterministic else "disabled",
            "device": device,
        },
    )
    if not keep_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    atomic_json(
        completion,
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "configuration_id": config_hash,
            "evaluation_scope": evaluation_scope,
            "status": "success",
        },
    )
    logging.info(
        "%s completed: %s macro-F1 %.4f", run_name, evaluated_split, metrics["macro_f1"]
    )
    return metrics


# ---------------------------------------------------------------------------
# Published ResNet-E training/evaluation reproduction (EventSleep1 only)
# ---------------------------------------------------------------------------


def published_resnete_configuration(
    profile: str | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    frequencies = np.asarray(RESNETE_FREQUENCIES, dtype=np.float64)
    configuration: dict[str, Any] = {
        "architecture": "ImageNet ResNet18; conv1 replaced by random 2-channel convolution; original 1000-way fc retained; appended Linear(1000,10)",
        "pretrained": bool(pretrained),
        "input_channels": 2,
        "input_height": RESNETE_OUTPUT_HEIGHT,
        "input_width": RESNETE_OUTPUT_WIDTH,
        "event_window_ms": RESNETE_CHUNK_US / 1000,
        "time_surface_history_ms": RESNETE_HISTORY_US / 1000,
        "events_per_pixel_polarity": RESNETE_K,
        "spatial_stride": RESNETE_STEP_PIXELS,
        "batch_size": RESNETE_BATCH_SIZE,
        "epochs": RESNETE_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": RESNETE_LEARNING_RATE,
        "weight_decay": RESNETE_WEIGHT_DECAY,
        "scheduler": None,
        "warmup": None,
        "mixed_precision": False,
        "label_smoothing": 0.0,
        "loss": "weighted_cross_entropy",
        "class_frequencies_from_authors_code": frequencies.tolist(),
        "class_weights_max_frequency_over_frequency": (frequencies.max() / frequencies).tolist(),
        "train_subjects": list(RESNETE_TRAIN_SUBJECTS),
        "validation_subjects": list(RESNETE_VALIDATION_SUBJECTS),
        "test_subjects": list(RESNETE_TEST_SUBJECTS),
        "configurations": list(RESNETE_CONFIGURATIONS),
        "checkpoint_metric": "frame_level_balanced_accuracy_on_subject11",
        "primary_test_metric": "clip_level_balanced_accuracy",
        "paper_reported_clips": 1016,
        "paper_reported_resnete_balanced_accuracy": 0.82,
        "source_urls": RESNETE_SOURCE_URLS,
    }
    if profile is not None:
        configuration["profile"] = resnete_profile_settings(profile)
    return configuration


def published_resnete_split(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    subject = manifest["subject"].astype(int)
    configuration = manifest["configuration"].astype(int)
    train = manifest[subject.isin(RESNETE_TRAIN_SUBJECTS) & configuration.isin(RESNETE_CONFIGURATIONS)].copy()
    validation = manifest[
        subject.isin(RESNETE_VALIDATION_SUBJECTS) & configuration.isin(RESNETE_CONFIGURATIONS)
    ].copy()
    test = manifest[subject.isin(RESNETE_TEST_SUBJECTS) & configuration.isin(RESNETE_CONFIGURATIONS)].copy()
    selected = set(train["sample_id"].astype(str)) | set(validation["sample_id"].astype(str)) | set(test["sample_id"].astype(str))
    if len(selected) != len(manifest):
        missing = sorted(set(manifest["sample_id"].astype(str)) - selected)
        raise RuntimeError(f"ResNet-E split did not assign all clips: {missing[:10]}")
    if (
        set(train["sample_id"]) & set(validation["sample_id"])
        or set(train["sample_id"]) & set(test["sample_id"])
        or set(validation["sample_id"]) & set(test["sample_id"])
    ):
        raise RuntimeError("ResNet-E split contains clip leakage")
    split = {
        "protocol": "published_resnete_subject_split",
        "train_subjects": sorted(train["subject"].astype(int).unique().tolist()),
        "validation_subjects": sorted(validation["subject"].astype(int).unique().tolist()),
        "test_subjects": sorted(test["subject"].astype(int).unique().tolist()),
        "configurations": sorted(manifest["configuration"].astype(int).unique().tolist()),
        "train_clips": int(len(train)),
        "validation_clips": int(len(validation)),
        "test_clips": int(len(test)),
        "current_release_total_clips": int(len(manifest)),
        "paper_reported_total_clips": 1016,
        "exact_paper_data_cardinality": bool(len(manifest) == 1016),
    }
    if split["train_subjects"] != list(RESNETE_TRAIN_SUBJECTS):
        raise RuntimeError(f"Incorrect ResNet-E training subjects: {split['train_subjects']}")
    if split["validation_subjects"] != list(RESNETE_VALIDATION_SUBJECTS):
        raise RuntimeError(f"Incorrect ResNet-E validation subjects: {split['validation_subjects']}")
    if split["test_subjects"] != list(RESNETE_TEST_SUBJECTS):
        raise RuntimeError(f"Incorrect ResNet-E test subjects: {split['test_subjects']}")
    return train, validation, test, split


def published_resnete_loader(
    dataset: PublishedResNetEFrameDataset,
    workers: int,
    shuffle: bool,
    seed: int,
) -> Any:
    torch_module = _load_torch()
    generator = torch_module.Generator()
    generator.manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": RESNETE_BATCH_SIZE,
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": bool(torch_module.cuda.is_available()),
        "generator": generator,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return torch_module.utils.data.DataLoader(**kwargs)


def train_published_resnete_epoch(
    model: PublishedResNetE,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    samples = 0
    batches = 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite published ResNet-E loss: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        batch_samples = int(len(labels))
        total_loss += float(loss.detach().cpu()) * batch_samples
        samples += batch_samples
        batches += 1
    if samples == 0:
        raise RuntimeError("Published ResNet-E training loader produced no samples")
    return {"loss": total_loss / samples, "samples": samples, "batches": batches}


def collect_published_resnete_predictions(
    model: PublishedResNetE,
    loader: Any,
    device: str,
    validation_train_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    if validation_train_mode:
        # Diagnostic only: the released validation loop does not call eval(),
        # so BatchNorm observes validation batches.  We reproduce this only
        # under authors_code_exact and label it accordingly.
        model.train()
    else:
        model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            logits = model(frames).float().cpu().numpy()
            labels = batch["labels"].cpu().numpy().astype(np.int64)
            all_logits.append(logits)
            all_labels.append(labels)
            for index in range(len(labels)):
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "subject": int(batch["subject"][index]),
                        "configuration": int(batch["configuration"][index]),
                        "recording_key": str(batch["recording_key"][index]),
                        "frame_index": int(batch["frame_index"][index]),
                        "timestamp_us": int(batch["timestamp_us"][index]),
                        "augmented": bool(batch["augmented"][index]),
                        "y_true": int(labels[index]),
                    }
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not all_logits:
        raise RuntimeError("Published ResNet-E evaluation loader produced no samples")
    logits_array = np.concatenate(all_logits, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    metadata = pd.DataFrame(rows)
    if len(metadata) != len(labels_array):
        raise RuntimeError("Published ResNet-E prediction metadata length mismatch")
    return logits_array, labels_array, metadata, elapsed


def aggregate_resnete_clips_explicit(
    frame_logits: np.ndarray,
    frame_metadata: pd.DataFrame,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    probabilities = numpy_softmax(frame_logits)
    frame_predictions = probabilities.argmax(axis=1)
    mode_scores: list[np.ndarray] = []
    probability_scores: list[np.ndarray] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []
    for sample_id, group in frame_metadata.groupby("sample_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        true_values = group["y_true"].to_numpy(dtype=np.int64)
        if len(np.unique(true_values)) != 1:
            raise RuntimeError(f"Clip {sample_id} has inconsistent frame labels")
        counts = np.bincount(frame_predictions[indices], minlength=num_classes).astype(np.float64)
        summed = probabilities[indices].sum(axis=0)
        summed /= max(float(summed.sum()), 1e-12)
        mode_scores.append(counts)
        probability_scores.append(np.log(np.clip(summed, 1e-12, 1.0)))
        labels.append(int(true_values[0]))
        first = group.iloc[0]
        rows.append(
            {
                "sample_id": str(sample_id),
                "subject": int(first["subject"]),
                "configuration": int(first["configuration"]),
                "recording_key": str(first["recording_key"]),
                "frame_index": -1,
                "timestamp_us": int(group["timestamp_us"].max()),
                "frames_in_clip": int(len(group)),
                "y_true": int(true_values[0]),
            }
        )
    return (
        np.stack(mode_scores),
        np.asarray(labels, dtype=np.int64),
        pd.DataFrame(rows),
        np.stack(probability_scores),
        np.asarray(labels, dtype=np.int64),
        pd.DataFrame(rows),
    )


def aggregate_resnete_label_runs(
    frame_logits: np.ndarray,
    frame_metadata: pd.DataFrame,
    num_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Released label-run aggregation, reported only as a diagnostic."""
    labels = frame_metadata["y_true"].to_numpy(dtype=np.int64)
    predictions = np.asarray(frame_logits).argmax(axis=1)
    boundaries = np.concatenate(
        [np.array([0]), np.flatnonzero(labels[1:] != labels[:-1]) + 1, np.array([len(labels)])]
    )
    scores: list[np.ndarray] = []
    targets: list[int] = []
    rows: list[dict[str, Any]] = []
    for run_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        counts = np.bincount(predictions[start:stop], minlength=num_classes).astype(np.float64)
        scores.append(counts)
        targets.append(int(labels[start]))
        first = frame_metadata.iloc[int(start)]
        rows.append(
            {
                "sample_id": f"released_label_run_{run_index:04d}",
                "subject": int(first["subject"]),
                "configuration": int(first["configuration"]),
                "recording_key": str(first["recording_key"]),
                "frame_index": -1,
                "timestamp_us": int(frame_metadata.iloc[int(stop) - 1]["timestamp_us"]),
                "frames_in_run": int(stop - start),
                "y_true": int(labels[start]),
            }
        )
    return np.stack(scores), np.asarray(targets, dtype=np.int64), pd.DataFrame(rows)


def check_published_resnete_model(
    paths: DataPaths,
    output_root: Path,
    profile: str,
    device: str,
    pretrained: bool,
    full_cache_check: bool,
) -> dict[str, Any]:
    cache = verify_published_resnete_cache(paths, profile, full_cache_check)
    manifest = pd.read_csv(paths.resnete_manifest(profile), low_memory=False)
    row = manifest.iloc[0]
    frames = np.load(paths.preprocess_root / str(row["resnete_frame_path"]), mmap_mode="r", allow_pickle=False)
    sample = torch.from_numpy(np.asarray(frames[: min(2, len(frames))], dtype=np.float32)).to(device)
    model = PublishedResNetE(pretrained=pretrained, num_classes=10).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(sample)
    if tuple(logits.shape) != (len(sample), 10) or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError(f"Published ResNet-E model check failed: logits {tuple(logits.shape)}")
    frequencies = torch.tensor(RESNETE_FREQUENCIES, dtype=torch.float32, device=device)
    weights = frequencies.max() / frequencies
    labels = torch.arange(len(sample), device=device) % 10
    loss = F.cross_entropy(logits, labels, weight=weights)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Published ResNet-E model check returned non-finite weighted CE")
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "device": device,
        "pretrained": bool(pretrained),
        "input_shape": list(sample.shape),
        "logit_shape": list(logits.shape),
        "weighted_ce": float(loss.detach().cpu()),
        **model_parameter_counts(model),
        "cache": cache,
        "published_configuration": published_resnete_configuration(profile, pretrained),
        "passed": True,
    }
    check_dir = output_root / "published_resnete" / profile
    atomic_json(check_dir / "model_check.json", report)
    return report


def synthetic_published_resnete_component_check(device: str) -> dict[str, Any]:
    rng = np.random.default_rng(20260831)
    count = 1_500
    timestamps = 2_000_000 + np.arange(count, dtype=np.int64) * 1_000
    events = np.column_stack(
        [
            rng.integers(95, 595, size=count, dtype=np.int64),
            rng.integers(80, 420, size=count, dtype=np.int64),
            timestamps,
            rng.integers(0, 2, size=count, dtype=np.int64),
        ]
    )
    full, anchors = _resnete_process_event_stream(events, "paper_intended")
    x0, y0, x1, y1 = _resnete_crop_for_subject(5, "paper_intended")
    cropped = full[:, y0:y1:2, x0:x1:2, :, :]
    cropped = cropped.reshape(len(cropped), RESNETE_OUTPUT_HEIGHT, RESNETE_OUTPUT_WIDTH, 2)
    cropped = cropped.transpose(0, 3, 1, 2).astype(np.float32)
    if tuple(cropped.shape[1:]) != (2, RESNETE_OUTPUT_HEIGHT, RESNETE_OUTPUT_WIDTH):
        raise RuntimeError(f"Synthetic ResNet-E crop failed: {cropped.shape}")
    model = PublishedResNetE(pretrained=False, num_classes=10).to(device)
    sample = torch.from_numpy(cropped[: min(2, len(cropped))]).to(device)
    with torch.inference_mode():
        logits = model(sample)
    frequencies = torch.tensor(RESNETE_FREQUENCIES, dtype=torch.float32, device=device)
    weights = frequencies.max() / frequencies
    labels = torch.arange(len(sample), device=device) % 10
    loss = F.cross_entropy(logits, labels, weight=weights)
    if tuple(logits.shape) != (len(sample), 10) or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Synthetic published ResNet-E component check failed")
    if RESNETE_FLIPPED_LABELS[2] != 3 or RESNETE_FLIPPED_LABELS[7] != 6:
        raise RuntimeError("Published ResNet-E flip-label map is invalid")
    return {
        "frames": int(len(cropped)),
        "anchors_monotonic": bool(np.all(np.diff(anchors) >= 0)),
        "frame_shape": list(cropped.shape),
        "logit_shape": list(logits.shape),
        "weighted_ce": float(loss.detach().cpu()),
        "passed": True,
    }


def run_published_resnete_experiment(
    paths: DataPaths,
    output_root: Path,
    profile: str,
    seed: int,
    device: str,
    workers: int,
    pretrained: bool,
    overwrite: bool,
) -> dict[str, Any]:
    run_name = f"eventsleep1_activity_official_test_resnet_e_published_weighted_ce_variant-{profile}_seed-{seed}"
    run_dir = output_root / "runs" / run_name
    completion = run_dir / "completed_successfully.json"
    if completion.exists() and not overwrite:
        logging.info("Reusing completed published ResNet-E run: %s", run_name)
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Incomplete published ResNet-E run exists: {run_dir}; inspect it or use --overwrite-runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed, deterministic=True)
    manifest = pd.read_csv(paths.resnete_manifest(profile), low_memory=False)
    train_frame, validation_frame, test_frame, split = published_resnete_split(manifest)
    settings = resnete_profile_settings(profile)
    train_dataset = PublishedResNetEFrameDataset(
        paths.preprocess_root,
        train_frame,
        profile,
        augment=bool(settings["augment_training"]),
    )
    validation_dataset = PublishedResNetEFrameDataset(
        paths.preprocess_root,
        validation_frame,
        profile,
        augment=bool(settings["augment_validation"]),
    )
    test_dataset = PublishedResNetEFrameDataset(
        paths.preprocess_root,
        test_frame,
        profile,
        augment=False,
    )
    train_loader = published_resnete_loader(train_dataset, workers, True, seed)
    validation_loader = published_resnete_loader(validation_dataset, workers, True, seed + 1)
    test_loader = published_resnete_loader(test_dataset, workers, False, seed + 2)
    frequencies = torch.tensor(RESNETE_FREQUENCIES, dtype=torch.float32, device=device)
    weights = frequencies.max() / frequencies
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = PublishedResNetE(pretrained=pretrained, num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=RESNETE_LEARNING_RATE,
        weight_decay=RESNETE_WEIGHT_DECAY,
    )
    parameters = model_parameter_counts(model)
    atomic_json(run_dir / "split.json", split)
    atomic_csv(
        run_dir / "published_class_weights.csv",
        pd.DataFrame(
            {
                "class_id": range(10),
                "class_name": [STAGE_NAMES[index] for index in range(10)],
                "published_frequency": list(RESNETE_FREQUENCIES),
                "published_weight": weights.detach().cpu().numpy(),
            }
        ),
    )
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_validation_macro_f1 = float("nan")
    checkpoint_path = run_dir / "best_model.pt"
    training_start = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, RESNETE_EPOCHS + 1):
        train_metrics = train_published_resnete_epoch(model, train_loader, optimizer, criterion, device)
        validation_logits, validation_labels, _, validation_seconds = collect_published_resnete_predictions(
            model,
            validation_loader,
            device,
            validation_train_mode=settings["validation_batchnorm_mode"] == "train_as_released",
        )
        validation_metrics, _, _ = classification_metrics(
            validation_labels,
            validation_logits,
            [STAGE_NAMES[index] for index in range(10)],
            temperature=1.0,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_frames_with_augmentation": train_metrics["samples"],
            "validation_frames_with_augmentation": int(len(validation_labels)),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "validation_seconds": validation_seconds,
            "learning_rate": RESNETE_LEARNING_RATE,
        }
        history.append(row)
        atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
        score = float(validation_metrics["balanced_accuracy"])
        logging.info(
            "%s | epoch %d/%d | loss %.4f | val balanced accuracy %.4f",
            run_name,
            epoch,
            RESNETE_EPOCHS,
            train_metrics["loss"],
            score,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation_macro_f1 = float(validation_metrics["macro_f1"])
            torch.save(
                {
                    "pipeline_version": VERSION,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "lr": RESNETE_LEARNING_RATE,
                    "batch_size": RESNETE_BATCH_SIZE,
                    "weight_decay": RESNETE_WEIGHT_DECAY,
                    "best_validation_balanced_accuracy": best_score,
                    "profile": profile,
                },
                checkpoint_path,
            )
    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    frame_logits, frame_labels, frame_metadata, inference_seconds = collect_published_resnete_predictions(
        model,
        test_loader,
        device,
        validation_train_mode=False,
    )
    class_names = [STAGE_NAMES[index] for index in range(10)]
    frame_metrics, frame_per_class, frame_confusion = classification_metrics(
        frame_labels,
        frame_logits,
        class_names,
        temperature=1.0,
    )
    frame_predictions = attach_predictions(frame_metadata, frame_logits, 1.0, class_names)
    (
        mode_logits,
        clip_labels,
        clip_metadata,
        probability_logits,
        probability_labels,
        probability_metadata,
    ) = aggregate_resnete_clips_explicit(frame_logits, frame_metadata, num_classes=10)
    metrics, per_class, confusion = classification_metrics(
        clip_labels,
        mode_logits,
        class_names,
        temperature=1.0,
    )
    probability_metrics, probability_per_class, probability_confusion = classification_metrics(
        probability_labels,
        probability_logits,
        class_names,
        temperature=1.0,
    )
    label_run_logits, label_run_labels, label_run_metadata = aggregate_resnete_label_runs(
        frame_logits,
        frame_metadata,
        num_classes=10,
    )
    label_run_metrics, label_run_per_class, label_run_confusion = classification_metrics(
        label_run_labels,
        label_run_logits,
        class_names,
        temperature=1.0,
    )
    predictions = attach_predictions(clip_metadata, mode_logits, 1.0, class_names)
    probability_predictions = attach_predictions(
        probability_metadata,
        probability_logits,
        1.0,
        class_names,
    )
    label_run_predictions = attach_predictions(
        label_run_metadata,
        label_run_logits,
        1.0,
        class_names,
    )
    for table in [predictions, probability_predictions, label_run_predictions, frame_predictions]:
        table.insert(0, "dataset", "eventsleep1")
        table.insert(1, "task", "activity")
        table.insert(2, "method", "resnet_e_published")
        table.insert(3, "loss", "weighted_ce")
        table.insert(4, "variant", profile)
        table.insert(5, "seed", int(seed))
        table.insert(6, "fold", "official_test")
    metrics.update(
        {
            "dataset": "eventsleep1",
            "task": "activity",
            "method": "resnet_e_published",
            "loss": "weighted_ce",
            "variant": profile,
            "seed": int(seed),
            "fold": "official_test",
            "evaluation_unit": "explicit_clip_id_mode_vote",
            "primary_metric_matching_paper": "balanced_accuracy",
            "best_epoch": int(best_epoch),
            "best_validation_balanced_accuracy": float(best_score),
            "best_validation_macro_f1": float(best_validation_macro_f1),
            "training_seconds": float(training_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_ms_per_prediction": 1000.0 * inference_seconds / max(len(frame_labels), 1),
            "inference_ms_per_frame": 1000.0 * inference_seconds / max(len(frame_labels), 1),
            "frame_balanced_accuracy": float(frame_metrics["balanced_accuracy"]),
            "frame_macro_f1": float(frame_metrics["macro_f1"]),
            "probability_sum_clip_balanced_accuracy": float(probability_metrics["balanced_accuracy"]),
            "probability_sum_clip_macro_f1": float(probability_metrics["macro_f1"]),
            "released_label_run_balanced_accuracy": float(label_run_metrics["balanced_accuracy"]),
            "released_label_run_macro_f1": float(label_run_metrics["macro_f1"]),
            "paper_reported_reference_balanced_accuracy": 0.82,
            "paper_reported_clips": 1016,
            "current_release_clips": int(len(manifest)),
            "exact_paper_data_cardinality": bool(len(manifest) == 1016),
            "pretrained": bool(pretrained),
            **parameters,
        }
    )
    if device.startswith("cuda"):
        metrics["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
    per_class.insert(0, "dataset", "eventsleep1")
    per_class.insert(1, "task", "activity")
    per_class.insert(2, "method", "resnet_e_published")
    per_class.insert(3, "loss", "weighted_ce")
    per_class.insert(4, "variant", profile)
    per_class.insert(5, "seed", int(seed))
    per_class.insert(6, "fold", "official_test")
    aggregation_table = pd.DataFrame(
        [
            {"aggregation": "explicit_clip_mode_primary", **metrics},
            {"aggregation": "explicit_clip_probability_sum", **probability_metrics},
            {"aggregation": "released_contiguous_label_runs_diagnostic", **label_run_metrics},
            {"aggregation": "frame_level", **frame_metrics},
        ]
    )
    atomic_json(run_dir / "metrics.json", metrics)
    atomic_json(run_dir / "frame_metrics.json", frame_metrics)
    atomic_csv(run_dir / "per_class_metrics.csv", per_class)
    atomic_csv(run_dir / "frame_per_class_metrics.csv", frame_per_class)
    atomic_csv(run_dir / "probability_sum_per_class_metrics.csv", probability_per_class)
    atomic_csv(run_dir / "released_label_run_per_class_metrics.csv", label_run_per_class)
    atomic_csv(run_dir / "predictions.csv", predictions)
    atomic_csv(run_dir / "frame_predictions.csv", frame_predictions)
    atomic_csv(run_dir / "probability_sum_predictions.csv", probability_predictions)
    atomic_csv(run_dir / "released_label_run_predictions.csv", label_run_predictions)
    atomic_csv(run_dir / "clip_aggregation_metrics.csv", aggregation_table)
    atomic_csv(run_dir / "per_unit_metrics.csv", per_unit_metrics(predictions, class_names))
    atomic_csv(
        run_dir / "confusion_matrix.csv",
        pd.DataFrame(confusion, index=class_names, columns=class_names).reset_index(names="true_class"),
    )
    atomic_csv(
        run_dir / "frame_confusion_matrix.csv",
        pd.DataFrame(frame_confusion, index=class_names, columns=class_names).reset_index(names="true_class"),
    )
    atomic_csv(
        run_dir / "probability_sum_confusion_matrix.csv",
        pd.DataFrame(probability_confusion, index=class_names, columns=class_names).reset_index(names="true_class"),
    )
    atomic_csv(
        run_dir / "released_label_run_confusion_matrix.csv",
        pd.DataFrame(label_run_confusion, index=class_names, columns=class_names).reset_index(names="true_class"),
    )
    atomic_json(
        run_dir / "run_configuration.json",
        {
            "created_utc": utc_now(),
            "pipeline_version": VERSION,
            "profile": profile,
            "profile_settings": settings,
            "published_configuration": published_resnete_configuration(profile, pretrained),
            "split": split,
            "device": device,
            "seed": int(seed),
            "important_interpretation": (
                "This is a current-release reproduction. The downloaded release contains "
                f"{len(manifest)} clips while the paper reports 1016."
            ),
        },
    )
    atomic_json(
        completion,
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "status": "success",
            "profile": profile,
        },
    )
    logging.info(
        "%s completed: clip balanced accuracy %.4f, macro-F1 %.4f",
        run_name,
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )
    return metrics


def run_published_resnete_command(
    args: argparse.Namespace,
    paths: DataPaths,
) -> dict[str, Any]:
    profile = str(args.profile)
    stage = str(args.stage)
    output = args.output.resolve()
    device = resolve_device(args.device)
    report: dict[str, Any] = {
        "started_utc": utc_now(),
        "pipeline_version": VERSION,
        "profile": profile,
        "stage": stage,
        "published_configuration": published_resnete_configuration(profile, bool(args.pretrained)),
    }
    if stage in {"all", "prepare"}:
        manifest = build_published_resnete_cache(
            paths,
            profile,
            int(args.cache_workers),
            bool(args.overwrite_cache),
        )
        report["prepared_clips"] = int(len(manifest))
        report["prepared_frames"] = int(manifest["resnete_frame_count"].sum())
    cache_report = verify_published_resnete_cache(
        paths,
        profile,
        bool(args.full_cache_check),
    )
    report["cache_verification"] = cache_report
    if stage in {"all", "check"}:
        report["model_check"] = check_published_resnete_model(
            paths,
            output,
            profile,
            device,
            bool(args.pretrained),
            bool(args.full_cache_check),
        )
    results: list[dict[str, Any]] = []
    if stage in {"all", "train"}:
        for seed in args.seeds:
            results.append(
                run_published_resnete_experiment(
                    paths,
                    output,
                    profile,
                    int(seed),
                    device,
                    int(args.workers),
                    bool(args.pretrained),
                    bool(args.overwrite_runs),
                )
            )
        result_frame = pd.DataFrame(results)
        summary_dir = output / "published_resnete" / profile
        atomic_csv(summary_dir / "run_metrics.csv", result_frame)
        report["completed_runs"] = int(len(results))
        report["seeds"] = [int(value) for value in args.seeds]
    report["completed_utc"] = utc_now()
    report["passed"] = True
    command_dir = output / "published_resnete" / profile
    atomic_json(command_dir / f"{stage}_completed_successfully.json", report)
    return report


# ---------------------------------------------------------------------------
# Exploratory analysis and paper analysis
# ---------------------------------------------------------------------------


def run_eda(paths: DataPaths, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    es1 = pd.read_csv(paths.es1_manifest, low_memory=False)
    es2 = pd.read_csv(paths.es2_manifest, low_memory=False)

    es1_distribution = (
        es1.groupby(["split", "label", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    atomic_csv(table_dir / "eventsleep1_class_distribution.csv", es1_distribution)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    sns.barplot(data=es1_distribution, x="label_name", y="samples", hue="split", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Clips")
    ax.tick_params(axis="x", rotation=38)
    ax.legend(frameon=False, title=None)
    save_figure(fig, figure_dir / "eventsleep1_class_distribution.pdf")
    plt.close(fig)

    movement = es2[bool_series(es2["supervised_movement"])].copy()
    movement["movement_label"] = pd.to_numeric(movement["movement_label"], errors="raise").astype(int)
    movement_distribution = (
        movement.groupby(["subject", "movement_label", "movement_label_name"], as_index=False)
        .size()
        .rename(columns={"size": "frames"})
    )
    atomic_csv(table_dir / "eventsleep2_movement_distribution_by_subject.csv", movement_distribution)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    sns.barplot(data=movement_distribution, x="movement_label_name", y="frames", hue="subject", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Supervised frames")
    ax.tick_params(axis="x", rotation=32)
    ax.legend(frameon=False, title="Subject")
    save_figure(fig, figure_dir / "eventsleep2_movement_distribution.pdf")
    plt.close(fig)

    posture = es2[bool_series(es2["supervised_posture"])].copy()
    posture_distribution = (
        posture.groupby(["subject", "posture_label", "posture_label_name"], as_index=False)
        .size()
        .rename(columns={"size": "frames"})
    )
    atomic_csv(table_dir / "eventsleep2_posture_distribution_by_subject.csv", posture_distribution)

    gaps: list[pd.DataFrame] = []
    transitions = np.zeros((6, 6), dtype=np.int64)
    for recording, group in es2.groupby("recording_key", sort=True):
        group = group.sort_values("array_index", kind="stable")
        timestamps = group["anchor_timestamp_us"].to_numpy(dtype=np.int64)
        if len(timestamps) > 1:
            gaps.append(
                pd.DataFrame(
                    {
                        "recording_key": recording,
                        "subject": int(group["subject"].iloc[0]),
                        "gap_ms": np.maximum(np.diff(timestamps), 0) / 1000.0,
                    }
                )
            )
        mask = bool_series(group["supervised_movement"]).to_numpy(dtype=bool)
        labels = pd.to_numeric(group["movement_label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        adjacent = mask[:-1] & mask[1:]
        for left, right in zip(labels[:-1][adjacent], labels[1:][adjacent]):
            if 0 <= left < 6 and 0 <= right < 6:
                transitions[left, right] += 1
    gap_frame = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame(columns=["recording_key", "subject", "gap_ms"])
    atomic_csv(table_dir / "eventsleep2_timestamp_gaps.csv", gap_frame)
    transition_frame = pd.DataFrame(transitions, index=[STAGE_NAMES[i] for i in range(6)], columns=[STAGE_NAMES[i] for i in range(6)])
    atomic_csv(table_dir / "eventsleep2_transition_counts.csv", transition_frame.reset_index(names="from_class"))
    if len(gap_frame):
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        clipped = gap_frame.copy()
        upper = float(clipped["gap_ms"].quantile(0.99))
        clipped = clipped[clipped["gap_ms"] <= upper]
        sns.histplot(data=clipped, x="gap_ms", hue="subject", element="step", stat="density", common_norm=False, ax=ax)
        ax.set_xlabel("Inter-frame timestamp gap (ms)")
        ax.set_ylabel("Density")
        save_figure(fig, figure_dir / "eventsleep2_timestamp_gap_distribution.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    normalized = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1)
    sns.heatmap(normalized, cmap="mako", vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "Row-normalized frequency"}, ax=ax)
    ax.set_xticklabels([STAGE_NAMES[i] for i in range(6)], rotation=38, ha="right")
    ax.set_yticklabels([STAGE_NAMES[i] for i in range(6)], rotation=0)
    ax.set_xlabel("Next class")
    ax.set_ylabel("Current class")
    save_figure(fig, figure_dir / "eventsleep2_transition_matrix.pdf")
    plt.close(fig)

    dataset_table = pd.DataFrame(
        [
            {
                "dataset": "EventSleep1",
                "subjects": es1["subject"].nunique(),
                "recordings_or_clips": len(es1),
                "supervised_samples": len(es1),
                "classes": 10,
                "evaluation": "Official subject-independent TRAIN/TEST",
            },
            {
                "dataset": "EventSleep2 movement",
                "subjects": es2["subject"].nunique(),
                "recordings_or_clips": es2["recording_key"].nunique(),
                "supervised_samples": int(bool_series(es2["supervised_movement"]).sum()),
                "classes": 6,
                "evaluation": "3-fold LOSO",
            },
            {
                "dataset": "EventSleep2 posture",
                "subjects": es2["subject"].nunique(),
                "recordings_or_clips": es2["recording_key"].nunique(),
                "supervised_samples": int(bool_series(es2["supervised_posture"]).sum()),
                "classes": 4,
                "evaluation": "Secondary 3-fold LOSO",
            },
        ]
    )
    atomic_csv(table_dir / "dataset_characteristics.csv", dataset_table)
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "eventsleep1_clips": len(es1),
        "eventsleep2_frames": len(es2),
        "figures": len(list(figure_dir.glob("*.pdf"))),
        "tables": len(list(table_dir.glob("*.csv"))),
    }
    atomic_json(output / "eda_completed.json", report)
    return report


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_unit_difference(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    iterations: int,
    seed: int = 20260829,
) -> dict[str, float]:
    keys = ["seed", "fold", "subject", "recording_key"]
    left = proposed[keys + ["macro_f1"]].rename(columns={"macro_f1": "proposed"})
    right = baseline[keys + ["macro_f1"]].rename(columns={"macro_f1": "baseline"})
    merged = left.merge(right, on=keys, how="inner")
    if merged.empty:
        return {"mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_two_sided": float("nan"), "units": 0}
    unit = merged.groupby("subject", as_index=False)[["proposed", "baseline"]].mean()
    differences = (unit["proposed"] - unit["baseline"]).to_numpy(dtype=np.float64)
    if len(differences) < 3:
        return {
            "mean_difference": float(differences.mean()),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_two_sided": float("nan"),
            "units": len(differences),
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selection = rng.integers(0, len(differences), size=len(differences))
        samples[index] = differences[selection].mean()
    p_value = 2.0 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "p_two_sided": min(1.0, p_value),
        "units": int(len(differences)),
    }


def analyze_results(output_root: Path, bootstrap_iterations: int = 10_000) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    analysis = output_root / "analysis"
    table_dir = analysis / "tables"
    paper_dir = analysis / "paper_tables"
    figure_dir = analysis / "figures"
    statistics_dir = analysis / "statistical_tests"
    for path in [table_dir, paper_dir, figure_dir, statistics_dir]:
        path.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    unit_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for metrics_path in sorted((output_root / "runs").glob("*/metrics.json")):
        run_dir = metrics_path.parent
        completion = run_dir / "completed_successfully.json"
        if not completion.exists():
            logging.warning("Ignoring incomplete run: %s", run_dir)
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_rows.append(metrics)
        if (run_dir / "per_class_metrics.csv").exists():
            per_class_frames.append(pd.read_csv(run_dir / "per_class_metrics.csv"))
        if (run_dir / "per_unit_metrics.csv").exists():
            units = pd.read_csv(run_dir / "per_unit_metrics.csv")
            for key in ["dataset", "task", "method", "loss", "variant", "seed", "fold"]:
                units[key] = metrics[key]
            unit_frames.append(units)
        if (run_dir / "predictions.csv").exists():
            prediction_frames.append(pd.read_csv(run_dir / "predictions.csv", low_memory=False))
    if not metric_rows:
        raise RuntimeError(f"No completed runs found under {output_root / 'runs'}")
    metrics_frame = pd.DataFrame(metric_rows)
    per_class = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    atomic_csv(table_dir / "all_run_metrics.csv", metrics_frame)
    if len(per_class):
        atomic_csv(table_dir / "all_per_class_metrics.csv", per_class)
    if len(units):
        atomic_csv(table_dir / "all_per_unit_metrics.csv", units)

    group_keys = ["dataset", "task", "method", "loss", "variant"]
    numeric_metrics = [
        key
        for key in [
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "worst_class_f1",
            "kappa",
            "ece",
            "nll",
            "brier",
            "segment_f1_at_10",
            "segment_f1_at_25",
            "segment_f1_at_50",
            "edit_score",
            "boundary_f1_tolerance_1",
            "training_seconds",
            "inference_ms_per_prediction",
            "trainable_parameters",
            "peak_gpu_memory_mib",
        ]
        if key in metrics_frame.columns
    ]
    seed_level = metrics_frame.groupby(group_keys + ["seed"], as_index=False)[numeric_metrics].mean(numeric_only=True)
    summary_mean = seed_level.groupby(group_keys, as_index=False)[numeric_metrics].mean(numeric_only=True)
    summary_std = seed_level.groupby(group_keys, as_index=False)[numeric_metrics].std(ddof=1, numeric_only=True).fillna(0)
    summary = summary_mean.merge(summary_std, on=group_keys, suffixes=("_mean", "_std"))
    atomic_csv(paper_dir / "main_results_mean_std.csv", summary)
    (paper_dir / "main_results_mean_std.tex").write_text(
        summary.to_latex(index=False, float_format=lambda value: f"{value:.4f}", escape=True), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    plot = summary.copy()
    plot["method_label"] = plot["method"] + "\n" + plot["loss"]
    sns.barplot(data=plot, x="method_label", y="macro_f1_mean", hue="dataset", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=38)
    ax.legend(frameon=False, title=None)
    save_figure(fig, figure_dir / "main_macro_f1_comparison.pdf")
    plt.close(fig)

    comparisons: list[dict[str, Any]] = []
    if len(units):
        for (dataset, task), task_units in units.groupby(["dataset", "task"], sort=True):
            proposed = task_units[(task_units["method"] == "eventsleepformer") & (task_units["variant"] == "full")]
            if proposed.empty:
                continue
            for (method, loss, variant), baseline in task_units.groupby(["method", "loss", "variant"], sort=True):
                if method == "eventsleepformer" and variant == "full":
                    continue
                result = bootstrap_unit_difference(proposed, baseline, bootstrap_iterations)
                comparisons.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "baseline_method": method,
                        "baseline_loss": loss,
                        "baseline_variant": variant,
                        **result,
                    }
                )
    comparison_frame = pd.DataFrame(comparisons)
    if len(comparison_frame):
        valid = comparison_frame["p_two_sided"].notna()
        comparison_frame.loc[valid, "p_holm"] = holm_adjust(comparison_frame.loc[valid, "p_two_sided"].tolist())
        atomic_csv(statistics_dir / "paired_hierarchical_bootstrap.csv", comparison_frame)

    if len(per_class):
        per_class_summary = (
            per_class.groupby(group_keys + ["class_id", "class_name"], as_index=False)[["f1", "precision", "recall"]]
            .mean(numeric_only=True)
        )
        atomic_csv(paper_dir / "per_class_results.csv", per_class_summary)
        proposed_class = per_class_summary[(per_class_summary["method"] == "eventsleepformer") & (per_class_summary["variant"] == "full")]
        if len(proposed_class):
            pivot = proposed_class.pivot_table(index=["dataset", "task"], columns="class_name", values="f1", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7.2, max(2.5, 0.7 * len(pivot))))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1, ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("")
            save_figure(fig, figure_dir / "eventsleepformer_per_class_f1.pdf")
            plt.close(fig)

    if len(predictions):
        ensemble_keys = ["dataset", "task", "method", "loss", "variant", "fold", "sample_id", "subject", "recording_key", "frame_index", "y_true"]
        ensemble_parts: list[pd.DataFrame] = []
        for _, task_predictions in predictions.groupby(["dataset", "task"], sort=True):
            probability_columns = [
                column
                for column in task_predictions
                if column.startswith("prob_") and task_predictions[column].notna().any()
            ]
            part = task_predictions.groupby(ensemble_keys, as_index=False)[probability_columns].mean(numeric_only=True)
            part["confidence"] = part[probability_columns].max(axis=1)
            part["y_pred"] = part[probability_columns].to_numpy().argmax(axis=1)
            part["correct"] = (part["y_pred"] == part["y_true"]).astype(int)
            ensemble_parts.append(part)
        ensemble = pd.concat(ensemble_parts, ignore_index=True, sort=False)
        atomic_csv(table_dir / "seed_ensembled_predictions.csv", ensemble)
        selected = ensemble[(ensemble["method"] == "eventsleepformer") & (ensemble["variant"] == "full")]
        if len(selected):
            for (dataset, task), task_selected in selected.groupby(["dataset", "task"], sort=True):
                fig, ax = plt.subplots(figsize=(4.2, 4.0))
                bins = np.linspace(0, 1, 11)
                task_selected = task_selected.copy()
                task_selected["bin"] = pd.cut(task_selected["confidence"], bins=bins, include_lowest=True)
                reliability = task_selected.groupby("bin", observed=False).agg(confidence=("confidence", "mean"), accuracy=("correct", "mean"), count=("correct", "size")).dropna()
                ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
                ax.plot(reliability["confidence"], reliability["accuracy"], marker="o")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Mean confidence")
                ax.set_ylabel("Empirical accuracy")
                safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{dataset}_{task}").strip("_").lower()
                save_figure(fig, figure_dir / f"eventsleepformer_reliability_{safe}.pdf")
                plt.close(fig)

    if "trainable_parameters_mean" in summary and "macro_f1_mean" in summary:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        sns.scatterplot(data=summary, x="trainable_parameters_mean", y="macro_f1_mean", hue="method", style="dataset", s=80, ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("Macro-F1")
        ax.legend(frameon=False, fontsize=7)
        save_figure(fig, figure_dir / "accuracy_efficiency_pareto.pdf")
        plt.close(fig)

    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "completed_runs": len(metrics_frame),
        "method_configurations": len(summary),
        "figures": len(list(figure_dir.glob("*.pdf"))),
        "bootstrap_iterations": bootstrap_iterations,
    }
    atomic_json(analysis / "analysis_completed.json", report)
    return report


# ---------------------------------------------------------------------------
# Command orchestration and smoke test
# ---------------------------------------------------------------------------


def model_config_from_args(args: argparse.Namespace, method: str = "eventsleepformer", variant: str = "full") -> ModelConfig:
    return ModelConfig(
        method=method,
        num_classes=10,
        clip_level=True,
        d_model=int(args.d_model),
        temporal_layers=int(args.temporal_layers),
        temporal_heads=int(args.temporal_heads),
        dropout=float(args.dropout),
        sequence_length=int(args.sequence_length),
        image_size=int(args.image_size),
        dino_model=str(args.dino_model),
        pretrained=bool(args.pretrained),
        backbone_mode=str(args.backbone_mode),
        lora_rank=int(args.lora_rank),
        lora_blocks=int(args.lora_blocks),
        spatial_microbatch=int(args.spatial_microbatch),
        causal=bool(args.causal),
        variant=variant,
    )


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        epochs=int(args.epochs),
        patience=int(args.patience),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        learning_rate=float(args.learning_rate),
        backbone_lr_multiplier=float(args.backbone_lr_multiplier),
        weight_decay=float(args.weight_decay),
        warmup_epochs=int(args.warmup_epochs),
        accumulation_steps=int(args.accumulation_steps),
        gradient_clip=float(args.gradient_clip),
        amp=bool(args.amp),
        label_smoothing=float(args.label_smoothing),
        boundary_weight=float(args.boundary_weight),
        deterministic=bool(args.deterministic),
    )


def load_locked_configuration(path: Path) -> tuple[str, ModelConfig, TrainConfig, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "method_spec" not in payload or "model" not in payload or "training" not in payload:
        raise ValueError(f"Locked configuration is missing method_spec/model/training: {path}")
    method_spec = str(payload["method_spec"])
    model = ModelConfig(**payload["model"])
    training = TrainConfig(**payload["training"])
    expected = payload.get("configuration_id")
    observed = configuration_id(model, training, method_spec)
    if expected is not None and str(expected) != observed:
        raise RuntimeError(
            f"Locked configuration hash mismatch: file={expected}, recomputed={observed}; refusing to evaluate"
        )
    return method_spec, model, training, payload


def _task_and_folds(args: argparse.Namespace, final_evaluation: bool) -> tuple[str, list[int | None]]:
    if args.dataset == "eventsleep1":
        if args.task != "activity":
            raise ValueError("EventSleep1 task must be 'activity'")
        return "activity", [None]
    if args.task not in {"movement", "posture"}:
        raise ValueError("EventSleep2 task must be movement or posture")
    folds: list[int | None] = [int(value) for value in args.held_out_subjects]
    if final_evaluation and sorted(int(value) for value in folds) != [1, 2, 3] and not args.allow_partial_folds:
        raise ValueError(
            "Final EventSleep2 evaluation must use held-out subjects 1 2 3; "
            "use --allow-partial-folds only for pilots"
        )
    return str(args.task), folds


def run_training_grid(
    args: argparse.Namespace,
    paths: DataPaths,
    variants: Sequence[str] = ("full",),
) -> pd.DataFrame:
    device = resolve_device(args.device)
    task, folds = _task_and_folds(args, final_evaluation=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    locked = getattr(args, "locked_config", None)
    if locked is not None:
        if len(variants) != 1 or variants[0] != "full":
            raise ValueError("--locked-config is supported only for full-model train evaluation, not ablations")
        method_spec, locked_model, locked_training, locked_payload = load_locked_configuration(Path(locked).resolve())
        methods = [method_spec]
        logging.info(
            "Using locked configuration %s from %s",
            locked_payload.get("configuration_id", configuration_id(locked_model, locked_training, method_spec)),
            Path(locked).resolve(),
        )
    else:
        methods = list(args.methods)
        locked_model = None
        locked_training = None

    for variant in variants:
        variant_methods = methods if variant == "full" else ["eventsleepformer@balanced_softmax"]
        for method_spec in variant_methods:
            method, _ = parse_method_spec(method_spec)
            if variant != "full" and method != "eventsleepformer":
                continue
            if locked_model is not None and locked_training is not None:
                base_model = copy.deepcopy(locked_model)
                base_model.variant = variant
                base_training = copy.deepcopy(locked_training)
            else:
                base_model = model_config_from_args(args, method, variant)
                base_training = train_config_from_args(args)
            for seed in args.seeds:
                for held_out in folds:
                    try:
                        result = run_single_experiment(
                            paths=paths,
                            output_root=output,
                            dataset_name=args.dataset,
                            task=task,
                            method_spec=method_spec,
                            seed=int(seed),
                            held_out_subject=held_out,
                            validation_fold=int(args.validation_fold),
                            es1_validation_policy=str(args.es1_validation_policy),
                            model_config=copy.deepcopy(base_model),
                            train_config=copy.deepcopy(base_training),
                            device=device,
                            overwrite=bool(args.overwrite),
                            evaluation_scope="test",
                            keep_checkpoint=True,
                        )
                        results.append(result)
                        atomic_csv(output / "grid_progress.csv", pd.DataFrame(results))
                    except Exception:
                        logging.exception(
                            "Run failed: dataset=%s task=%s method=%s variant=%s seed=%s held_out=%s",
                            args.dataset, task, method_spec, variant, seed, held_out,
                        )
                        if not args.continue_on_error:
                            raise
    result_frame = pd.DataFrame(results)
    atomic_csv(output / "grid_results.csv", result_frame)
    return result_frame


def _grid_values(value: Sequence[Any] | None, base: Any) -> list[Any]:
    return list(value) if value is not None and len(value) else [base]


def tuning_candidates_from_args(args: argparse.Namespace) -> list[tuple[str, ModelConfig, TrainConfig]]:
    """Build a bounded Cartesian search around the supplied base configuration."""
    import itertools

    base_model = model_config_from_args(args, "eventsleepformer", "full")
    base_training = train_config_from_args(args)
    losses = _grid_values(args.losses, "balanced_softmax")
    axes = [
        _grid_values(args.learning_rates, base_training.learning_rate),
        _grid_values(args.backbone_lr_multipliers, base_training.backbone_lr_multiplier),
        _grid_values(args.weight_decays, base_training.weight_decay),
        _grid_values(args.d_models, base_model.d_model),
        _grid_values(args.temporal_layers_grid, base_model.temporal_layers),
        _grid_values(args.temporal_heads_grid, base_model.temporal_heads),
        _grid_values(args.dropouts, base_model.dropout),
        _grid_values(args.lora_ranks, base_model.lora_rank),
        _grid_values(args.lora_blocks_grid, base_model.lora_blocks),
        _grid_values(args.label_smoothings, base_training.label_smoothing),
        _grid_values(args.warmup_epochs_grid, base_training.warmup_epochs),
        _grid_values(args.boundary_weights, base_training.boundary_weight),
        losses,
    ]
    raw_count = math.prod(len(axis) for axis in axes)
    if raw_count > int(args.max_configurations):
        raise ValueError(
            f"Requested tuning grid has {raw_count} configurations, exceeding --max-configurations "
            f"{args.max_configurations}. Use staged searches instead of a large blind grid."
        )
    candidates: list[tuple[str, ModelConfig, TrainConfig]] = []
    seen: set[str] = set()
    for values in itertools.product(*axes):
        (
            learning_rate, backbone_lr_multiplier, weight_decay, d_model, temporal_layers, temporal_heads,
            dropout, lora_rank, lora_blocks, label_smoothing, warmup_epochs, boundary_weight, loss_name,
        ) = values
        if int(d_model) % int(temporal_heads) != 0:
            logging.warning(
                "Skipping invalid candidate: d_model=%s is not divisible by temporal_heads=%s",
                d_model, temporal_heads,
            )
            continue
        model = copy.deepcopy(base_model)
        model.d_model = int(d_model)
        model.temporal_layers = int(temporal_layers)
        model.temporal_heads = int(temporal_heads)
        model.dropout = float(dropout)
        model.lora_rank = int(lora_rank)
        model.lora_blocks = int(lora_blocks)
        training = copy.deepcopy(base_training)
        training.learning_rate = float(learning_rate)
        training.backbone_lr_multiplier = float(backbone_lr_multiplier)
        training.weight_decay = float(weight_decay)
        training.label_smoothing = float(label_smoothing)
        training.warmup_epochs = int(warmup_epochs)
        training.boundary_weight = float(boundary_weight)
        method_spec = f"eventsleepformer@{loss_name}"
        cid = configuration_id(model, training, method_spec)
        if cid not in seen:
            candidates.append((method_spec, model, training))
            seen.add(cid)
    if not candidates:
        raise ValueError("No valid tuning candidates remain after validation")
    return candidates


def run_tuning_grid(args: argparse.Namespace, paths: DataPaths) -> pd.DataFrame:
    """Tune EventSleepFormer using validation data only; never instantiate test datasets."""
    device = resolve_device(args.device)
    task, folds = _task_and_folds(args, final_evaluation=False)
    if args.dataset == "eventsleep1" and str(args.es1_validation_policy) != "matched_resnete":
        logging.warning(
            "EventSleep1 tuning is using validation policy %s; matched_resnete is recommended for the final paper",
            args.es1_validation_policy,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = tuning_candidates_from_args(args)
    logging.info("Validation-only tuning candidates: %d", len(candidates))
    rows: list[dict[str, Any]] = []
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for method_spec, model_config, train_config in candidates:
        cid = configuration_id(model_config, train_config, method_spec)
        candidate_payloads[cid] = {
            "configuration_id": cid,
            "method_spec": method_spec,
            "model": asdict(model_config),
            "training": asdict(train_config),
        }
        for seed in args.seeds:
            for held_out in folds:
                try:
                    metrics = run_single_experiment(
                        paths=paths,
                        output_root=output,
                        dataset_name=args.dataset,
                        task=task,
                        method_spec=method_spec,
                        seed=int(seed),
                        held_out_subject=held_out,
                        validation_fold=int(args.validation_fold),
                        es1_validation_policy=str(args.es1_validation_policy),
                        model_config=copy.deepcopy(model_config),
                        train_config=copy.deepcopy(train_config),
                        device=device,
                        overwrite=bool(args.overwrite),
                        evaluation_scope="validation_only",
                        keep_checkpoint=bool(args.keep_tuning_checkpoints),
                    )
                    row = {
                        **metrics,
                        "configuration_id": cid,
                        "learning_rate": train_config.learning_rate,
                        "backbone_lr_multiplier": train_config.backbone_lr_multiplier,
                        "weight_decay": train_config.weight_decay,
                        "d_model": model_config.d_model,
                        "temporal_layers": model_config.temporal_layers,
                        "temporal_heads": model_config.temporal_heads,
                        "dropout": model_config.dropout,
                        "lora_rank": model_config.lora_rank,
                        "lora_blocks": model_config.lora_blocks,
                        "label_smoothing": train_config.label_smoothing,
                        "warmup_epochs": train_config.warmup_epochs,
                        "boundary_weight": train_config.boundary_weight,
                    }
                    rows.append(row)
                    atomic_csv(output / "tuning_progress.csv", pd.DataFrame(rows))
                except Exception:
                    logging.exception(
                        "Tuning run failed: config=%s seed=%s held_out=%s", cid, seed, held_out
                    )
                    if not args.continue_on_error:
                        raise
    runs = pd.DataFrame(rows)
    atomic_csv(output / "tuning_runs.csv", runs)
    if runs.empty:
        raise RuntimeError("No tuning runs completed")
    if runs["official_test_evaluated"].astype(bool).any():
        raise RuntimeError("Leakage guard failed: a tuning run evaluated the official test set")
    metric_columns = [
        column for column in ["macro_f1", "balanced_accuracy", "accuracy", "worst_class_f1"] if column in runs
    ]
    grouped = runs.groupby("configuration_id", as_index=False)
    summary = grouped[metric_columns].agg(["mean", "std"])
    summary.columns = [
        "configuration_id" if column[0] == "configuration_id" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    # pandas may omit configuration_id from MultiIndex columns depending on version; recover robustly.
    if "configuration_id" not in summary.columns:
        summary.insert(0, "configuration_id", [key for key, _ in grouped])
    counts = runs.groupby("configuration_id").size().rename("completed_runs").reset_index()
    summary = summary.merge(counts, on="configuration_id", how="left")
    parameter_columns = [
        "configuration_id", "loss", "learning_rate", "backbone_lr_multiplier", "weight_decay",
        "d_model", "temporal_layers", "temporal_heads", "dropout", "lora_rank", "lora_blocks",
        "label_smoothing", "warmup_epochs", "boundary_weight",
    ]
    parameters = runs[[column for column in parameter_columns if column in runs]].drop_duplicates("configuration_id")
    summary = summary.merge(parameters, on="configuration_id", how="left")
    summary = summary.sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    atomic_csv(output / "tuning_summary.csv", summary)
    best_id = str(summary.iloc[0]["configuration_id"])
    best = dict(candidate_payloads[best_id])
    best.update(
        {
            "selected_utc": utc_now(),
            "pipeline_version": VERSION,
            "selection_source": "validation_only",
            "selection_metric": "mean_validation_macro_f1",
            "official_test_evaluated_during_selection": False,
            "validation_macro_f1_mean": float(summary.iloc[0]["macro_f1_mean"]),
            "validation_macro_f1_std": float(summary.iloc[0]["macro_f1_std"]) if not pd.isna(summary.iloc[0]["macro_f1_std"]) else None,
            "validation_balanced_accuracy_mean": float(summary.iloc[0]["balanced_accuracy_mean"]),
            "dataset": args.dataset,
            "task": task,
            "es1_validation_policy": str(args.es1_validation_policy),
            "validation_fold": int(args.validation_fold),
        }
    )
    atomic_json(output / "best_configuration.json", best)
    atomic_json(
        output / "tuning_completed.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "candidates": len(candidates),
            "completed_runs": len(runs),
            "best_configuration_id": best_id,
            "official_test_evaluated": False,
        },
    )
    return summary


def check_model(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    config = model_config_from_args(args, "eventsleepformer", "full")
    config.num_classes = 6
    config.clip_level = False
    model = SleepActivityModel(config).to(device)
    model.eval()
    batch, length = 1, min(4, config.sequence_length)
    frames = torch.rand(batch, length, 2, 120, 160, device=device)
    # Use realistic large absolute AEDAT timestamps so the check guards
    # against mixed-precision overflow in time-gap attention.
    timestamp_origin = 1_742_341_455_332_413
    timestamps = timestamp_origin + torch.arange(length, device=device, dtype=torch.int64).view(1, -1).repeat(batch, 1) * 150_000
    descriptors = torch.rand(batch, length, config.descriptor_dim, device=device)
    mask = torch.ones(batch, length, dtype=torch.bool, device=device)
    with torch.inference_mode(), autocast_context(device, device.startswith("cuda")):
        output = model(frames, timestamps, descriptors, mask)
    logits = output["logits"]
    assert isinstance(logits, torch.Tensor)
    finite = bool(torch.isfinite(logits).all().item())
    if tuple(logits.shape) != (batch, length, 6) or not finite:
        nonfinite = int((~torch.isfinite(logits)).sum().item())
        raise RuntimeError(
            "Model check returned invalid logits: "
            f"shape={tuple(logits.shape)}, finite={finite}, nonfinite_values={nonfinite}"
        )
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "device": device,
        "dino_model": config.dino_model,
        "pretrained": config.pretrained,
        "logit_shape": list(logits.shape),
        **model_parameter_counts(model),
        "passed": True,
    }
    return report


def create_synthetic_preprocessed(root: Path, sequence_length: int = 8) -> None:
    rng = np.random.default_rng(20260829)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        root / "completed_successfully.json",
        {"completed_utc": utc_now(), "pipeline_version": "synthetic", "verification": "PASSED", "datasets_kept_separate": True},
    )
    es1_rows: list[dict[str, Any]] = []
    subject_folds: list[dict[str, Any]] = []
    for subject in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
        subject_folds.append({"subject": subject, "official_split": "TRAIN", "validation_fold_within_official_train": len(subject_folds) % 5})
        for label in range(10):
            sample_id = f"synthetic_train_s{subject:02d}_l{label}"
            frames = rng.random((sequence_length, 2, 24, 32), dtype=np.float32).astype(np.float16)
            timestamps = np.arange(sequence_length, dtype=np.int64) * 150_000
            relative = Path("eventsleep1") / "temporal_sequences" / "arrays" / f"{sample_id}.npz"
            atomic_npz(root / relative, frames=frames, timestamps_us=timestamps, descriptors=frame_descriptors(frames, timestamps))
            es1_rows.append({"sample_id": sample_id, "split": "TRAIN", "subject": subject, "configuration": 1, "label": label, "label_name": STAGE_NAMES[label], "frame_path": "synthetic", "source_path": "synthetic", "sequence_path": relative.as_posix()})
    for subject in [9, 12, 13, 14]:
        subject_folds.append({"subject": subject, "official_split": "TEST", "validation_fold_within_official_train": -1})
        for label in range(10):
            sample_id = f"synthetic_test_s{subject:02d}_l{label}"
            frames = rng.random((sequence_length, 2, 24, 32), dtype=np.float32).astype(np.float16)
            timestamps = np.arange(sequence_length, dtype=np.int64) * 150_000
            relative = Path("eventsleep1") / "temporal_sequences" / "arrays" / f"{sample_id}.npz"
            atomic_npz(root / relative, frames=frames, timestamps_us=timestamps, descriptors=frame_descriptors(frames, timestamps))
            es1_rows.append({"sample_id": sample_id, "split": "TEST", "subject": subject, "configuration": 1, "label": label, "label_name": STAGE_NAMES[label], "frame_path": "synthetic", "source_path": "synthetic", "sequence_path": relative.as_posix()})
    es1 = pd.DataFrame(es1_rows)
    atomic_csv(root / "eventsleep1" / "manifests" / "all_samples.csv", es1)
    atomic_csv(root / "eventsleep1" / "manifests" / "subject_folds.csv", pd.DataFrame(subject_folds))
    atomic_csv(root / "eventsleep1" / "temporal_sequences" / "manifest.csv", es1)

    es2_rows: list[dict[str, Any]] = []
    for subject in [1, 2, 3]:
        sequence_ids = [1, 2] if subject in {1, 2} else [1]
        for sequence in sequence_ids:
            key = f"subject{subject:02d}_seq{sequence:02d}"
            length = 24
            frames = rng.random((length, 2, 24, 32), dtype=np.float32).astype(np.float16)
            relative = Path("eventsleep2") / "frames" / f"{key}.npy"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            with (root / relative).open("wb") as handle:
                np.save(handle, frames, allow_pickle=False)
            timestamps = np.arange(length, dtype=np.int64) * 150_000 + subject * 10_000_000
            descriptors = frame_descriptors(frames, timestamps)
            descriptor_path = root / "eventsleep2" / "descriptors" / f"{key}.npy"
            descriptor_path.parent.mkdir(parents=True, exist_ok=True)
            with descriptor_path.open("wb") as handle:
                np.save(handle, descriptors, allow_pickle=False)
            for frame_index in range(length):
                movement = frame_index % 6
                posture = 6 + (frame_index % 4)
                es2_rows.append(
                    {
                        "frame_key": f"{key}_frame{frame_index:06d}",
                        "recording_key": key,
                        "subject": subject,
                        "sequence": sequence,
                        "array_index": frame_index,
                        "frame_index": frame_index,
                        "frame_path": relative.as_posix(),
                        "anchor_timestamp_us": int(timestamps[frame_index]),
                        "supervised_movement": True,
                        "movement_label": movement,
                        "movement_label_name": STAGE_NAMES[movement],
                        "supervised_posture": True,
                        "posture_label": posture,
                        "posture_label_name": STAGE_NAMES[posture],
                    }
                )
    atomic_csv(root / "eventsleep2" / "manifests" / "all_frames.csv", pd.DataFrame(es2_rows))


def smoke_test(output: Path) -> dict[str, Any]:
    existing = [] if not output.exists() else [path for path in output.iterdir() if path.name != "logs"]
    if existing:
        raise ValueError(f"Smoke-test output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    preprocess = output / "synthetic_preprocessed"
    create_synthetic_preprocessed(preprocess, sequence_length=8)
    paths = DataPaths(preprocess)
    preflight = validate_preprocessed(paths, full_array_check=False)
    synthetic_es1 = pd.read_csv(paths.es1_sequence_manifest, low_memory=False)
    synthetic_es1.attrs["subject_folds_path"] = str(
        paths.preprocess_root / "eventsleep1" / "manifests" / "subject_folds.csv"
    )
    matched_train, matched_validation, matched_test = es1_split(
        synthetic_es1, validation_fold=0, validation_policy="matched_resnete"
    )
    matched_split_check = {
        "train_subjects": sorted(matched_train["subject"].astype(int).unique().tolist()),
        "validation_subjects": sorted(matched_validation["subject"].astype(int).unique().tolist()),
        "test_subjects": sorted(matched_test["subject"].astype(int).unique().tolist()),
    }
    common_model = ModelConfig(
        method="eventsleepformer",
        num_classes=10,
        clip_level=True,
        d_model=64,
        temporal_layers=1,
        temporal_heads=4,
        sequence_length=8,
        image_size=0,
        dino_model="tiny_event_cnn",
        pretrained=False,
        backbone_mode="full",
        spatial_microbatch=32,
    )
    training = TrainConfig(
        epochs=1,
        patience=1,
        batch_size=4,
        workers=0,
        learning_rate=1e-3,
        warmup_epochs=0,
        amp=False,
        label_smoothing=0.0,
        boundary_weight=0.1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnete_check = synthetic_published_resnete_component_check(device)
    es1_result = run_single_experiment(
        paths,
        output / "results",
        "eventsleep1",
        "activity",
        "eventsleepformer@ce",
        13,
        None,
        0,
        "grouped_fold",
        common_model,
        training,
        device,
        False,
    )
    es2_model = copy.deepcopy(common_model)
    es2_model.num_classes = 6
    es2_model.clip_level = False
    es2_result = run_single_experiment(
        paths,
        output / "results",
        "eventsleep2",
        "movement",
        "eventsleepformer@ce",
        13,
        3,
        0,
        "grouped_fold",
        es2_model,
        training,
        device,
        False,
    )
    tuning_model = copy.deepcopy(common_model)
    tuning_training = copy.deepcopy(training)
    tuning_metrics = run_single_experiment(
        paths=paths,
        output_root=output / "tuning_results",
        dataset_name="eventsleep1",
        task="activity",
        method_spec="eventsleepformer@ce",
        seed=13,
        held_out_subject=None,
        validation_fold=0,
        es1_validation_policy="matched_resnete",
        model_config=tuning_model,
        train_config=tuning_training,
        device=device,
        overwrite=False,
        evaluation_scope="validation_only",
        keep_checkpoint=False,
    )
    if bool(tuning_metrics.get("official_test_evaluated", True)):
        raise RuntimeError("Smoke-test leakage guard failed: validation-only run touched test evaluation")
    eda = run_eda(paths, output / "results" / "eda")
    analysis = analyze_results(output / "results", bootstrap_iterations=200)
    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "device": device,
        "preflight": preflight,
        "eventsleep1_macro_f1": es1_result["macro_f1"],
        "eventsleep2_macro_f1": es2_result["macro_f1"],
        "published_resnete_component_check": resnete_check,
        "matched_es1_split_check": matched_split_check,
        "validation_only_tuning_macro_f1": tuning_metrics["macro_f1"],
        "validation_only_test_evaluated": tuning_metrics["official_test_evaluated"],
        "eda": eda,
        "analysis": analysis,
        "passed": True,
    }
    atomic_json(output / "smoke_test_completed.json", report)
    return report


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=DEFAULT_PREPROCESS_ROOT,
        help="Verified output directory from eventsleep_preprocess.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dino-model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone-mode", choices=["auto", "frozen", "lora", "last_block", "full"], default="auto")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-blocks", type=int, default=4)
    parser.add_argument("--spatial-microbatch", type=int, default=64)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EventSleepFormer server experiment pipeline")
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate verified preprocessing outputs")
    add_data_arguments(preflight)
    preflight.add_argument("--full-array-check", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Build EventSleep1 temporal cache and EventSleep2 descriptors")
    add_data_arguments(prepare)
    prepare.add_argument("--sequence-length", type=int, default=32)
    prepare.add_argument("--history-us", type=int, default=512_000)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--overwrite", action="store_true")

    eda = subparsers.add_parser("eda", help="Create leakage-safe EDA tables and PDF figures")
    add_data_arguments(eda)

    model_check_parser = subparsers.add_parser("model-check", help="Download/check the pretrained ViT and run a forward pass")
    add_data_arguments(model_check_parser)
    add_model_arguments(model_check_parser)
    model_check_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    resnete = subparsers.add_parser(
        "resnet-e",
        help="Reproduce the published EventSleep1 ResNet-E protocol with documented quirks",
    )
    add_data_arguments(resnete)
    resnete.add_argument(
        "--stage",
        choices=["prepare", "check", "train", "all"],
        default="all",
        help="prepare cache, check cache/model, train fixed protocol, or run all stages",
    )
    resnete.add_argument(
        "--profile",
        choices=list(RESNETE_PROFILES),
        default="paper_intended",
        help="paper_intended is the primary fair baseline; authors_code_exact is diagnostic only",
    )
    resnete.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    resnete.add_argument("--cache-workers", type=int, default=2)
    resnete.add_argument("--workers", type=int, default=4)
    resnete.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    resnete.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    resnete.add_argument("--full-cache-check", action="store_true")
    resnete.add_argument("--overwrite-cache", action="store_true")
    resnete.add_argument("--overwrite-runs", action="store_true")

    train = subparsers.add_parser("train", help="Run controlled baselines and EventSleepFormer")
    add_data_arguments(train)
    add_model_arguments(train)
    add_training_arguments(train)
    train.add_argument("--dataset", choices=["eventsleep1", "eventsleep2"], required=True)
    train.add_argument("--task", choices=["activity", "movement", "posture"], required=True)
    train.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    train.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    train.add_argument("--held-out-subjects", nargs="+", type=int, default=[1, 2, 3])
    train.add_argument(
        "--es1-validation-policy",
        choices=list(ES1_VALIDATION_POLICIES),
        default="matched_resnete",
        help=(
            "EventSleep1 only: matched_resnete is the recommended paper protocol (train 1-8,10; "
            "validation 11; test 9,12,13,14). grouped_fold is retained for legacy/secondary experiments."
        ),
    )
    train.add_argument(
        "--validation-fold",
        type=int,
        choices=range(5),
        default=0,
        help="EventSleep1 grouped_fold only; ignored by matched_resnete and EventSleep2",
    )
    train.add_argument("--allow-partial-folds", action="store_true", help="Pilot only; final EventSleep2 results require all three LOSO folds")
    train.add_argument("--locked-config", type=Path, default=None, help="Evaluate an exact best_configuration.json produced by the tune command; model/training/method flags are ignored")

    tune = subparsers.add_parser(
        "tune",
        help="Validation-only EventSleepFormer hyperparameter search; official test data are not instantiated",
    )
    add_data_arguments(tune)
    add_model_arguments(tune)
    add_training_arguments(tune)
    tune.add_argument("--dataset", choices=["eventsleep1", "eventsleep2"], required=True)
    tune.add_argument("--task", choices=["activity", "movement", "posture"], required=True)
    tune.add_argument("--seeds", nargs="+", type=int, default=[13])
    tune.add_argument("--held-out-subjects", nargs="+", type=int, default=[1, 2, 3])
    tune.add_argument(
        "--es1-validation-policy",
        choices=list(ES1_VALIDATION_POLICIES),
        default="matched_resnete",
        help="EventSleep1 tuning should normally use matched_resnete (Subject 11 validation)",
    )
    tune.add_argument("--validation-fold", type=int, choices=range(5), default=0)
    tune.add_argument("--allow-partial-folds", action="store_true")
    tune.add_argument("--losses", nargs="+", choices=["ce", "weighted_ce", "balanced_softmax", "focal"], default=None)
    tune.add_argument("--learning-rates", nargs="+", type=float, default=None)
    tune.add_argument("--backbone-lr-multipliers", nargs="+", type=float, default=None)
    tune.add_argument("--weight-decays", nargs="+", type=float, default=None)
    tune.add_argument("--d-models", nargs="+", type=int, default=None)
    tune.add_argument("--temporal-layers-grid", nargs="+", type=int, default=None)
    tune.add_argument("--temporal-heads-grid", nargs="+", type=int, default=None)
    tune.add_argument("--dropouts", nargs="+", type=float, default=None)
    tune.add_argument("--lora-ranks", nargs="+", type=int, default=None)
    tune.add_argument("--lora-blocks-grid", nargs="+", type=int, default=None)
    tune.add_argument("--label-smoothings", nargs="+", type=float, default=None)
    tune.add_argument("--warmup-epochs-grid", nargs="+", type=int, default=None)
    tune.add_argument("--boundary-weights", nargs="+", type=float, default=None)
    tune.add_argument("--max-configurations", type=int, default=24, help="Hard guard against accidental huge Cartesian searches")
    tune.add_argument("--keep-tuning-checkpoints", action="store_true", help="Retain each candidate best_model.pt; disabled by default to keep the repository compact")

    ablation = subparsers.add_parser("ablation", help="Run predefined EventSleepFormer ablations")
    add_data_arguments(ablation)
    add_model_arguments(ablation)
    add_training_arguments(ablation)
    ablation.add_argument("--dataset", choices=["eventsleep1", "eventsleep2"], required=True)
    ablation.add_argument("--task", choices=["activity", "movement", "posture"], required=True)
    ablation.add_argument("--methods", nargs="+", default=["eventsleepformer@balanced_softmax"])
    ablation.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ablation.add_argument("--held-out-subjects", nargs="+", type=int, default=[1, 2, 3])
    ablation.add_argument(
        "--es1-validation-policy",
        choices=list(ES1_VALIDATION_POLICIES),
        default="matched_resnete",
        help="EventSleep1 validation policy; matched_resnete is recommended for paper comparisons",
    )
    ablation.add_argument(
        "--validation-fold",
        type=int,
        choices=range(5),
        default=0,
        help="EventSleep1 grouped_fold only",
    )
    ablation.add_argument("--allow-partial-folds", action="store_true")
    ablation.add_argument(
        "--variants",
        nargs="+",
        choices=["full", "no_temporal", "no_time", "no_density", "no_boundary", "standard_attention", "frozen_no_lora"],
        default=["no_temporal", "no_time", "no_density", "no_boundary", "standard_attention", "frozen_no_lora"],
    )

    analyze = subparsers.add_parser("analyze", help="Aggregate runs and make paper tables/figures")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--bootstrap-iterations", type=int, default=10_000)
    analyze.add_argument("--verbose", action="store_true")

    smoke = subparsers.add_parser("smoke-test", help="Run synthetic end-to-end validation")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--verbose", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_root = args.output.resolve() / "logs"
    configure_logging(log_root / f"{args.command}.log", bool(getattr(args, "verbose", False)))
    logging.info("EventSleepFormer pipeline v%s", VERSION)
    try:
        if args.command == "smoke-test":
            report = smoke_test(args.output.resolve())
            logging.info("Smoke test PASSED: %s", report)
            return 0
        if args.command == "analyze":
            report = analyze_results(args.output.resolve(), int(args.bootstrap_iterations))
            logging.info("Analysis completed: %s", report)
            return 0

        paths = DataPaths(args.preprocess_root.resolve())
        if args.command == "preflight":
            report = validate_preprocessed(paths, bool(args.full_array_check))
            atomic_json(args.output.resolve() / "preflight.json", report)
            atomic_json(args.output.resolve() / "software_environment.json", software_environment())
            logging.info("Preflight PASSED: %s", report)
        elif args.command == "prepare":
            validate_preprocessed(paths, False)
            es1 = build_es1_sequences(paths, int(args.sequence_length), int(args.history_us), int(args.workers), bool(args.overwrite))
            es2 = build_es2_descriptors(paths, bool(args.overwrite))
            report = {"completed_utc": utc_now(), "eventsleep1_sequences": len(es1), "eventsleep2_recordings": len(es2), "pipeline_version": VERSION}
            atomic_json(args.output.resolve() / "prepare_completed.json", report)
            logging.info("Preparation completed: %s", report)
        elif args.command == "eda":
            validate_preprocessed(paths, False)
            report = run_eda(paths, args.output.resolve() / "eda")
            logging.info("EDA completed: %s", report)
        elif args.command == "model-check":
            report = check_model(args)
            atomic_json(args.output.resolve() / "model_check.json", report)
            logging.info("Model check PASSED: %s", report)
        elif args.command == "resnet-e":
            validate_preprocessed(paths, False)
            report = run_published_resnete_command(args, paths)
            logging.info("Published ResNet-E command completed: %s", report)
        elif args.command == "train":
            validate_preprocessed(paths, False)
            results = run_training_grid(args, paths, variants=["full"])
            logging.info("Training grid completed: %d runs", len(results))
        elif args.command == "tune":
            validate_preprocessed(paths, False)
            summary = run_tuning_grid(args, paths)
            logging.info("Validation-only tuning completed: %d configurations ranked", len(summary))
        elif args.command == "ablation":
            validate_preprocessed(paths, False)
            results = run_training_grid(args, paths, variants=list(args.variants))
            logging.info("Ablation grid completed: %d runs", len(results))
        else:
            parser.error(f"Unsupported command: {args.command}")
        logging.info("Pipeline command completed successfully: %s", args.command)
        return 0
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        logging.debug("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
