#!/usr/bin/env python3
"""Leakage-safe, resumable preprocessing for EventSleep1 and EventSleep2.

The program treats the releases as two independent datasets.  It never writes
inside either source root.  Event streams are converted to compact, two-channel
linear time surfaces (negative polarity first, positive polarity second).

EventSleep1
    One time-surface image is generated for every released labelled NPY clip.
    The official subject-independent TRAIN/TEST allocation is preserved.

EventSleep2
    The interval table is expanded into unique (subject, sequence, frame_index)
    keys.  Conflicting overlap claims are retained in the audit manifest but
    masked from supervised learning.  One time surface is generated for each
    unique frame key belonging to a labelled recording.  The unlabelled
    subject03_seq02 recording is preserved at source but is not fabricated into
    supervised samples.

The original downloads and LabelsDatasetV2.csv are read-only inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import platform
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


VERSION = "1.0.0"
SENSOR_WIDTH = 640
SENSOR_HEIGHT = 480
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ES1_ROOT = REPOSITORY_ROOT / "data" / "raw" / "eventsleep1"
DEFAULT_ES2_ROOT = REPOSITORY_ROOT / "data" / "raw" / "eventsleep2"

LABEL_NAMES = {
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
MOVEMENT_LABELS = set(range(6))
POSTURE_LABELS = set(range(6, 10))
ROLL_LABELS = {2, 3}

ES1_RELEASE_EXPECTED = {
    "train_clips": 736,
    "test_clips": 259,
    "total_clips": 995,
    "full_sequences": 7,
}
ES1_TRAIN_SUBJECTS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11}
ES1_TEST_SUBJECTS = {9, 12, 13, 14}
ES2_RELEASE_EXPECTED = {
    "csv_rows": 1022,
    "recordings": 11,
    "labelled_recordings": 10,
    "unlabelled_recording": "subject03_seq02",
    "ambiguous_unique_frames": 24,
}

ES1_CLIP_RE = re.compile(
    r"(?P<split>TRAIN|TEST)[/\\]"
    r"subject(?P<subject>\d+)_config(?P<configuration>\d+)[/\\]"
    r"(?P<clip>clip\d+)_label(?P<label>-?\d+)\.npy$",
    re.IGNORECASE,
)
ES2_RECORDING_RE = re.compile(
    r"subject(?P<subject>\d+)_seq(?P<sequence>\d+)\.aedat4$",
    re.IGNORECASE,
)
REQUIRED_ES2_COLUMNS = [
    "Subject",
    "Sequence",
    "FrameIni",
    "FrameEnd",
    "Label_mv",
    "Label_st",
    "nFrames",
    "TSInit",
    "TSEnd",
]


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
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def stable_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def configure_logging(output: Path, verbose: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(output / "preprocess.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def path_is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def require_separate_output(output: Path, roots: Iterable[Path]) -> None:
    output_resolved = output.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if (
            output_resolved == root_resolved
            or path_is_within(output_resolved, root_resolved)
            or path_is_within(root_resolved, output_resolved)
        ):
            raise ValueError(
                f"Output {output_resolved} and source dataset {root_resolved} are not separate trees. "
                "Choose a dedicated sibling output directory; source releases are read-only."
            )


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def relative_to(path: Path, parent: Path) -> str:
    return path.resolve().relative_to(parent.resolve()).as_posix()


def software_environment() -> dict[str, Any]:
    try:
        import aedat  # type: ignore

        aedat_version = getattr(aedat, "__version__", "unknown")
        aedat_module = getattr(aedat, "__file__", "unknown")
    except Exception as exc:  # pragma: no cover - environment dependent
        aedat_version = f"unavailable: {type(exc).__name__}: {exc}"
        aedat_module = ""
    return {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "aedat": aedat_version,
        "aedat_module": aedat_module,
    }


@dataclass(frozen=True)
class RepresentationConfig:
    output_height: int = 120
    output_width: int = 160
    history_us: int = 512_000
    roi_profile: str = "full"
    polarity_order: tuple[str, str] = ("negative", "positive")
    dtype: str = "float16"

    def validate(self) -> None:
        if self.output_height < 8 or self.output_width < 8:
            raise ValueError("Output height and width must both be at least 8")
        if self.history_us <= 0:
            raise ValueError("History must be positive")
        if self.roi_profile not in {"full", "official"}:
            raise ValueError("roi_profile must be 'full' or 'official'")
        if self.dtype not in {"float16", "uint8"}:
            raise ValueError("dtype must be 'float16' or 'uint8'")


def roi_for(dataset: str, subject: int, profile: str) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1), with exclusive x1/y1."""
    if profile == "full":
        return (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
    if dataset == "eventsleep1":
        if subject <= 4:
            return (95, 60, 595, 420)
        return (95, 80, 595, 440)
    if dataset == "eventsleep2":
        if subject == 1:
            return (95, 60, 595, 420)
        if subject == 2:
            return (95, 80, 595, 440)
        if subject == 3:
            return (40, 50, 520, 420)
    raise ValueError(f"Official ROI is undefined for dataset={dataset!r}, subject={subject}")


def find_field(dtype_names: Sequence[str] | None, candidates: Sequence[str]) -> str:
    if not dtype_names:
        raise ValueError("Event array is not a structured NumPy array")
    lower = {name.lower(): name for name in dtype_names}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Missing event field; expected one of {list(candidates)}, found {list(dtype_names)}")


def structured_events_to_columns(events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = events.dtype.names
    x_name = find_field(names, ["x"])
    y_name = find_field(names, ["y"])
    t_name = find_field(names, ["t", "timestamp", "ts"])
    p_name = find_field(names, ["p", "on", "polarity"])
    x = np.asarray(events[x_name], dtype=np.int32).reshape(-1)
    y = np.asarray(events[y_name], dtype=np.int32).reshape(-1)
    t = np.asarray(events[t_name], dtype=np.int64).reshape(-1)
    p = np.asarray(events[p_name], dtype=np.int8).reshape(-1)
    return x, y, t, p


def normalize_columns(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (len(x) == len(y) == len(t) == len(p)):
        raise ValueError("Event columns have inconsistent lengths")
    if len(t) == 0:
        return x, y, t, p
    if np.any((p != 0) & (p != 1)):
        raise ValueError(f"Polarity must be binary; found {np.unique(p).tolist()}")
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    return x, y, t, p


def update_latest_surface(
    latest_flat: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    roi: tuple[int, int, int, int],
    output_height: int,
    output_width: int,
) -> int:
    if len(t) == 0:
        return 0
    x0, y0, x1, y1 = roi
    valid = (
        (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
        & (x >= 0)
        & (x < SENSOR_WIDTH)
        & (y >= 0)
        & (y < SENSOR_HEIGHT)
    )
    if not np.any(valid):
        return 0
    xv = x[valid] - x0
    yv = y[valid] - y0
    tv = t[valid]
    pv = p[valid]
    mapped_x = np.minimum((xv * output_width) // (x1 - x0), output_width - 1)
    mapped_y = np.minimum((yv * output_height) // (y1 - y0), output_height - 1)
    flat_pixel = (mapped_y * output_width + mapped_x).astype(np.int64, copy=False)
    indices = pv.astype(np.int64, copy=False) * (output_height * output_width) + flat_pixel
    np.maximum.at(latest_flat, indices, tv)
    return int(valid.sum())


def surface_from_latest(
    latest_flat: np.ndarray,
    anchor_us: int,
    history_us: int,
    storage_dtype: str,
) -> np.ndarray:
    lower = int(anchor_us) - int(history_us)
    delta = np.zeros_like(latest_flat, dtype=np.int64)
    valid = latest_flat > lower
    if np.any(valid):
        delta[valid] = np.minimum(latest_flat[valid], anchor_us) - lower
        delta[valid] = np.clip(delta[valid], 0, history_us)
    if storage_dtype == "uint8":
        return ((delta * 255 + history_us // 2) // history_us).astype(np.uint8)
    return (delta.astype(np.float32) / float(history_us)).astype(np.float16)


def render_event_snapshot(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    anchor_us: int,
    roi: tuple[int, int, int, int],
    config: RepresentationConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    x, y, t, p = normalize_columns(x, y, t, p)
    keep = t <= anchor_us
    x, y, t, p = x[keep], y[keep], t[keep], p[keep]
    latest = np.full(2 * config.output_height * config.output_width, -10**18, dtype=np.int64)
    in_roi = update_latest_surface(
        latest,
        x,
        y,
        t,
        p,
        roi,
        config.output_height,
        config.output_width,
    )
    frame = surface_from_latest(latest, anchor_us, config.history_us, config.dtype)
    frame = frame.reshape(2, config.output_height, config.output_width)
    return frame, {
        "events_through_anchor": int(len(t)),
        "events_in_roi_through_anchor": int(in_roi),
        "active_output_values": int(np.count_nonzero(frame)),
    }


def parse_es1_clip(path: Path, root: Path) -> dict[str, Any]:
    relpath = relative_to(path, root)
    match = ES1_CLIP_RE.search(relpath)
    if not match:
        raise ValueError(f"EventSleep1 path does not match release layout: {relpath}")
    split = match.group("split").upper()
    subject = int(match.group("subject"))
    configuration = int(match.group("configuration"))
    clip_id = match.group("clip").lower()
    label = int(match.group("label"))
    if label not in LABEL_NAMES:
        raise ValueError(f"Invalid EventSleep1 label {label} in {relpath}")
    expected_split = "TRAIN" if subject in ES1_TRAIN_SUBJECTS else "TEST" if subject in ES1_TEST_SUBJECTS else None
    if expected_split is None or split != expected_split:
        raise ValueError(
            f"Subject allocation mismatch for {relpath}: expected {expected_split}, found {split}"
        )
    return {
        "source_relpath": relpath,
        "split": split,
        "subject": subject,
        "configuration": configuration,
        "clip_id": clip_id,
        "label": label,
        "label_name": LABEL_NAMES[label],
    }


def discover_es1_clips(root: Path) -> list[Path]:
    paths = sorted(
        path
        for split in ["TRAIN", "TEST"]
        for path in (root / split).rglob("*.npy")
        if path.is_file()
    )
    return paths


def es1_output_paths(output: Path, parsed: Mapping[str, Any]) -> tuple[Path, Path]:
    stem = Path(str(parsed["source_relpath"])).with_suffix("")
    frame_path = output / "eventsleep1" / "frames" / stem.with_suffix(".npy")
    marker_path = output / "eventsleep1" / "metadata" / stem.with_suffix(".json")
    return frame_path, marker_path


def valid_npy(path: Path, expected_shape: tuple[int, ...], dtype: str = "float16") -> bool:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return tuple(array.shape) == tuple(expected_shape) and str(array.dtype) == dtype
    except Exception:
        return False


def preprocess_es1_clip(task: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(task["source"]))
    root = Path(str(task["root"]))
    output = Path(str(task["output"]))
    config = RepresentationConfig(**task["config"])
    parsed = parse_es1_clip(source, root)
    frame_path, marker_path = es1_output_paths(output, parsed)
    roi = roi_for("eventsleep1", int(parsed["subject"]), config.roi_profile)
    signature = source_signature(source)
    job_digest = stable_digest(
        {
            "pipeline_version": VERSION,
            "dataset": "eventsleep1",
            "source": signature,
            "parsed": parsed,
            "representation": asdict(config),
            "roi": roi,
        }
    )
    expected_shape = (2, config.output_height, config.output_width)
    if marker_path.exists() and frame_path.exists() and not bool(task["overwrite"]):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("job_digest") == job_digest and valid_npy(frame_path, expected_shape, config.dtype):
            marker["resume_status"] = "reused"
            return marker
        raise RuntimeError(
            f"Stale/incompatible output exists for {source}. Use --overwrite or a new output directory."
        )

    events = np.load(source, mmap_mode="r", allow_pickle=False)
    x, y, t, p = structured_events_to_columns(events)
    x, y, t, p = normalize_columns(x, y, t, p)
    if len(t) == 0:
        raise ValueError(f"No events in {source}")
    anchor_us = int(t.max())
    frame, statistics = render_event_snapshot(x, y, t, p, anchor_us, roi, config)
    atomic_npy(frame_path, frame)
    row = {
        "dataset": "EventSleep1",
        "sample_id": (
            f"es1_{parsed['split'].lower()}_subject{parsed['subject']:02d}_"
            f"config{parsed['configuration']}_{parsed['clip_id']}"
        ),
        **parsed,
        "source_path": str(source.resolve()),
        "frame_path": relative_to(frame_path, output),
        "frame_shape": "x".join(map(str, expected_shape)),
        "frame_dtype": config.dtype,
        "anchor_timestamp_us": anchor_us,
        "first_timestamp_us": int(t.min()),
        "duration_us": int(t.max() - t.min()),
        "raw_event_count": int(len(t)),
        "events_in_roi": int(statistics["events_in_roi_through_anchor"]),
        "active_output_values": int(statistics["active_output_values"]),
        "roi_x0": roi[0],
        "roi_y0": roi[1],
        "roi_x1": roi[2],
        "roi_y1": roi[3],
    }
    marker = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "job_digest": job_digest,
        "source_signature": signature,
        "representation": asdict(config),
        "row": row,
        "resume_status": "created",
    }
    atomic_json(marker_path, marker)
    return marker


def deterministic_subject_folds() -> pd.DataFrame:
    rng = np.random.default_rng(20260828)
    train_subjects = np.array(sorted(ES1_TRAIN_SUBJECTS), dtype=int)
    rng.shuffle(train_subjects)
    fold_map = {int(subject): int(index % 5) for index, subject in enumerate(train_subjects)}
    rows = []
    for subject in sorted(ES1_TRAIN_SUBJECTS | ES1_TEST_SUBJECTS):
        rows.append(
            {
                "subject": subject,
                "official_split": "TRAIN" if subject in ES1_TRAIN_SUBJECTS else "TEST",
                "validation_fold_within_official_train": fold_map.get(subject, -1),
                "note": (
                    "Use this fold only within official TRAIN"
                    if subject in ES1_TRAIN_SUBJECTS
                    else "Untouched official TEST subject"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_es1_manifests(output: Path, markers: Sequence[Mapping[str, Any]], partial: bool) -> pd.DataFrame:
    rows = [dict(marker["row"]) for marker in markers]
    frame = pd.DataFrame(rows).sort_values(
        ["split", "subject", "configuration", "clip_id"], kind="stable"
    )
    manifest_dir = output / "eventsleep1" / "manifests"
    atomic_csv(manifest_dir / "all_samples.csv", frame)
    atomic_csv(manifest_dir / "train_samples.csv", frame[frame["split"] == "TRAIN"].copy())
    atomic_csv(manifest_dir / "test_samples.csv", frame[frame["split"] == "TEST"].copy())
    distribution = (
        frame.groupby(["split", "label", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    atomic_csv(manifest_dir / "class_distribution.csv", distribution)
    subject_counts = (
        frame.groupby(["split", "subject", "configuration"], as_index=False)
        .size()
        .rename(columns={"size": "samples"})
    )
    atomic_csv(manifest_dir / "subject_configuration_counts.csv", subject_counts)
    atomic_csv(manifest_dir / "subject_folds.csv", deterministic_subject_folds())
    atomic_json(
        output / "eventsleep1" / "preprocessing_completed.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "partial": partial,
            "samples": int(len(frame)),
            "train_samples": int((frame["split"] == "TRAIN").sum()),
            "test_samples": int((frame["split"] == "TEST").sum()),
        },
    )
    return frame


def locate_es2_labels(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"EventSleep2 label CSV does not exist: {explicit}")
        return explicit
    candidates = sorted(root.rglob("LabelsDatasetV2.csv"), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f"LabelsDatasetV2.csv not found below {root}")
    return candidates[0]


def load_es2_labels(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = [column for column in REQUIRED_ES2_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing EventSleep2 label columns: {missing}")
    labels = raw[REQUIRED_ES2_COLUMNS].copy()
    for column in REQUIRED_ES2_COLUMNS:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    mandatory = [column for column in REQUIRED_ES2_COLUMNS if column != "Label_st"]
    bad = labels[mandatory].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} EventSleep2 rows have missing/non-numeric mandatory values")
    integer_columns = [
        "Subject",
        "Sequence",
        "FrameIni",
        "FrameEnd",
        "Label_mv",
        "nFrames",
        "TSInit",
        "TSEnd",
    ]
    for column in integer_columns:
        labels[column] = labels[column].round().astype(np.int64)
    calculated = labels["FrameEnd"] - labels["FrameIni"] + 1
    if not np.array_equal(calculated.to_numpy(), labels["nFrames"].to_numpy()):
        mismatches = int((calculated != labels["nFrames"]).sum())
        raise ValueError(f"{mismatches} rows violate nFrames = FrameEnd - FrameIni + 1")
    labels.insert(0, "source_row", np.arange(len(labels), dtype=np.int64))
    return labels


def _joined_ints(values: Iterable[int]) -> str:
    return ";".join(str(value) for value in sorted(set(int(v) for v in values)))


def expand_es2_unique_frames(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    claims: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labels.itertuples(index=False):
        n_frames = int(row.nFrames)
        if n_frames == 1:
            anchors = np.array([int(round((int(row.TSInit) + int(row.TSEnd)) / 2))], dtype=np.int64)
        else:
            anchors = np.rint(
                np.linspace(int(row.TSInit), int(row.TSEnd), n_frames, dtype=np.float64)
            ).astype(np.int64)
        for offset, frame_index in enumerate(range(int(row.FrameIni), int(row.FrameEnd) + 1)):
            claims[(int(row.Subject), int(row.Sequence), frame_index)].append(
                {
                    "movement": int(row.Label_mv),
                    "posture": None if pd.isna(row.Label_st) else int(round(float(row.Label_st))),
                    "anchor": int(anchors[offset]),
                    "source_row": int(row.source_row),
                }
            )

    rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for (subject, sequence, frame_index), frame_claims in sorted(claims.items()):
        movement_claims = sorted({int(item["movement"]) for item in frame_claims})
        posture_claims = sorted(
            {int(item["posture"]) for item in frame_claims if item["posture"] is not None}
        )
        anchors = [int(item["anchor"]) for item in frame_claims]
        source_rows = [int(item["source_row"]) for item in frame_claims]
        movement_conflict = len(movement_claims) != 1
        posture_conflict = len(posture_claims) > 1
        movement_label = movement_claims[0] if len(movement_claims) == 1 else None
        posture_label = posture_claims[0] if len(posture_claims) == 1 else None
        movement_valid = (
            not movement_conflict
            and movement_label is not None
            and movement_label in MOVEMENT_LABELS
        )
        movement_reason = ""
        if movement_conflict:
            movement_reason = "conflicting_overlapping_movement_labels"
        elif movement_label == -1:
            movement_reason = "released_unlabelled_movement"
        elif movement_label not in MOVEMENT_LABELS:
            movement_reason = "invalid_movement_label"

        posture_expected = movement_valid and movement_label in ROLL_LABELS
        posture_valid = (
            posture_expected
            and not posture_conflict
            and posture_label is not None
            and posture_label in POSTURE_LABELS
        )
        posture_reason = ""
        if movement_conflict:
            posture_reason = "movement_label_conflict"
        elif not posture_expected:
            posture_reason = "posture_not_applicable"
        elif posture_conflict:
            posture_reason = "conflicting_overlapping_posture_labels"
        elif posture_label is None:
            posture_reason = "missing_posture_for_roll"
        elif posture_label not in POSTURE_LABELS:
            posture_reason = "invalid_posture_label"

        anchor = int(np.rint(np.median(np.asarray(anchors, dtype=np.float64))))
        row = {
            "dataset": "EventSleep2",
            "frame_key": f"subject{subject:02d}_seq{sequence:02d}_frame{frame_index:06d}",
            "recording_key": f"subject{subject:02d}_seq{sequence:02d}",
            "subject": subject,
            "sequence": sequence,
            "frame_index": frame_index,
            "anchor_timestamp_us": anchor,
            "timestamp_claim_spread_us": int(max(anchors) - min(anchors)),
            "claim_count": len(frame_claims),
            "source_row_indices": _joined_ints(source_rows),
            "source_csv_line_numbers": _joined_ints(row_index + 2 for row_index in source_rows),
            "movement_claims": _joined_ints(movement_claims),
            "movement_label": movement_label,
            "movement_label_name": "" if movement_label not in LABEL_NAMES else LABEL_NAMES[movement_label],
            "movement_conflict": movement_conflict,
            "supervised_movement": movement_valid,
            "movement_exclusion_reason": movement_reason,
            "posture_claims": _joined_ints(posture_claims),
            "posture_label": posture_label,
            "posture_label_name": "" if posture_label not in LABEL_NAMES else LABEL_NAMES[posture_label],
            "posture_expected": posture_expected,
            "posture_conflict": posture_conflict,
            "supervised_posture": posture_valid,
            "posture_exclusion_reason": posture_reason,
        }
        rows.append(row)
        if movement_conflict or posture_conflict:
            conflict_rows.append(row.copy())

    frame = pd.DataFrame(rows)
    for column in ["movement_label", "posture_label"]:
        frame[column] = pd.array(frame[column], dtype="Int64")
    conflicts = pd.DataFrame(conflict_rows, columns=frame.columns)
    return frame, conflicts


def discover_es2_recordings(root: Path) -> dict[str, Path]:
    recordings: dict[str, Path] = {}
    for path in sorted(root.rglob("*.aedat4")):
        match = ES2_RECORDING_RE.search(path.name)
        if not match:
            continue
        key = f"subject{int(match.group('subject')):02d}_seq{int(match.group('sequence')):02d}"
        if key in recordings:
            raise ValueError(f"Duplicate AEDAT4 recording key {key}: {recordings[key]} and {path}")
        recordings[key] = path
    return recordings


def aedat_event_packets(path: Path) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        import aedat  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Package 'aedat' is required; install aedat==2.2.0") from exc
    decoder = aedat.Decoder(str(path))
    for packet in decoder:
        if "events" not in packet:
            continue
        events = packet["events"]
        if len(events) == 0:
            continue
        yield normalize_columns(*structured_events_to_columns(events))


def render_target_sequence(
    packets: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    target_times: np.ndarray,
    roi: tuple[int, int, int, int],
    config: RepresentationConfig,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    target_times = np.asarray(target_times, dtype=np.int64)
    if len(target_times) == 0:
        empty = np.empty(
            (0, 2, config.output_height, config.output_width), dtype=np.dtype(config.dtype)
        )
        return empty, pd.DataFrame(), {}
    if np.any(np.diff(target_times) < 0):
        raise ValueError("Target timestamps must be non-decreasing")

    latest = np.full(2 * config.output_height * config.output_width, -10**18, dtype=np.int64)
    frames = np.empty(
        (len(target_times), 2, config.output_height, config.output_width),
        dtype=np.dtype(config.dtype),
    )
    statistics: list[dict[str, Any]] = []
    target_index = 0
    cumulative_raw = 0
    cumulative_roi = 0
    previous_raw = 0
    previous_roi = 0
    processed_min: int | None = None
    processed_max: int | None = None
    observed_min: int | None = None
    observed_max: int | None = None
    prior_packet_max: int | None = None

    def apply_slice(
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> None:
        nonlocal cumulative_raw, cumulative_roi, processed_min, processed_max
        if len(t) == 0:
            return
        cumulative_raw += int(len(t))
        cumulative_roi += update_latest_surface(
            latest, x, y, t, p, roi, config.output_height, config.output_width
        )
        local_min, local_max = int(t.min()), int(t.max())
        processed_min = local_min if processed_min is None else min(processed_min, local_min)
        processed_max = local_max if processed_max is None else max(processed_max, local_max)

    def emit_target() -> None:
        nonlocal target_index, previous_raw, previous_roi
        anchor = int(target_times[target_index])
        frame = surface_from_latest(latest, anchor, config.history_us, config.dtype)
        frames[target_index] = frame.reshape(2, config.output_height, config.output_width)
        statistics.append(
            {
                "array_index": target_index,
                "events_since_previous_anchor": cumulative_raw - previous_raw,
                "roi_events_since_previous_anchor": cumulative_roi - previous_roi,
                "cumulative_events": cumulative_raw,
                "cumulative_roi_events": cumulative_roi,
                "active_output_values": int(np.count_nonzero(frames[target_index])),
            }
        )
        previous_raw = cumulative_raw
        previous_roi = cumulative_roi
        target_index += 1

    for x, y, t, p in packets:
        if len(t) == 0:
            continue
        packet_min, packet_max = int(t.min()), int(t.max())
        observed_min = packet_min if observed_min is None else min(observed_min, packet_min)
        observed_max = packet_max if observed_max is None else max(observed_max, packet_max)
        if prior_packet_max is not None and packet_min < prior_packet_max:
            raise ValueError(
                f"AEDAT packets are not timestamp-monotonic: {packet_min} < {prior_packet_max}"
            )
        prior_packet_max = packet_max
        start = 0
        while target_index < len(target_times) and int(target_times[target_index]) <= packet_max:
            cut = int(np.searchsorted(t, int(target_times[target_index]), side="right"))
            apply_slice(x[start:cut], y[start:cut], t[start:cut], p[start:cut])
            start = cut
            emit_target()
        if target_index == len(target_times):
            break
        apply_slice(x[start:], y[start:], t[start:], p[start:])

    while target_index < len(target_times):
        emit_target()

    stats_frame = pd.DataFrame(statistics)
    if observed_min is None or observed_max is None:
        stats_frame["anchor_within_observed_event_range"] = False
    else:
        stats_frame["anchor_within_observed_event_range"] = (
            (target_times >= observed_min) & (target_times <= observed_max)
        )
    summary = {
        "recording_first_observed_timestamp_us": observed_min,
        "recording_last_observed_timestamp_us": observed_max,
        "recording_first_processed_timestamp_us": processed_min,
        "recording_last_processed_timestamp_us": processed_max,
        "events_processed_through_last_target": cumulative_raw,
        "roi_events_processed_through_last_target": cumulative_roi,
        "targets": int(len(target_times)),
        "targets_outside_observed_event_range": int(
            (~stats_frame["anchor_within_observed_event_range"]).sum()
        ),
    }
    return frames, stats_frame, summary


def preprocess_es2_recording(task: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(task["source"]))
    output = Path(str(task["output"]))
    recording_key = str(task["recording_key"])
    subject = int(task["subject"])
    sequence = int(task["sequence"])
    config = RepresentationConfig(**task["config"])
    frame_table = pd.DataFrame(task["frame_records"])
    frame_table = frame_table.sort_values(
        ["anchor_timestamp_us", "frame_index"], kind="stable"
    ).reset_index(drop=True)
    roi = roi_for("eventsleep2", subject, config.roi_profile)
    frame_path = output / "eventsleep2" / "frames" / f"{recording_key}.npy"
    metadata_path = output / "eventsleep2" / "metadata" / f"{recording_key}.csv"
    marker_path = output / "eventsleep2" / "metadata" / f"{recording_key}.json"
    signature = source_signature(source)
    label_digest = stable_digest(
        frame_table[
            [
                "frame_key",
                "frame_index",
                "anchor_timestamp_us",
                "movement_claims",
                "posture_claims",
                "supervised_movement",
                "supervised_posture",
            ]
        ].to_dict(orient="records")
    )
    job_digest = stable_digest(
        {
            "pipeline_version": VERSION,
            "dataset": "eventsleep2",
            "source": signature,
            "recording_key": recording_key,
            "representation": asdict(config),
            "roi": roi,
            "label_digest": label_digest,
        }
    )
    expected_shape = (
        len(frame_table),
        2,
        config.output_height,
        config.output_width,
    )
    if marker_path.exists() and metadata_path.exists() and frame_path.exists() and not bool(task["overwrite"]):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("job_digest") == job_digest and valid_npy(frame_path, expected_shape, config.dtype):
            marker["resume_status"] = "reused"
            return marker
        raise RuntimeError(
            f"Stale/incompatible output exists for {recording_key}. Use --overwrite or a new output directory."
        )

    target_times = frame_table["anchor_timestamp_us"].to_numpy(dtype=np.int64)
    frames, stats, stream_summary = render_target_sequence(
        aedat_event_packets(source), target_times, roi, config
    )
    atomic_npy(frame_path, frames)
    enriched = pd.concat([frame_table.reset_index(drop=True), stats], axis=1)
    enriched["source_path"] = str(source.resolve())
    enriched["source_relpath"] = source.name
    enriched["frame_path"] = relative_to(frame_path, output)
    enriched["frame_dtype"] = config.dtype
    enriched["roi_x0"] = roi[0]
    enriched["roi_y0"] = roi[1]
    enriched["roi_x1"] = roi[2]
    enriched["roi_y1"] = roi[3]
    atomic_csv(metadata_path, enriched)
    marker = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "job_digest": job_digest,
        "label_digest": label_digest,
        "recording_key": recording_key,
        "subject": subject,
        "sequence": sequence,
        "source_signature": signature,
        "representation": asdict(config),
        "frame_path": relative_to(frame_path, output),
        "metadata_path": relative_to(metadata_path, output),
        "array_shape": list(frames.shape),
        "stream_summary": stream_summary,
        "resume_status": "created",
    }
    atomic_json(marker_path, marker)
    return marker


def write_es2_manifests(
    output: Path,
    unique_frames: pd.DataFrame,
    conflicts: pd.DataFrame,
    recordings: Mapping[str, Path],
    markers: Sequence[Mapping[str, Any]],
    partial: bool,
) -> pd.DataFrame:
    manifest_dir = output / "eventsleep2" / "manifests"
    atomic_csv(manifest_dir / "cleaned_unique_frame_labels.csv", unique_frames)
    atomic_csv(manifest_dir / "label_conflicts.csv", conflicts)
    processed_frames: list[pd.DataFrame] = []
    marker_keys: set[str] = set()
    summary_rows: list[dict[str, Any]] = []
    for marker in markers:
        key = str(marker["recording_key"])
        marker_keys.add(key)
        metadata_path = output / str(marker["metadata_path"])
        processed_frames.append(pd.read_csv(metadata_path))
        summary_rows.append(
            {
                "recording_key": key,
                "subject": int(marker["subject"]),
                "sequence": int(marker["sequence"]),
                "source_path": marker["source_signature"]["path"],
                "source_size_bytes": marker["source_signature"]["size_bytes"],
                "labelled": True,
                "processed": True,
                "frames": marker["stream_summary"]["targets"],
                "targets_outside_observed_event_range": marker["stream_summary"][
                    "targets_outside_observed_event_range"
                ],
            }
        )
    labelled_keys = set(unique_frames["recording_key"].unique())
    for key, source in sorted(recordings.items()):
        if key in marker_keys:
            continue
        match = ES2_RECORDING_RE.search(source.name)
        summary_rows.append(
            {
                "recording_key": key,
                "subject": int(match.group("subject")) if match else -1,
                "sequence": int(match.group("sequence")) if match else -1,
                "source_path": str(source.resolve()),
                "source_size_bytes": source.stat().st_size,
                "labelled": key in labelled_keys,
                "processed": False,
                "frames": 0,
                "targets_outside_observed_event_range": 0,
            }
        )

    if processed_frames:
        frame = pd.concat(processed_frames, ignore_index=True)
        frame = frame.sort_values(
            ["subject", "sequence", "frame_index"], kind="stable"
        ).reset_index(drop=True)
    else:
        frame = pd.DataFrame()
    atomic_csv(manifest_dir / "all_frames.csv", frame)
    if len(frame):
        movement = frame[frame["supervised_movement"].astype(bool)].copy()
        posture = frame[frame["supervised_posture"].astype(bool)].copy()
        excluded = frame[
            ~frame["supervised_movement"].astype(bool)
            | (frame["posture_expected"].astype(bool) & ~frame["supervised_posture"].astype(bool))
        ].copy()
    else:
        movement = posture = excluded = frame.copy()
    atomic_csv(manifest_dir / "movement_supervised.csv", movement)
    atomic_csv(manifest_dir / "posture_supervised.csv", posture)
    atomic_csv(manifest_dir / "excluded_or_masked_frames.csv", excluded)
    atomic_csv(manifest_dir / "recording_summary.csv", pd.DataFrame(summary_rows))
    if len(movement):
        movement_distribution = (
            movement.groupby(["movement_label", "movement_label_name"], as_index=False)
            .size()
            .rename(columns={"size": "frames"})
        )
    else:
        movement_distribution = pd.DataFrame(
            columns=["movement_label", "movement_label_name", "frames"]
        )
    if len(posture):
        posture_distribution = (
            posture.groupby(["posture_label", "posture_label_name"], as_index=False)
            .size()
            .rename(columns={"size": "frames"})
        )
    else:
        posture_distribution = pd.DataFrame(
            columns=["posture_label", "posture_label_name", "frames"]
        )
    atomic_csv(manifest_dir / "movement_class_distribution.csv", movement_distribution)
    atomic_csv(manifest_dir / "posture_class_distribution.csv", posture_distribution)

    loso_rows = []
    labelled_subjects = sorted(int(value) for value in unique_frames["subject"].unique())
    for held_out in labelled_subjects:
        for subject in labelled_subjects:
            loso_rows.append(
                {
                    "fold": f"held_out_subject{held_out:02d}",
                    "subject": subject,
                    "role": "test" if subject == held_out else "train_pool",
                    "note": (
                        "Never use held-out subject for tuning"
                        if subject == held_out
                        else "Tune with fixed settings or sequence-grouped source-only validation"
                    ),
                }
            )
    atomic_csv(manifest_dir / "loso_subject_folds.csv", pd.DataFrame(loso_rows))
    atomic_json(
        output / "eventsleep2" / "preprocessing_completed.json",
        {
            "completed_utc": utc_now(),
            "pipeline_version": VERSION,
            "partial": partial,
            "recordings_processed": len(markers),
            "unique_label_frame_keys": int(len(unique_frames)),
            "generated_frames": int(len(frame)),
            "movement_supervised_frames": int(len(movement)),
            "posture_supervised_frames": int(len(posture)),
            "ambiguous_movement_frames_masked": int(unique_frames["movement_conflict"].sum()),
            "unlabelled_recordings_not_processed": sorted(set(recordings) - labelled_keys),
        },
    )
    return frame


def run_tasks(
    worker: Any,
    tasks: Sequence[Mapping[str, Any]],
    workers: int,
    progress_label: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            results.append(worker(task))
            if index % progress_every == 0 or index == len(tasks):
                logging.info("%s: %d/%d", progress_label, index, len(tasks))
        return results
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker, task): task for task in tasks}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                task = future_map[future]
                raise RuntimeError(f"{progress_label} failed for {task.get('source')}: {exc}") from exc
            if index % progress_every == 0 or index == len(tasks):
                logging.info("%s: %d/%d", progress_label, index, len(tasks))
    return results


def run_eventsleep1(
    root: Path,
    output: Path,
    config: RepresentationConfig,
    workers: int,
    overwrite: bool,
    max_clips: int | None,
) -> pd.DataFrame:
    clips = discover_es1_clips(root)
    partial = max_clips is not None and max_clips < len(clips)
    if max_clips is not None:
        clips = clips[:max_clips]
    tasks = [
        {
            "source": str(path),
            "root": str(root),
            "output": str(output),
            "config": asdict(config),
            "overwrite": overwrite,
        }
        for path in clips
    ]
    logging.info("EventSleep1: converting %d labelled clips with %d worker(s)", len(tasks), workers)
    markers = run_tasks(preprocess_es1_clip, tasks, workers, "EventSleep1 clips", 25)
    frame = write_es1_manifests(output, markers, partial)
    logging.info("EventSleep1 preprocessing completed: %d samples", len(frame))
    return frame


def run_eventsleep2(
    root: Path,
    labels_path: Path,
    output: Path,
    config: RepresentationConfig,
    workers: int,
    overwrite: bool,
    max_recordings: int | None,
) -> pd.DataFrame:
    labels = load_es2_labels(labels_path)
    unique_frames, conflicts = expand_es2_unique_frames(labels)
    recordings = discover_es2_recordings(root)
    labelled_keys = sorted(set(unique_frames["recording_key"]) & set(recordings))
    missing_recordings = sorted(set(unique_frames["recording_key"]) - set(recordings))
    if missing_recordings:
        raise FileNotFoundError(f"Labelled EventSleep2 recordings are missing: {missing_recordings}")
    partial = max_recordings is not None and max_recordings < len(labelled_keys)
    if max_recordings is not None:
        labelled_keys = labelled_keys[:max_recordings]
    tasks: list[dict[str, Any]] = []
    for key in labelled_keys:
        match = ES2_RECORDING_RE.search(recordings[key].name)
        subject, sequence = int(match.group("subject")), int(match.group("sequence"))
        subset = unique_frames[unique_frames["recording_key"] == key].copy()
        tasks.append(
            {
                "source": str(recordings[key]),
                "output": str(output),
                "recording_key": key,
                "subject": subject,
                "sequence": sequence,
                "frame_records": subset.to_dict(orient="records"),
                "config": asdict(config),
                "overwrite": overwrite,
            }
        )
    effective_workers = max(1, min(workers, 2, len(tasks)))
    if workers > effective_workers:
        logging.info(
            "EventSleep2 decoder concurrency capped at %d to avoid shared-disk contention",
            effective_workers,
        )
    logging.info(
        "EventSleep2: converting %d labelled recordings (%d unique frame keys)",
        len(tasks),
        len(unique_frames),
    )
    markers = run_tasks(
        preprocess_es2_recording,
        tasks,
        effective_workers,
        "EventSleep2 recordings",
        1,
    )
    frame = write_es2_manifests(
        output, unique_frames, conflicts, recordings, markers, partial
    )
    logging.info("EventSleep2 preprocessing completed: %d generated frames", len(frame))
    return frame


def preflight(
    es1_root: Path,
    es2_root: Path,
    labels_path: Path,
    allow_release_drift: bool,
) -> dict[str, Any]:
    release_errors: list[str] = []
    fatal_errors: list[str] = []
    warnings: list[str] = []
    clips = discover_es1_clips(es1_root)
    parsed = [parse_es1_clip(path, es1_root) for path in clips]
    train_count = sum(row["split"] == "TRAIN" for row in parsed)
    test_count = sum(row["split"] == "TEST" for row in parsed)
    es1_full = sorted((es1_root / "TEST_FULL_SEQUENCE").rglob("*.aedat4"))
    labels = load_es2_labels(labels_path)
    unique_frames, conflicts = expand_es2_unique_frames(labels)
    recordings = discover_es2_recordings(es2_root)
    labelled_keys = sorted(unique_frames["recording_key"].unique())
    unlabelled_keys = sorted(set(recordings) - set(labelled_keys))
    missing_labelled_recordings = sorted(set(labelled_keys) - set(recordings))
    if missing_labelled_recordings:
        fatal_errors.append(
            f"Label table references missing AEDAT4 recordings: {missing_labelled_recordings}"
        )
    actual = {
        "eventsleep1": {
            "train_clips": train_count,
            "test_clips": test_count,
            "total_clips": len(clips),
            "full_sequences": len(es1_full),
        },
        "eventsleep2": {
            "csv_rows": len(labels),
            "recordings": len(recordings),
            "labelled_recordings": len(labelled_keys),
            "unlabelled_recordings": unlabelled_keys,
            "unique_frame_keys": len(unique_frames),
            "ambiguous_unique_frames": int(unique_frames["movement_conflict"].sum()),
            "conflict_manifest_rows": len(conflicts),
        },
    }
    for key, expected in ES1_RELEASE_EXPECTED.items():
        if actual["eventsleep1"][key] != expected:
            release_errors.append(
                f"EventSleep1 {key}: expected current release {expected}, found {actual['eventsleep1'][key]}"
            )
    checks = {
        "csv_rows": ES2_RELEASE_EXPECTED["csv_rows"],
        "recordings": ES2_RELEASE_EXPECTED["recordings"],
        "labelled_recordings": ES2_RELEASE_EXPECTED["labelled_recordings"],
        "ambiguous_unique_frames": ES2_RELEASE_EXPECTED["ambiguous_unique_frames"],
    }
    for key, expected in checks.items():
        if actual["eventsleep2"][key] != expected:
            release_errors.append(
                f"EventSleep2 {key}: expected current release {expected}, found {actual['eventsleep2'][key]}"
            )
    if unlabelled_keys != [ES2_RELEASE_EXPECTED["unlabelled_recording"]]:
        release_errors.append(
            f"EventSleep2 unlabelled recordings: expected {[ES2_RELEASE_EXPECTED['unlabelled_recording']]}, "
            f"found {unlabelled_keys}"
        )
    try:
        import aedat  # noqa: F401  # type: ignore
    except Exception as exc:
        fatal_errors.append(f"AEDAT decoder is unavailable: {type(exc).__name__}: {exc}")
    if allow_release_drift and release_errors:
        warnings.extend(release_errors)
        release_errors = []
    errors = fatal_errors + release_errors
    report = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "source_roots": {
            "eventsleep1": str(es1_root.resolve()),
            "eventsleep2": str(es2_root.resolve()),
            "eventsleep2_labels": str(labels_path.resolve()),
        },
        "actual": actual,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return report


def verify_preprocessed(output: Path, require_both: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "verified_utc": utc_now(),
        "pipeline_version": VERSION,
        "output": str(output.resolve()),
    }
    configuration_path = output / "run_configuration.json"
    if not configuration_path.exists():
        errors.append("run_configuration.json is missing")
        config = RepresentationConfig()
    else:
        run_config = json.loads(configuration_path.read_text(encoding="utf-8"))
        config = RepresentationConfig(**run_config["representation"])

    es1_manifest_path = output / "eventsleep1" / "manifests" / "all_samples.csv"
    if es1_manifest_path.exists():
        es1 = pd.read_csv(es1_manifest_path)
        if es1["sample_id"].duplicated().any():
            errors.append("EventSleep1 sample_id is not unique")
        train_subjects = set(es1.loc[es1["split"] == "TRAIN", "subject"].astype(int))
        test_subjects = set(es1.loc[es1["split"] == "TEST", "subject"].astype(int))
        if train_subjects & test_subjects:
            errors.append(f"EventSleep1 train/test subject leakage: {sorted(train_subjects & test_subjects)}")
        invalid_arrays = []
        expected_shape = (2, config.output_height, config.output_width)
        for relpath in es1["frame_path"]:
            if not valid_npy(output / relpath, expected_shape, config.dtype):
                invalid_arrays.append(relpath)
        if invalid_arrays:
            errors.append(f"EventSleep1 has {len(invalid_arrays)} invalid/missing arrays")
        report["eventsleep1"] = {
            "samples": len(es1),
            "train": int((es1["split"] == "TRAIN").sum()),
            "test": int((es1["split"] == "TEST").sum()),
            "invalid_arrays": len(invalid_arrays),
            "subject_overlap": sorted(train_subjects & test_subjects),
        }
    elif require_both:
        errors.append("EventSleep1 manifest is missing")

    es2_manifest_path = output / "eventsleep2" / "manifests" / "all_frames.csv"
    es2_clean_path = output / "eventsleep2" / "manifests" / "cleaned_unique_frame_labels.csv"
    if es2_manifest_path.exists() and es2_clean_path.exists():
        es2 = pd.read_csv(es2_manifest_path)
        clean = pd.read_csv(es2_clean_path)
        key_columns = ["subject", "sequence", "frame_index"]
        if es2.duplicated(key_columns).any():
            errors.append("EventSleep2 generated manifest has duplicate physical frame keys")
        ambiguous = int(clean["movement_conflict"].astype(bool).sum())
        ambiguous_supervised = es2[
            es2["movement_conflict"].astype(bool) & es2["supervised_movement"].astype(bool)
        ]
        if len(ambiguous_supervised):
            errors.append("Ambiguous EventSleep2 frames entered the supervised movement manifest")
        invalid_arrays = []
        for relpath, group in es2.groupby("frame_path"):
            expected_shape = (
                len(group),
                2,
                config.output_height,
                config.output_width,
            )
            # Array indices are timestamp ordered and must form 0..N-1.
            indices = sorted(group["array_index"].astype(int).tolist())
            if indices != list(range(len(group))):
                errors.append(f"Non-contiguous array indices for {relpath}")
            if not valid_npy(output / relpath, expected_shape, config.dtype):
                invalid_arrays.append(relpath)
        outside = int((~es2["anchor_within_observed_event_range"].astype(bool)).sum())
        if outside:
            errors.append(f"{outside} EventSleep2 anchors fall outside decoded event ranges")
        report["eventsleep2"] = {
            "generated_frames": len(es2),
            "movement_supervised": int(es2["supervised_movement"].astype(bool).sum()),
            "posture_supervised": int(es2["supervised_posture"].astype(bool).sum()),
            "ambiguous_frames_masked": ambiguous,
            "invalid_arrays": len(invalid_arrays),
            "anchors_outside_event_range": outside,
        }
        if invalid_arrays:
            errors.append(f"EventSleep2 has {len(invalid_arrays)} invalid/missing arrays")
    elif require_both:
        errors.append("EventSleep2 manifests are missing")

    report["errors"] = errors
    report["warnings"] = warnings
    report["passed"] = not errors
    atomic_json(output / "verification.json", report)
    if errors:
        raise RuntimeError("Verification failed:\n- " + "\n- ".join(errors))
    if require_both:
        atomic_json(
            output / "completed_successfully.json",
            {
                "completed_utc": utc_now(),
                "pipeline_version": VERSION,
                "verification": "PASSED",
                "datasets_kept_separate": True,
            },
        )
    return report


def write_run_configuration(
    output: Path,
    args: argparse.Namespace,
    config: RepresentationConfig,
) -> None:
    payload = {
        "created_utc": utc_now(),
        "pipeline_version": VERSION,
        "command": args.command,
        "representation": asdict(config),
        "policy": {
            "datasets_are_independent": True,
            "cross_dataset_training_or_testing": False,
            "eventsleep1_split": "official subject-independent TRAIN/TEST",
            "eventsleep2_split": "leave-one-subject-out",
            "eventsleep2_ambiguous_frame_policy": "retain representation; mask supervised targets",
            "eventsleep2_unlabelled_recording_policy": "preserve source; exclude supervised preprocessing",
        },
        "arguments": vars(args),
    }
    atomic_json(output / "run_configuration.json", payload)
    atomic_json(output / "software_environment.json", software_environment())


def smoke_test(output: Path) -> dict[str, Any]:
    existing = [] if not output.exists() else [path for path in output.iterdir() if path.name != "preprocess.log"]
    if existing:
        raise ValueError(f"Smoke-test output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "synthetic_sources"
    es1_root = sources / "EventSleep1"
    clip_dir = es1_root / "TRAIN" / "subject01_config1"
    clip_dir.mkdir(parents=True)
    dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<u8"), ("on", "u1")])
    events = np.zeros(8, dtype=dtype)
    events["x"] = [0, 4, 8, 12, 16, 20, 24, 28]
    events["y"] = [0, 4, 8, 12, 16, 20, 24, 28]
    events["t"] = np.arange(8, dtype=np.uint64) * 50_000 + 1_000_000
    events["on"] = np.arange(8) % 2
    source_clip = clip_dir / "clip01_label2.npy"
    atomic_npy(source_clip, events)
    source_hash_before = hashlib.sha256(source_clip.read_bytes()).hexdigest()

    config = RepresentationConfig(output_height=12, output_width=16)
    marker = preprocess_es1_clip(
        {
            "source": str(source_clip),
            "root": str(es1_root),
            "output": str(output / "processed"),
            "config": asdict(config),
            "overwrite": False,
        }
    )
    frame_path = output / "processed" / marker["row"]["frame_path"]
    frame = np.load(frame_path, allow_pickle=False)
    assert frame.shape == (2, 12, 16)
    assert frame.dtype == np.float16
    assert np.count_nonzero(frame) > 0
    assert float(frame.min()) >= 0.0 and float(frame.max()) <= 1.0
    compact_config = RepresentationConfig(output_height=12, output_width=16, dtype="uint8")
    compact_frame, _ = render_event_snapshot(
        *structured_events_to_columns(events),
        int(events["t"].max()),
        (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT),
        compact_config,
    )
    assert compact_frame.shape == (2, 12, 16) and compact_frame.dtype == np.uint8
    source_hash_after = hashlib.sha256(source_clip.read_bytes()).hexdigest()
    assert source_hash_before == source_hash_after

    synthetic_labels = pd.DataFrame(
        [
            {
                "source_row": 0,
                "Subject": 1,
                "Sequence": 1,
                "FrameIni": 10,
                "FrameEnd": 12,
                "Label_mv": 2,
                "Label_st": 6,
                "nFrames": 3,
                "TSInit": 1_000_000,
                "TSEnd": 1_300_000,
            },
            {
                "source_row": 1,
                "Subject": 1,
                "Sequence": 1,
                "FrameIni": 12,
                "FrameEnd": 13,
                "Label_mv": 3,
                "Label_st": 7,
                "nFrames": 2,
                "TSInit": 1_300_000,
                "TSEnd": 1_450_000,
            },
        ]
    )
    unique, conflicts = expand_es2_unique_frames(synthetic_labels)
    assert len(unique) == 4
    assert int(unique["movement_conflict"].sum()) == 1
    assert len(conflicts) == 1
    conflict = unique[unique["frame_index"] == 12].iloc[0]
    assert not bool(conflict["supervised_movement"])

    # Regression test for the three conflicting pairs in the current release.
    release_overlap_rows = pd.DataFrame(
        [
            dict(source_row=483, Subject=1, Sequence=4, FrameIni=235, FrameEnd=250,
                 Label_mv=0, Label_st=np.nan, nFrames=16,
                 TSInit=1743026919929000, TSEnd=1743026922180220),
            dict(source_row=484, Subject=1, Sequence=4, FrameIni=209, FrameEnd=261,
                 Label_mv=2, Label_st=6, nFrames=53,
                 TSInit=1743026916026880, TSEnd=1743026923757070),
            dict(source_row=618, Subject=2, Sequence=1, FrameIni=799, FrameEnd=818,
                 Label_mv=3, Label_st=7, nFrames=20,
                 TSInit=1743134206720320, TSEnd=1743134209717940),
            dict(source_row=619, Subject=2, Sequence=1, FrameIni=818, FrameEnd=828,
                 Label_mv=5, Label_st=np.nan, nFrames=11,
                 TSInit=1743134209717940, TSEnd=1743135557706470),
            dict(source_row=623, Subject=2, Sequence=1, FrameIni=862, FrameEnd=882,
                 Label_mv=2, Label_st=6, nFrames=21,
                 TSInit=1743135562809240, TSEnd=1743135565786860),
            dict(source_row=624, Subject=2, Sequence=1, FrameIni=876, FrameEnd=923,
                 Label_mv=3, Label_st=7, nFrames=48,
                 TSInit=1743135564910380, TSEnd=1743135571976220),
        ]
    )
    release_unique, _ = expand_es2_unique_frames(release_overlap_rows)
    assert int(release_unique["movement_conflict"].sum()) == 24
    release_counts = (
        release_unique[release_unique["movement_conflict"]]
        .groupby(["subject", "sequence"])
        .size()
        .to_dict()
    )
    assert release_counts == {(1, 4): 16, (2, 1): 8}

    packet_dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("t", "<u8"), ("p", "u1")])
    packet_events = np.zeros(20, dtype=packet_dtype)
    packet_events["x"] = np.arange(20) * 10 % SENSOR_WIDTH
    packet_events["y"] = np.arange(20) * 7 % SENSOR_HEIGHT
    packet_events["t"] = np.arange(20) * 50_000 + 600_000
    packet_events["p"] = np.arange(20) % 2
    packet_columns = structured_events_to_columns(packet_events)
    targets = unique.sort_values(["anchor_timestamp_us", "frame_index"])[
        "anchor_timestamp_us"
    ].to_numpy(dtype=np.int64)
    frames, stats, summary = render_target_sequence(
        [packet_columns], targets, (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT), config
    )
    assert frames.shape == (4, 2, 12, 16)
    assert frames.dtype == np.float16
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    assert len(stats) == 4
    assert summary["targets"] == 4

    report = {
        "completed_utc": utc_now(),
        "pipeline_version": VERSION,
        "status": "PASSED",
        "eventsleep1_frame_shape": list(frame.shape),
        "uint8_sensitivity_path": "PASSED",
        "eventsleep1_source_unchanged": source_hash_before == source_hash_after,
        "eventsleep2_unique_keys": len(unique),
        "eventsleep2_conflicting_keys_masked": int(unique["movement_conflict"].sum()),
        "current_release_overlap_regression_frames": int(
            release_unique["movement_conflict"].sum()
        ),
        "eventsleep2_rendered_shape": list(frames.shape),
    }
    atomic_json(output / "smoke_test_completed.json", report)
    return report


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eventsleep1-root", type=Path, default=DEFAULT_ES1_ROOT)
    parser.add_argument("--eventsleep2-root", type=Path, default=DEFAULT_ES2_ROOT)
    parser.add_argument(
        "--eventsleep2-labels",
        type=Path,
        default=None,
        help="Optional explicit LabelsDatasetV2.csv path; otherwise discovered recursively",
    )


def add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-height", type=int, default=120)
    parser.add_argument("--output-width", type=int, default=160)
    parser.add_argument("--history-ms", type=int, default=512)
    parser.add_argument(
        "--storage-dtype",
        choices=["float16", "uint8"],
        default="float16",
        help="float16 is the primary scientific cache; uint8 is a compact sensitivity option",
    )
    parser.add_argument(
        "--roi-profile",
        choices=["full", "official"],
        default="full",
        help="Use full for the primary experiment; official is a reproduction sensitivity analysis",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace stale per-source outputs; never modifies source datasets",
    )
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EventSleep1 and EventSleep2 camera data without mixing datasets"
    )
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="Validate release inventory and labels")
    add_source_arguments(preflight_parser)
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--allow-release-drift", action="store_true")
    preflight_parser.add_argument("--verbose", action="store_true")

    es1_parser = subparsers.add_parser("eventsleep1", help="Convert official EventSleep1 clips")
    add_source_arguments(es1_parser)
    add_processing_arguments(es1_parser)
    es1_parser.add_argument("--max-clips", type=int, default=None, help="Pilot only")

    es2_parser = subparsers.add_parser("eventsleep2", help="Convert cleaned EventSleep2 frame keys")
    add_source_arguments(es2_parser)
    add_processing_arguments(es2_parser)
    es2_parser.add_argument("--max-recordings", type=int, default=None, help="Pilot only")

    all_parser = subparsers.add_parser("all", help="Preflight, preprocess both separately, verify")
    add_source_arguments(all_parser)
    add_processing_arguments(all_parser)
    all_parser.add_argument("--allow-release-drift", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="Verify a completed output tree")
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--verbose", action="store_true")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run deterministic synthetic tests")
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--verbose", action="store_true")
    return parser


def representation_from_args(args: argparse.Namespace) -> RepresentationConfig:
    config = RepresentationConfig(
        output_height=int(args.output_height),
        output_width=int(args.output_width),
        history_us=int(args.history_ms) * 1000,
        roi_profile=str(args.roi_profile),
        dtype=str(args.storage_dtype),
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.resolve()
    configure_logging(output, bool(getattr(args, "verbose", False)))
    logging.info("EventSleep preprocessing pipeline v%s", VERSION)
    try:
        if args.command == "smoke-test":
            report = smoke_test(output)
            logging.info("Smoke test PASSED: %s", report)
            return 0
        if args.command == "verify":
            report = verify_preprocessed(output, require_both=True)
            logging.info("Verification PASSED: %s", report)
            return 0

        es1_root = args.eventsleep1_root.resolve()
        es2_root = args.eventsleep2_root.resolve()
        labels_path: Path | None = None
        if args.command in {"preflight", "eventsleep2", "all"}:
            labels_path = locate_es2_labels(
                es2_root,
                None if args.eventsleep2_labels is None else args.eventsleep2_labels.resolve(),
            )
        relevant_roots = {
            "preflight": [es1_root, es2_root],
            "eventsleep1": [es1_root],
            "eventsleep2": [es2_root],
            "all": [es1_root, es2_root],
        }[args.command]
        require_separate_output(output, relevant_roots)

        if args.command == "preflight":
            assert labels_path is not None
            report = preflight(
                es1_root,
                es2_root,
                labels_path,
                bool(args.allow_release_drift),
            )
            atomic_json(output / "preflight.json", report)
            atomic_json(output / "software_environment.json", software_environment())
            logging.info("Preflight PASSED: %s", report["actual"])
            return 0

        config = representation_from_args(args)
        write_run_configuration(output, args, config)
        if args.command == "eventsleep1":
            run_eventsleep1(
                es1_root,
                output,
                config,
                max(1, int(args.workers)),
                bool(args.overwrite),
                args.max_clips,
            )
        elif args.command == "eventsleep2":
            assert labels_path is not None
            run_eventsleep2(
                es2_root,
                labels_path,
                output,
                config,
                max(1, int(args.workers)),
                bool(args.overwrite),
                args.max_recordings,
            )
        elif args.command == "all":
            assert labels_path is not None
            report = preflight(
                es1_root,
                es2_root,
                labels_path,
                bool(args.allow_release_drift),
            )
            atomic_json(output / "preflight.json", report)
            run_eventsleep1(
                es1_root,
                output,
                config,
                max(1, int(args.workers)),
                bool(args.overwrite),
                None,
            )
            run_eventsleep2(
                es2_root,
                labels_path,
                output,
                config,
                max(1, int(args.workers)),
                bool(args.overwrite),
                None,
            )
            verify_preprocessed(output, require_both=True)
        else:  # pragma: no cover - argparse prevents this
            parser.error(f"Unhandled command {args.command}")
        logging.info("Pipeline command completed successfully: %s", args.command)
        return 0
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        logging.debug("%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
