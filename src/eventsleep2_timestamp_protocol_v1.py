"""Explicit timestamp-interval annotation adapter; never renders or edits frames.

The source-row timestamp interval, not its released frame-index interval,
defines each claim. This is an adapted dataset protocol, not a reconstruction
of the authors' exact indexed cache. No predictions enter this module.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "timestamp_intervals_v1"
RULE = {
    "id": PROTOCOL,
    "anchor": "cached maximum event timestamp in microseconds",
    "membership": "TSInit <= anchor <= TSEnd",
    "tolerance_us": 0,
    "order": "original_csv_source_row_then_array_index",
    "overlap_policy": "preserve_each_source_row_claim",
    "empty_interval_policy": "omit_from_supervision_and_report_source_row",
    "unannotated_policy": "omit; no nearest-frame assignment or label propagation",
    "movement_minus_one_policy": "omit_from_supervision",
    "posture_policy": "supervise_released_static_label_only_on_movement_2_or_3",
    "exact_published_reproduction": False,
}
SAMPLE_COLUMNS = [
    "sample_id", "recording_key", "subject", "sequence", "source_row",
    "gt_segment_id", "array_index", "timestamp_us", "movement_label",
    "static_label", "frame_path",
]
BUNDLE_FILES = (
    "samples.csv", "annotation_rows.csv", "recordings.csv", "interval_audit.csv",
    "source_recordings.csv", "class_support.csv", "overlap_claims.csv",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_name(m, key):
    pair = m.parse_recording_key(key)
    for name, values in (("train", m.PUBLISHED_TRAIN_RECORDINGS),
                         ("validation", m.PUBLISHED_VALIDATION_RECORDINGS),
                         ("test", m.PUBLISHED_TEST_RECORDINGS)):
        if pair in values:
            return name
    raise ValueError(f"Recording is outside the fixed split: {key}")


def cache_path(preprocess_root, relative):
    root = Path(preprocess_root).resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root / "eventsleep2"):
        raise ValueError(f"Expected an EventSleep2 cache path under {root}: {relative}")
    return path


def validate_timestamps(timestamps, count):
    a = np.asarray(timestamps)
    if a.dtype != np.dtype("int64") or a.ndim != 1 or len(a) != int(count) or not len(a):
        raise ValueError("Cache timestamps must be a nonempty int64 vector matching frame count")
    if np.any(a[1:] < a[:-1]):
        raise ValueError("Cache timestamps are not monotonic")
    return a


def map_recording(m, selected, timestamps, record, encoding):
    """Pure mapping function; input frames are never opened here."""
    key = str(record["recording_key"])
    subject, sequence = m.parse_recording_key(key)
    anchors = validate_timestamps(timestamps, record["frames"])
    samples, audits = [], []
    for row in selected.sort_values("source_row", kind="stable").itertuples(index=False):
        source_row = int(row.source_row)
        start, end = int(row.TSInit), int(row.TSEnd)
        if start > end or int(row.Subject) != subject or int(row.Sequence) != sequence:
            raise ValueError(f"Invalid annotation interval/recording: {key}, row {source_row}")
        lo = int(np.searchsorted(anchors, start, side="left"))
        hi = int(np.searchsorted(anchors, end, side="right"))
        movement = int(row.Label_mv)
        if movement not in (-1, *range(6)):
            raise ValueError(f"Invalid movement class: {key}, row {source_row}")
        static = m._normalise_static_label(row.Label_st, movement, encoding)
        supervised = movement >= 0
        status = ("unsupervised_source_row" if not supervised else
                  "omitted_no_anchor_in_interval" if hi == lo else "mapped")
        audits.append({
            "recording_key": key, "split": split_name(m, key), "source_row": source_row,
            "movement_label": movement, "static_label": static,
            "released_start_index": int(row.FrameIni), "released_end_index": int(row.FrameEnd),
            "released_claims": int(row.FrameEnd) - int(row.FrameIni) + 1 if supervised else 0,
            "TSInit": start, "TSEnd": end, "timestamp_start_index": lo,
            "timestamp_stop_index_exclusive": hi,
            "mapped_claims": hi - lo if supervised else 0, "status": status,
        })
        if not supervised:
            continue
        for i in range(lo, hi):
            samples.append({
                "sample_id": f"tsv1_{key}_row{source_row:05d}_frame{i:06d}",
                "recording_key": key, "subject": subject, "sequence": sequence,
                "source_row": source_row, "gt_segment_id": f"tsv1_{key}_row{source_row:05d}",
                "array_index": i, "timestamp_us": int(anchors[i]),
                "movement_label": movement, "static_label": static,
                "frame_path": str(record["frame_path"]),
            })
    return pd.DataFrame(samples, columns=SAMPLE_COLUMNS), pd.DataFrame(audits)


def map_inventory(m, labels, records, preprocess_root):
    required = [m.recording_key(*p) for p in m.PUBLISHED_ALL_RECORDINGS]
    if records.recording_key.duplicated().any() or set(records.recording_key) != set(required):
        raise ValueError("Source recording inventory does not match the ten-recording fixed split")
    encoding = m._detect_static_label_encoding(labels)
    samples, audits, mapped_records = [], [], []
    for key in required:
        row = records.set_index("recording_key", drop=False).loc[key].to_dict()
        subject, sequence = m.parse_recording_key(key)
        selected = labels[(labels.Subject == subject) & (labels.Sequence == sequence)]
        if selected.empty:
            raise ValueError(f"No annotation rows for required recording: {key}")
        timestamps_path = cache_path(preprocess_root, row["timestamp_path"])
        # Validate the reference, but never open frame arrays during preparation.
        cache_path(preprocess_root, row["frame_path"])
        before = digest(timestamps_path)
        timestamps = np.load(timestamps_path, allow_pickle=False)
        frame, audit = map_recording(m, selected, timestamps, row, encoding)
        if digest(timestamps_path) != before:
            raise RuntimeError(f"Timestamp file changed during mapping: {key}")
        if frame.empty:
            raise ValueError(f"No supervised anchors in the entire recording: {key}")
        sizes = frame.groupby("array_index", sort=False).size()
        omitted = audit.status.eq("omitted_no_anchor_in_interval")
        row = {k: row[k] for k in (
            "recording_key", "frame_path", "timestamp_path", "frames",
            "raw_sha256", "semantic_signature",
        )}
        row.update({
            "split": split_name(m, key), "timestamps_sha256": before,
            "mapped_samples": len(frame), "static_samples": int(frame.static_label.ge(0).sum()),
            "unique_supervised_frames": int(len(sizes)),
            "overlap_positions": int(sizes.gt(1).sum()),
            "overlap_extra_claims": int((sizes - 1).sum()),
            "omitted_supervised_intervals": int(omitted.sum()),
        })
        samples.append(frame)
        audits.append(audit)
        mapped_records.append(row)
        logging.info("Timestamp mapping %s: claims=%d omitted_intervals=%d overlaps=%d",
                     key, len(frame), int(omitted.sum()), int(sizes.gt(1).sum()))
    return (pd.concat(samples, ignore_index=True), pd.concat(audits, ignore_index=True),
            pd.DataFrame(mapped_records))


def support_table(m, manifest):
    rows = []
    for split, frame in zip(("train", "validation", "test"), m.split_published_manifest(manifest)):
        for task, names, col in (("movement", m.MOVEMENT_NAMES, "movement_label"),
                                 ("static", m.STATIC_NAMES, "static_label")):
            for i, name in enumerate(names):
                rows.append({"split": split, "task": task, "class_id": i,
                             "class_name": name, "support": int(frame[col].eq(i).sum())})
    return pd.DataFrame(rows)


def create_bundle(m, preprocess_root, profile, labels_path, output):
    output = Path(output).resolve()
    # main() already creates output/logs. A repeated successful command verifies
    # and reuses the same bundle; an incomplete bundle must use a fresh path.
    if (output / "timestamp_dataset.json").is_file():
        _, marker = load_bundle(m, output, preprocess_root, profile)
        if labels_path is not None and digest(labels_path) != marker["labels_sha256"]:
            raise ValueError("Requested labels differ from the existing timestamp dataset")
        return {**marker, "bundle_status": "reused"}
    if output.exists() and any(p.name != "logs" for p in output.iterdir()):
        raise FileExistsError(f"Incomplete or unrelated output exists: {output}; choose a new output")
    source = m.published_manifest_path(preprocess_root, profile).parent
    if output == source or output.is_relative_to(source):
        raise ValueError("Timestamp dataset must be separate from the original frame cache")
    marker = json.loads((source / "completed_successfully.json").read_text())
    if marker.get("partial") is not False or marker.get("profile") != profile:
        raise ValueError("Source cache must be complete and have the selected profile")
    provenance = m.published_dataset_provenance(preprocess_root, profile)
    labels_path = Path(labels_path or provenance["labels_path"]).resolve()
    if digest(labels_path) != provenance["labels_sha256"]:
        raise ValueError("Source cache and requested annotation file have different SHA-256 values")
    # Pin the original metadata before reading it; recheck before committing.
    source_hashes = {name: digest(source / name) for name in (
        "completed_successfully.json", "recordings.csv", "samples.csv", "release_inventory.json",
    )}
    labels = m.load_es2_labels(labels_path)
    records = pd.read_csv(source / "recordings.csv")
    manifest, intervals, mapped_records = map_inventory(m, labels, records, preprocess_root)
    m.split_published_manifest(manifest)
    if manifest.sample_id.duplicated().any():
        raise ValueError("Duplicate timestamp sample IDs")
    supports = support_table(m, manifest)
    duplicates = manifest.duplicated(["recording_key", "array_index"], keep=False)
    overlap = manifest.loc[duplicates].copy()
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("samples.csv", manifest), ("annotation_rows.csv", labels),
        ("source_recordings.csv", records), ("recordings.csv", mapped_records),
        ("interval_audit.csv", intervals), ("class_support.csv", supports),
        ("overlap_claims.csv", overlap),
    ):
        m.atomic_csv(output / name, frame)
    if source_hashes != {name: digest(source / name) for name in source_hashes}:
        raise RuntimeError("Source cache metadata changed during timestamp preparation")
    if digest(labels_path) != provenance["labels_sha256"]:
        raise RuntimeError("Source annotations changed during timestamp preparation")
    file_hashes = {name: digest(output / name) for name in BUNDLE_FILES}
    payload = {
        "annotation_protocol": RULE,
        "profile": profile,
        "paper_split": m.published_configuration(profile)["split"],
        "source_dataset_fingerprint": provenance["dataset_fingerprint"],
        "source_cache_metadata_sha256": source_hashes,
        "labels_sha256": provenance["labels_sha256"],
        "adapter_source_sha256": digest(Path(__file__)),
        "files_sha256": file_hashes,
    }
    partitions = m.split_published_manifest(manifest)
    summary = {
        "completed_utc": m.utc_now(), "pipeline_version": m.VERSION,
        "annotation_protocol": PROTOCOL, "rule": RULE, "profile": profile,
        "partial": False, "bundle_status": "created", "cache_version": "timestamp_manifest_v1",
        "source_cache_version": m.PUBLISHED_CACHE_VERSION,
        "source_cache_metadata_sha256": source_hashes,
        "labels_path": str(labels_path), "labels_sha256": provenance["labels_sha256"],
        "source_dataset_fingerprint": provenance["dataset_fingerprint"],
        "dataset_fingerprint_payload": payload, "dataset_fingerprint": m.stable_hash(payload, 64),
        "samples": {name: int(len(frame)) for name, frame in zip(("train", "validation", "test"), partitions)},
        "supervised_samples": int(len(manifest)),
        "rendered_frame_total": int(mapped_records.frames.sum()),
        "omitted_supervised_intervals": int(intervals.status.eq("omitted_no_anchor_in_interval").sum()),
        "multiply_claimed_physical_frame_positions": int(mapped_records.overlap_positions.sum()),
        "overlap_extra_sample_claims": int(mapped_records.overlap_extra_claims.sum()),
        "overlap_policy": RULE["overlap_policy"], "timestamp_membership_verified": True,
        "semantic_alignment_verified": False,
        "exact_published_reproduction": False, "frame_cache_changed": False,
        "source_annotation_file_changed": False, "official_test_evaluated": False,
        "test_frame_arrays_opened": [],
        "limitation": "Timestamp membership verifies the mapping rule, not human annotation accuracy or exact author-cache reproduction.",
    }
    m.atomic_json(output / "timestamp_dataset.json", summary)
    return summary


def load_bundle(m, bundle, preprocess_root, profile):
    bundle = Path(bundle).resolve()
    marker = json.loads((bundle / "timestamp_dataset.json").read_text())
    if (marker.get("annotation_protocol") != PROTOCOL or marker.get("rule") != RULE
            or marker.get("profile") != profile or marker.get("partial") is not False):
        raise ValueError("Timestamp dataset protocol/profile/complete marker mismatch")
    payload = marker.get("dataset_fingerprint_payload", {})
    if (payload.get("annotation_protocol") != RULE or payload.get("profile") != profile
            or payload.get("paper_split") != m.published_configuration(profile)["split"]
            or payload.get("labels_sha256") != marker.get("labels_sha256")
            or payload.get("source_dataset_fingerprint") != marker.get("source_dataset_fingerprint")
            or payload.get("source_cache_metadata_sha256") != marker.get("source_cache_metadata_sha256")
            or payload.get("adapter_source_sha256") != digest(Path(__file__))
            or m.stable_hash(payload, 64) != marker.get("dataset_fingerprint")):
        raise ValueError("Timestamp dataset fingerprint, rule or implementation mismatch")
    expected_files = payload.get("files_sha256", {})
    if set(expected_files) != set(BUNDLE_FILES):
        raise ValueError("Timestamp dataset artifact inventory mismatch")
    for name, expected in expected_files.items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Timestamp dataset file SHA-256 mismatch: {name}")
    source = m.published_manifest_path(preprocess_root, profile).parent
    provenance = m.published_dataset_provenance(preprocess_root, profile)
    if provenance["dataset_fingerprint"] != marker["source_dataset_fingerprint"]:
        raise ValueError("Source frame cache dataset fingerprint changed")
    source_hashes = marker.get("source_cache_metadata_sha256", {})
    if set(source_hashes) != {"completed_successfully.json", "recordings.csv", "samples.csv", "release_inventory.json"}:
        raise ValueError("Source metadata hash inventory mismatch")
    for name, expected in source_hashes.items():
        if digest(source / name) != expected:
            raise ValueError(f"Source frame cache metadata changed: {name}")
    records = pd.read_csv(bundle / "recordings.csv")
    for row in records.itertuples(index=False):
        if digest(cache_path(preprocess_root, row.timestamp_path)) != row.timestamps_sha256:
            raise ValueError(f"Cached timestamp SHA-256 mismatch: {row.recording_key}")
        cache_path(preprocess_root, row.frame_path)
    frame = pd.read_csv(bundle / "samples.csv")
    train, val, test = m.split_published_manifest(frame)
    counts = dict(zip(("train", "validation", "test"), map(len, (train, val, test))))
    if (counts != marker.get("samples") or len(frame) != marker.get("supervised_samples")
            or frame.sample_id.duplicated().any()
            or not frame.movement_label.isin(range(6)).all()
            or not frame.static_label.isin([-100, 0, 1, 2, 3]).all()):
        raise ValueError("Timestamp manifest counts, sample IDs or classes are invalid")
    return frame, marker


def preflight(m, args):
    bundle = Path(args.timestamp_dataset).resolve()
    root = args.preprocess_root.resolve()
    frame, marker = load_bundle(m, bundle, root, str(args.profile))
    labels = pd.read_csv(bundle / "annotation_rows.csv")
    source_records = pd.read_csv(bundle / "source_recordings.csv")
    expected, intervals, records = map_inventory(m, labels, source_records, root)
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(bundle / "interval_audit.csv"), intervals, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(bundle / "recordings.csv"), records, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_csv(bundle / "class_support.csv"), support_table(m, frame), check_dtype=False)
    opened, invalid, array_stats = [], [], []
    # Preflight deliberately leaves test frames unopened. The same deterministic
    # rule is checked against test timestamps/annotations, without model scores.
    for row in records.itertuples(index=False):
        if row.split == "test":
            continue
        path = cache_path(root, row.frame_path)
        opened.append(row.recording_key)
        try:
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.shape != (int(row.frames), 2, m.PAPER_OUTPUT_HEIGHT, m.PAPER_OUTPUT_WIDTH) or a.dtype != np.float32:
                raise ValueError(f"Unexpected shape/dtype: {a.shape}, {a.dtype}")
            empty = 0
            if args.full_array_check:
                for i in range(0, len(a), 16):
                    block = a[i:i + 16]
                    if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
                        raise ValueError("Surface values must be finite and within [0, 1]")
                    empty += int(np.count_nonzero(~np.any(block != 0, axis=(1, 2, 3))))
            array_stats.append({"recording_key": row.recording_key, "frames": int(len(a)),
                                "empty_frames": empty if args.full_array_check else None})
            del a
        except (OSError, ValueError) as exc:
            invalid.append({"recording_key": row.recording_key, "error": str(exc)})
    report = {
        "completed_utc": m.utc_now(), "pipeline_version": m.VERSION, "passed": not invalid,
        "annotation_protocol": PROTOCOL, "dataset_fingerprint": marker["dataset_fingerprint"],
        "samples": marker["samples"], "claim_order_valid": True,
        "timestamp_membership_verified": True, "semantic_alignment_verified": False,
        "exact_published_reproduction": False,
        "omitted_supervised_intervals": marker["omitted_supervised_intervals"],
        "overlap_extra_sample_claims": marker["overlap_extra_sample_claims"],
        "invalid_arrays": invalid, "frame_arrays_opened": opened,
        "array_statistics": array_stats, "full_array_check": bool(args.full_array_check),
        "test_frame_arrays_opened": [], "official_test_evaluated": False,
    }
    m.atomic_json(args.output.resolve() / "timestamp_preflight.json", report)
    if invalid:
        raise RuntimeError(f"Timestamp preflight found invalid training/validation arrays: {invalid}")
    return report
