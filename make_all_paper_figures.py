#!/usr/bin/env python3
"""Reproduce all quantitative figures in the EventSleepFormer paper.

This script generates Figures 2, 3, and 4 directly from the archived
server outputs collected in ``paper_figure_sources_*.tar.gz`` (or from
an extracted copy of that archive). Figure 1 is intentionally excluded
because it is the manually designed workflow diagram.

No experimental metric or confusion-matrix value is hard-coded here.
All plotted scientific values are read from the archived CSV/JSON files.

Required packages
-----------------
    numpy
    pandas
    matplotlib
    PyMuPDF

Example
-------
Run directly from the downloaded archive::

    python make_all_paper_figures.py \
        --source paper_figure_sources_20260922_133931.tar.gz \
        --out-dir reproduced_figures

Or, after extracting it::

    python make_all_paper_figures.py \
        --source paper_figure_sources_20260922_133931 \
        --out-dir reproduced_figures

Outputs expected by the current LaTeX manuscript
------------------------------------------------
    fig2a_eventsleep1_per_class_f1.pdf
    fig2b_eventsleep1_confusion_matrix.pdf
    fig2_eventsleep1_classwise.pdf
    fig3_eventsleep2_method_comparison.pdf
    fig4a_eventsleep2_movement_confusion.pdf
    fig4b_eventsleep2_posture_confusion.pdf
    fig4c_eventsleep2_per_class_f1.pdf
    fig4_eventsleep2_diagnostics.pdf

An audit JSON file is also written as ``paper_figure_values.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_SEEDS = (13, 37, 73, 101, 137)

# Consistent, compact settings suitable for IEEE two-column figures.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_outer_archive_checksum(archive: Path) -> None:
    checksum_path = Path(str(archive) + ".sha256")
    if not checksum_path.exists():
        print(f"[checksum] Adjacent checksum not found; skipping outer archive check: {checksum_path.name}")
        return
    line = checksum_path.read_text(encoding="utf-8").strip().splitlines()[0]
    expected = line.split()[0]
    actual = sha256_file(archive)
    if actual != expected:
        raise RuntimeError(
            f"Outer archive SHA-256 mismatch:\nexpected {expected}\nactual   {actual}"
        )
    print(f"[checksum] Outer archive verified: {actual}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in archive: {member.name}") from exc
        # Python 3.12+ supports extraction filters; use the safe data filter when available.
        try:
            tf.extractall(destination, filter="data")
        except TypeError:
            # Compatibility with older Python versions after the explicit path-safety check above.
            tf.extractall(destination)


def locate_source_root(path: Path) -> Path:
    """Return directory that directly contains figure2_eventsleep1, etc."""
    required = {"figure2_eventsleep1", "figure3_eventsleep2", "figure4_eventsleep2"}
    if path.is_dir() and required.issubset({p.name for p in path.iterdir()}):
        return path
    if path.is_dir():
        candidates = []
        for p in path.rglob("figure2_eventsleep1"):
            parent = p.parent
            if all((parent / r).is_dir() for r in required):
                candidates.append(parent)
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple source roots found under {path}: {candidates}")
    raise RuntimeError(
        f"Could not locate figure source root under {path}. Expected figure2_eventsleep1/, "
        "figure3_eventsleep2/, and figure4_eventsleep2/."
    )


def verify_internal_checksums(root: Path) -> None:
    sums = root / "SHA256SUMS.txt"
    if not sums.exists():
        print("[checksum] Internal SHA256SUMS.txt not found; skipping internal verification.")
        return
    failures = []
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(maxsplit=1)
        rel = rel.strip()
        if rel.startswith("./"):
            rel = rel[2:]
        p = root / rel
        if not p.exists():
            failures.append(f"MISSING {rel}")
            continue
        actual = sha256_file(p)
        checked += 1
        if actual != expected:
            failures.append(f"MISMATCH {rel}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Internal checksum verification failed:\n" + "\n".join(failures))
    print(f"[checksum] Internal source files verified: {checked} files")


def numeric_seed_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception as exc:
        raise RuntimeError(f"Unexpected seed directory name: {path}") from exc


def seed_dirs(method_root: Path) -> List[Path]:
    dirs = sorted(
        [p for p in method_root.glob("seed_*") if p.is_dir()],
        key=numeric_seed_from_dir,
    )
    seeds = tuple(numeric_seed_from_dir(p) for p in dirs)
    if seeds != PAPER_SEEDS:
        raise RuntimeError(
            f"Unexpected seeds in {method_root}: {seeds}; expected {PAPER_SEEDS}"
        )
    return dirs


def load_es1_method(method_root: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return class names, per-seed F1 [seed,class], confusion [seed,class,class], supports."""
    f1_rows = []
    cm_rows = []
    class_names_ref = None
    support_ref = None

    for run_dir in seed_dirs(method_root):
        pc = pd.read_csv(run_dir / "per_class_metrics.csv").sort_values("class_id")
        cm_df = pd.read_csv(run_dir / "confusion_matrix.csv")

        class_names = pc["class_name"].astype(str).tolist()
        supports = pc["support"].to_numpy(dtype=float)
        cm_classes = [c for c in cm_df.columns if c != "true_class"]
        row_names = cm_df["true_class"].astype(str).tolist()

        if class_names_ref is None:
            class_names_ref = class_names
            support_ref = supports
        if class_names != class_names_ref:
            raise RuntimeError(f"Class order mismatch in {run_dir / 'per_class_metrics.csv'}")
        if cm_classes != class_names_ref or row_names != class_names_ref:
            raise RuntimeError(f"Confusion matrix class order mismatch in {run_dir}")
        if not np.array_equal(supports, support_ref):
            raise RuntimeError(f"Class support mismatch across seeds in {method_root}")

        cm = cm_df[class_names_ref].to_numpy(dtype=float)
        if not np.array_equal(cm.sum(axis=1), supports):
            raise RuntimeError(f"Confusion row sums do not match supports in {run_dir}")

        f1_rows.append(pc["f1"].to_numpy(dtype=float))
        cm_rows.append(cm)

    assert class_names_ref is not None and support_ref is not None
    return class_names_ref, np.stack(f1_rows), np.stack(cm_rows), support_ref


def stitch_pdfs(paths: Sequence[Path], output_path: Path, gap_pt: float = 8.0) -> None:
    import fitz

    docs = [fitz.open(str(p)) for p in paths]
    try:
        widths = [d[0].rect.width for d in docs]
        heights = [d[0].rect.height for d in docs]
        target_h = max(heights)
        scaled_w = [w * target_h / h for w, h in zip(widths, heights)]
        total_w = sum(scaled_w) + gap_pt * (len(docs) - 1)

        outdoc = fitz.open()
        page = outdoc.new_page(width=total_w, height=target_h)
        x = 0.0
        for d, w in zip(docs, scaled_w):
            page.show_pdf_page(fitz.Rect(x, 0, x + w, target_h), d, 0)
            x += w + gap_pt
        outdoc.save(str(output_path))
        outdoc.close()
    finally:
        for d in docs:
            d.close()


def make_figure2(root: Path, out_dir: Path, audit: Dict) -> None:
    src = root / "figure2_eventsleep1"
    es_names, es_f1, es_cm, es_support = load_es1_method(src / "eventsleepformer")
    re_names, re_f1, re_cm, re_support = load_es1_method(src / "resnet_e")

    if es_names != re_names or not np.array_equal(es_support, re_support):
        raise RuntimeError("EventSleepFormer and ResNet-E class definitions/supports do not match.")

    classes = es_names
    es_mean_f1 = es_f1.mean(axis=0)
    re_mean_f1 = re_f1.mean(axis=0)

    # Figure 2(a): five-seed mean per-class F1.
    x = np.arange(len(classes))
    width = 0.38
    fig = plt.figure(figsize=(8.6, 3.8))
    ax = fig.add_axes([0.08, 0.30, 0.90, 0.64])
    ax.bar(x - width / 2, re_mean_f1, width, label="ResNet-E")
    ax.bar(x + width / 2, es_mean_f1, width, label="EventSleepFormer")
    ax.set_ylabel("Mean F1-score")
    ax.set_xlabel("Activity class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=38, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(loc="upper center", ncol=2, frameon=True)
    p2a = out_dir / "fig2a_eventsleep1_per_class_f1.pdf"
    fig.savefig(p2a, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # Figure 2(b): average row-normalized EventSleepFormer confusion matrix.
    row_sums = es_cm.sum(axis=2, keepdims=True)
    normalized_per_seed = np.divide(
        es_cm,
        row_sums,
        out=np.zeros_like(es_cm, dtype=float),
        where=row_sums != 0,
    )
    avg_norm_cm = normalized_per_seed.mean(axis=0)

    fig = plt.figure(figsize=(4.3, 3.8))
    ax = fig.add_axes([0.15, 0.18, 0.68, 0.72])
    im = ax.imshow(avg_norm_cm, cmap="cividis", vmin=0.0, vmax=1.0, aspect="equal")
    tick_idx = np.arange(0, len(classes), 2)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_idx)
    ax.set_yticklabels(tick_idx)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    cax = fig.add_axes([0.85, 0.18, 0.035, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    p2b = out_dir / "fig2b_eventsleep1_confusion_matrix.pdf"
    fig.savefig(p2b, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    stitch_pdfs((p2a, p2b), out_dir / "fig2_eventsleep1_classwise.pdf", gap_pt=8.0)

    audit["figure2"] = {
        "classes": classes,
        "seeds": list(PAPER_SEEDS),
        "resnet_e_mean_f1": re_mean_f1.tolist(),
        "eventsleepformer_mean_f1": es_mean_f1.tolist(),
        "eventsleepformer_average_row_normalized_confusion": avg_norm_cm.tolist(),
        "support_per_seed": es_support.astype(int).tolist(),
    }
    print("[Figure 2] generated from 10 EventSleep1 seed runs (5 per method)")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_summary_mean(summary: Dict, key: str) -> float:
    return float(summary["individual_seed_summary"][key]["mean"])


def get_adapted_laplace_mean(summary: Dict, key: str) -> float:
    return float(summary["member_metric_summary"][f"laplace_{key}"]["mean"])


def make_figure3(root: Path, out_dir: Path, audit: Dict) -> None:
    src = root / "figure3_eventsleep2"
    adapted = read_json(src / "adapted_evs2net_final_summary.json")
    reference = read_json(src / "reference_eventsleepformer_final_summary.json")
    proposed = read_json(src / "proposed_eventsleepformer_final_summary.json")

    metric_keys = (
        "motion_per_frame_macro_f1",
        "static_per_frame_macro_f1",
        "motion_per_gt_clip_macro_f1",
        "static_per_gt_clip_macro_f1",
    )

    adapted_values = [get_adapted_laplace_mean(adapted, k) for k in metric_keys]
    reference_values = [get_summary_mean(reference, k) for k in metric_keys]
    proposed_values = [get_summary_mean(proposed, k) for k in metric_keys]

    methods = [
        "Adapted\nEvS2-net",
        "Reference\nEventSleepFormer",
        "Proposed\nEventSleepFormer",
    ]
    values = np.array([adapted_values, reference_values, proposed_values], dtype=float)
    labels = [
        "Frame movement Macro-F1",
        "Frame posture Macro-F1",
        "Clip movement Macro-F1",
        "Clip posture Macro-F1",
    ]

    x = np.arange(len(methods))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig = plt.figure(figsize=(8.8, 5.2))
    ax = fig.add_axes([0.10, 0.16, 0.87, 0.72])
    containers = []
    for j, label in enumerate(labels):
        containers.append(
            ax.bar(x + offsets[j], values[:, j], width, label=label, edgecolor="black", linewidth=0.7)
        )

    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    upper = max(0.55, float(values.max()) + 0.07)
    ax.set_ylim(0.0, upper)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2, frameon=True)

    for container in containers:
        for bar in container:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    out = out_dir / "fig3_eventsleep2_method_comparison.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    audit["figure3"] = {
        "methods": [m.replace("\n", " ") for m in methods],
        "metrics": metric_keys,
        "values": values.tolist(),
        "source_rule": {
            "adapted_evs2net": "member_metric_summary laplace_* five-member means",
            "reference_eventsleepformer": "individual_seed_summary five-seed means",
            "proposed_eventsleepformer": "individual_seed_summary five-seed means",
        },
    }
    print("[Figure 3] generated from three EventSleep2 final-summary JSON files")


def load_figure4_data(data_dir: Path):
    mov_cm_df = pd.read_csv(data_dir / "movement_frame_confusion.csv")
    pos_cm_df = pd.read_csv(data_dir / "static_frame_confusion.csv")
    mov_pc = pd.read_csv(data_dir / "movement_frame_per_class.csv")
    pos_pc = pd.read_csv(data_dir / "static_frame_per_class.csv")

    mov_classes = [c for c in mov_cm_df.columns if c != "true_class"]
    pos_classes = [c for c in pos_cm_df.columns if c != "true_class"]
    mov_cm = mov_cm_df[mov_classes].to_numpy(dtype=int)
    pos_cm = pos_cm_df[pos_classes].to_numpy(dtype=int)
    mov_f1 = mov_pc["f1"].to_numpy(dtype=float)
    pos_f1 = pos_pc["f1"].to_numpy(dtype=float)

    if mov_classes != mov_pc["class_name"].astype(str).tolist():
        raise RuntimeError("Figure 4 movement class order mismatch")
    if pos_classes != pos_pc["class_name"].astype(str).tolist():
        raise RuntimeError("Figure 4 posture class order mismatch")
    if mov_classes[-1] != "ArmsShake":
        raise RuntimeError(f"Expected final movement class ArmsShake, got {mov_classes[-1]}")
    if mov_cm.sum() != int(mov_pc["support"].sum()):
        raise RuntimeError("Figure 4 movement confusion total does not match support")
    if pos_cm.sum() != int(pos_pc["support"].sum()):
        raise RuntimeError("Figure 4 posture confusion total does not match support")
    return mov_classes, pos_classes, mov_cm, pos_cm, mov_f1, pos_f1


def annotate_confusion(ax, cm: np.ndarray) -> None:
    threshold = 0.50 * float(cm.max()) if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if cm[i, j] > threshold else "white",
            )


def make_figure4(root: Path, out_dir: Path, audit: Dict) -> None:
    src = root / "figure4_eventsleep2"
    mov_classes, pos_classes, mov_cm, pos_cm, mov_f1, pos_f1 = load_figure4_data(src)

    # Panel (a): movement confusion matrix.
    fig = plt.figure(figsize=(3.2, 2.7))
    ax = fig.add_axes([0.24, 0.28, 0.70, 0.65])
    ax.imshow(mov_cm, cmap="cividis", aspect="equal")
    ax.set_xticks(np.arange(len(mov_classes)))
    ax.set_yticks(np.arange(len(mov_classes)))
    ax.set_xticklabels(mov_classes, rotation=38, ha="right", fontsize=6.5)
    ax.set_yticklabels(mov_classes, fontsize=6.5)
    ax.set_xlabel("Predicted class", fontsize=8)
    ax.set_ylabel("True class", fontsize=8)
    annotate_confusion(ax, mov_cm)
    ax.text(-0.22, 1.03, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    p4a = out_dir / "fig4a_eventsleep2_movement_confusion.pdf"
    fig.savefig(p4a, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Panel (b): posture confusion matrix.
    fig = plt.figure(figsize=(2.6, 2.7))
    ax = fig.add_axes([0.25, 0.28, 0.68, 0.65])
    ax.imshow(pos_cm, cmap="cividis", aspect="equal")
    ax.set_xticks(np.arange(len(pos_classes)))
    ax.set_yticks(np.arange(len(pos_classes)))
    ax.set_xticklabels(pos_classes, rotation=38, ha="right", fontsize=6.5)
    ax.set_yticklabels(pos_classes, fontsize=6.5)
    ax.set_xlabel("Predicted class", fontsize=8)
    ax.set_ylabel("True class", fontsize=8)
    annotate_confusion(ax, pos_cm)
    ax.text(-0.22, 1.03, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    p4b = out_dir / "fig4b_eventsleep2_posture_confusion.pdf"
    fig.savefig(p4b, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Panel (c): per-class F1.
    fig = plt.figure(figsize=(4.3, 2.7))
    ax = fig.add_axes([0.12, 0.30, 0.86, 0.62])
    gap = 1.0
    x_mov = np.arange(len(mov_f1))
    x_pos = np.arange(len(pos_f1)) + len(mov_f1) + gap
    bars1 = ax.bar(x_mov, mov_f1, width=0.72, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x_pos, pos_f1, width=0.72, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x_mov) + list(x_pos))
    ax.set_xticklabels(mov_classes + pos_classes, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("F1 score", fontsize=8)
    upper = max(0.72, float(max(mov_f1.max(), pos_f1.max())) + 0.12)
    ax.set_ylim(0.0, upper)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    sep_x = len(mov_f1) - 0.5 + gap / 2
    ax.axvline(sep_x, linestyle="--", linewidth=0.8)
    ax.text((x_mov[0] + x_mov[-1]) / 2, upper - 0.02, "Movement", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text((x_pos[0] + x_pos[-1]) / 2, upper - 0.02, "Posture", ha="center", va="top", fontsize=8, fontweight="bold")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.text(-0.10, 1.03, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    p4c = out_dir / "fig4c_eventsleep2_per_class_f1.pdf"
    fig.savefig(p4c, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    final = out_dir / "fig4_eventsleep2_diagnostics.pdf"
    stitch_pdfs((p4a, p4b, p4c), final, gap_pt=8.0)

    audit["figure4"] = {
        "movement_classes": mov_classes,
        "movement_confusion": mov_cm.tolist(),
        "movement_f1": mov_f1.tolist(),
        "posture_classes": pos_classes,
        "posture_confusion": pos_cm.tolist(),
        "posture_f1": pos_f1.tolist(),
        "movement_samples": int(mov_cm.sum()),
        "posture_supervised_samples": int(pos_cm.sum()),
    }
    print(
        f"[Figure 4] generated from ensemble outputs: movement n={mov_cm.sum()}, posture-supervised n={pos_cm.sum()}"
    )


def prepare_source(source: Path, temp_parent: Path | None = None) -> Tuple[Path, tempfile.TemporaryDirectory | None]:
    source = source.expanduser().resolve()
    if source.is_dir():
        return locate_source_root(source), None
    if not source.is_file():
        raise FileNotFoundError(source)
    if not (source.name.endswith(".tar.gz") or source.name.endswith(".tgz")):
        raise ValueError("--source must be an extracted directory or a .tar.gz/.tgz archive")

    verify_outer_archive_checksum(source)
    tmp = tempfile.TemporaryDirectory(dir=str(temp_parent) if temp_parent else None)
    tmp_path = Path(tmp.name)
    safe_extract_tar(source, tmp_path)
    return locate_source_root(tmp_path), tmp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce EventSleepFormer paper Figures 2, 3, and 4 from archived server outputs."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="paper_figure_sources_*.tar.gz or its extracted directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced_figures"),
        help="Output directory (default: reproduced_figures)",
    )
    parser.add_argument(
        "--skip-internal-checksums",
        action="store_true",
        help="Skip verification of SHA256SUMS.txt after source extraction",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_root, temp_handle = prepare_source(args.source)
    try:
        print(f"[source] {source_root}")
        if not args.skip_internal_checksums:
            verify_internal_checksums(source_root)

        audit: Dict = {
            "source_root_name": source_root.name,
            "paper_seeds": list(PAPER_SEEDS),
            "note": "All plotted scientific values are derived from archived CSV/JSON outputs, not hard-coded in the plotting code.",
        }

        make_figure2(source_root, out_dir, audit)
        make_figure3(source_root, out_dir, audit)
        make_figure4(source_root, out_dir, audit)

        audit_path = out_dir / "paper_figure_values.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

        print("\nReproduction complete. Generated figure files:")
        for name in (
            "fig2a_eventsleep1_per_class_f1.pdf",
            "fig2b_eventsleep1_confusion_matrix.pdf",
            "fig2_eventsleep1_classwise.pdf",
            "fig3_eventsleep2_method_comparison.pdf",
            "fig4a_eventsleep2_movement_confusion.pdf",
            "fig4b_eventsleep2_posture_confusion.pdf",
            "fig4c_eventsleep2_per_class_f1.pdf",
            "fig4_eventsleep2_diagnostics.pdf",
        ):
            p = out_dir / name
            print(f"  {p}")
        print(f"\nAudit values: {audit_path}")
    finally:
        if temp_handle is not None:
            temp_handle.cleanup()


if __name__ == "__main__":
    main()
