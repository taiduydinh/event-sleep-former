\
#!/usr/bin/env python3
"""Download exactly the EventSleep files used by the paper from Synapse."""
from __future__ import annotations
import argparse, csv, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTS = ROOT / "data" / "file_lists"
DEFAULT_ROOT = ROOT / "data" / "raw"


def load_rows(dataset: str):
    with (LISTS / f"{dataset}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def login():
    try:
        import synapseclient
    except ImportError as exc:
        raise SystemExit("synapseclient is required. Run: pip install -r requirements.txt") from exc
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token:
        return synapseclient.login(authToken=token, silent=True)
    # Falls back to ~/.synapseConfig / SYNAPSE_PROFILE / client-supported auth.
    return synapseclient.login(silent=True)


def download_dataset(syn, dataset: str, root: Path, overwrite: bool) -> None:
    rows = load_rows(dataset)
    target_root = root / dataset
    print(f"[{dataset}] {len(rows)} files -> {target_root}")
    for i, row in enumerate(rows, 1):
        target = target_root / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            if i % 100 == 0 or i == len(rows):
                print(f"[{dataset}] checked {i}/{len(rows)}")
            continue
        if target.exists():
            target.unlink()
        entity = syn.get(row["synapse_id"], downloadLocation=str(target.parent))
        downloaded = Path(entity.path)
        if downloaded.resolve() != target.resolve():
            if target.exists(): target.unlink()
            shutil.move(str(downloaded), str(target))
        if target.name != row["expected_name"]:
            raise RuntimeError(f"Unexpected target name: {target}")
        if i % 25 == 0 or i == len(rows):
            print(f"[{dataset}] downloaded {i}/{len(rows)}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["eventsleep1","eventsleep2","all"], default="all")
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--overwrite", action="store_true")
    args=p.parse_args()
    syn=login()
    datasets=["eventsleep1","eventsleep2"] if args.dataset=="all" else [args.dataset]
    for dataset in datasets:
        download_dataset(syn,dataset,args.output_root.resolve(),args.overwrite)
    print("Download complete.")

if __name__ == "__main__": main()
