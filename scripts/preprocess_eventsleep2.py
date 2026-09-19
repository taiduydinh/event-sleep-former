\
#!/usr/bin/env python3
"""Build the EventSleep2 paper cache and timestamp-aligned claim dataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import eventsleep2
PAPER_AUDIT_FP='7e27838c43449abcb72c06641031e46cb2b2140581c75a6cc6f55f68fedb1efa'
EXPECTED_SOURCE_FP='0339ef7ad3c0e622e584762cc5a34e0679f26e7e9ffd2b6de511fec775e4905c'
EXPECTED_LABELS_SHA='094f5ac69ae89680e81f48f901e43b4a5cbe6664f041be186c01e32ab5cc9f17'


def run(argv):
    rc=eventsleep2.main(argv)
    if rc: raise SystemExit(rc)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=ROOT/'data/raw/eventsleep2')
    p.add_argument('--cache-root',type=Path,default=ROOT/'data/processed/eventsleep2_published')
    p.add_argument('--timestamp-root',type=Path,default=ROOT/'data/processed/eventsleep2_timestamp')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); raw=a.raw_root.resolve(); cache=a.cache_root.resolve(); ts=a.timestamp_root.resolve()
    stage=ROOT/'outputs/preprocess_eventsleep2'; stage.mkdir(parents=True,exist_ok=True)
    argv=['prepare-published','--eventsleep2-root',str(raw),'--preprocess-root',str(cache),
          '--output',str(stage),'--profile','paper_declared']
    if a.overwrite: argv.append('--overwrite-cache')
    run(argv)
    run(['prepare-timestamp-aligned','--preprocess-root',str(cache),'--output',str(ts),
         '--profile','paper_declared'])
    marker=json.loads((ts/'timestamp_dataset.json').read_text())
    if marker['source_dataset_fingerprint'] != EXPECTED_SOURCE_FP:
        raise RuntimeError('EventSleep2 source-cache fingerprint mismatch: '+marker['source_dataset_fingerprint'])
    if marker['labels_sha256'] != EXPECTED_LABELS_SHA:
        raise RuntimeError('EventSleep2 label-file SHA-256 mismatch: '+marker['labels_sha256'])
    if marker['samples'] != {'train':5153,'validation':1055,'test':10536}:
        raise RuntimeError('Unexpected timestamp-aligned sample counts: '+repr(marker['samples']))
    print(json.dumps({'timestamp_dataset_fingerprint':marker['dataset_fingerprint'],
                      'paper_audit_fingerprint':PAPER_AUDIT_FP,
                      'source_dataset_fingerprint':marker['source_dataset_fingerprint'],
                      'labels_sha256':marker['labels_sha256'],
                      'samples':marker['samples']},indent=2))

if __name__=='__main__': main()
