\
#!/usr/bin/env python3
"""Fast metadata/config consistency checks; does not require datasets or PyTorch."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_ADAPTER_SHA='fe67eb66e2e69a7a20fdcab9b80c9a46dc46356e1699a5d68925dfaa36b51eb5'
EXPECTED_ES2_FP='7e27838c43449abcb72c06641031e46cb2b2140581c75a6cc6f55f68fedb1efa'

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def rows(name):
    with (ROOT/'data/file_lists'/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def main():
    assert len(rows('eventsleep1.csv'))==995
    assert len(rows('eventsleep2.csv'))==11
    assert sha(ROOT/'src/eventsleep2_timestamp_protocol_v1.py')==EXPECTED_ADAPTER_SHA
    es1=json.loads((ROOT/'configs/eventsleep1.json').read_text()); assert es1['configuration_id']=='577106a0cce2'
    es2=json.loads((ROOT/'configs/eventsleep2.json').read_text()); assert es2['configuration_id']=='53ca6700c00a'; assert es2['dataset_fingerprint']==EXPECTED_ES2_FP
    print('Release metadata checks PASSED')
if __name__=='__main__': main()
