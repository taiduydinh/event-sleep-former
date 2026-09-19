\
#!/usr/bin/env python3
"""Build the EventSleep1 paper input from the downloaded 995 released clips."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import eventsleep_preprocess as prep
import eventsleepformer as model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=ROOT/'data/raw/eventsleep1')
    p.add_argument('--output',type=Path,default=ROOT/'data/processed/eventsleep1')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); out=a.output.resolve(); raw=a.raw_root.resolve()
    argv=['eventsleep1','--eventsleep1-root',str(raw),'--output',str(out),
          '--output-height','120','--output-width','160','--history-ms','512',
          '--storage-dtype','float16','--roi-profile','full','--workers',str(a.workers)]
    if a.overwrite: argv.append('--overwrite')
    rc=prep.main(argv)
    if rc: raise SystemExit(rc)
    report=prep.verify_preprocessed(out,require_both=False)
    if report.get('eventsleep1',{}).get('samples') != 995:
        raise RuntimeError(f"Expected 995 EventSleep1 clips, found {report.get('eventsleep1')}")
    paths=model.DataPaths(out)
    seq=model.build_es1_sequences(paths,sequence_length=32,history_us=512_000,
                                  workers=max(1,a.workers),overwrite=a.overwrite)
    if len(seq)!=995: raise RuntimeError(f'Expected 995 temporal sequences, found {len(seq)}')
    summary={'raw_clips':995,'temporal_sequences':995,'sequence_length':32,
             'shape_per_step':[2,120,160],'history_us':512000}
    (out/'eventsleep1_paper_preprocess.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
