\
#!/usr/bin/env python3
"""Train/evaluate the exact five-seed EventSleep1 configuration reported in the paper."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import eventsleepformer as core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preprocess-root',type=Path,default=ROOT/'data/processed/eventsleep1')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/eventsleep1')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    manifest=a.preprocess_root/'eventsleep1/temporal_sequences/manifest.csv'
    if not manifest.is_file():
        raise SystemExit('Missing EventSleep1 temporal cache. Run scripts/preprocess_eventsleep1.py first.')
    cli=['train','--preprocess-root',str(a.preprocess_root),'--output',str(a.output),
         '--dataset','eventsleep1','--task','activity','--locked-config',str(ROOT/'configs/eventsleep1.json'),
         '--seeds','13','37','73','101','137','--device',a.device,'--workers',str(a.workers)]
    if a.overwrite: cli.append('--overwrite')
    args=core.build_parser().parse_args(cli)
    core.configure_logging(a.output.resolve()/'logs/train.log',False)
    results=core.run_training_grid(args,core.DataPaths(a.preprocess_root.resolve()),variants=['full'])
    print(results.to_string(index=False))

if __name__=='__main__': main()
