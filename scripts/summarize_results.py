\
#!/usr/bin/env python3
"""Summarize newly reproduced results and compare them with paper reference values."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def mean_std(xs): return sum(xs)/len(xs), statistics.stdev(xs) if len(xs)>1 else float('nan')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eventsleep1-output',type=Path,default=ROOT/'outputs/eventsleep1')
    p.add_argument('--eventsleep2-output',type=Path,default=ROOT/'outputs/eventsleep2')
    a=p.parse_args()
    print('Paper reference:')
    ref=json.loads((ROOT/'results/paper_metrics.json').read_text())
    print(json.dumps(ref,indent=2))
    es1=[]
    for f in a.eventsleep1_output.rglob('metrics.json'):
        try: d=json.loads(f.read_text())
        except Exception: continue
        if d.get('dataset')=='eventsleep1' and d.get('method')=='eventsleepformer' and d.get('official_test_evaluated') is True:
            es1.append(d)
    if es1:
        print('\nReproduced EventSleep1:')
        for key in ['accuracy','macro_f1','balanced_accuracy']:
            m,s=mean_std([float(x[key]) for x in es1]); print(f'{key}: {m:.6f} ± {s:.6f} (n={len(es1)})')
    finals=list(a.eventsleep2_output.rglob('final_summary.json'))
    if finals:
        d=json.loads(sorted(finals)[-1].read_text())
        print('\nReproduced EventSleep2 ensemble:')
        h=d['headline']
        print(f"movement macro-F1: {h['motion_per_frame_macro_f1']:.6f}")
        print(f"posture macro-F1:  {h['static_per_frame_macro_f1']:.6f}")
        print(f"joint macro-F1:    {h['joint_macro_f1']:.6f}")

if __name__=='__main__': main()
