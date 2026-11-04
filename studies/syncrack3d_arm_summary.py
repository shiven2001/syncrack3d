"""Print one SynCrack3D arm's BD3, CUBIT and BuildCrack results against its baselines.

    python studies/syncrack3d_arm_summary.py <name>
"""

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
DT = APP / "outputs/domain_transfer"
sys.path.insert(0, str(APP / "eval"))
from metrics import centreline_scores  # noqa: E402

name = sys.argv[1]
for base in ("tilestone", "networkctl", "network"):
    r = json.loads((DT / f"syncrack3d_bd3_compare_{name}_vs_{base}.json").read_text())
    m = r["mean_of_seeds"][name]
    d = r.get("minus_baseline") or r["minus_v1.2.0"]
    d = d[name]
    print(f"BD3 vs {base:10} IoU {m['crack_iou']:.3f} ({d['crack_iou'][0]:+.3f} [{d['crack_iou'][1]:+.3f},{d['crack_iou'][2]:+.3f}])"
          f"  P {m['centreline_precision']:.3f} R {m['centreline_recall']:.3f}  plain FA {m['plain_false_alarm']:.3f}"
          f" ({d['plain_false_alarm'][0]:+.3f} [{d['plain_false_alarm'][1]:+.3f},{d['plain_false_alarm'][2]:+.3f}])")
for base in ("tilestone", "networkctl", "network"):
    c = json.loads((DT / f"cubit_compare_{name}_vs_{base}.json").read_text())["results"]["crack_photos_CR"]
    print(f"CUBIT vs {base:10} " + "  ".join(f"{k.split('_')[1] if k != 'centreline_f1_3px' else 'F1'} {c[k]['b']:.3f} "
                                         f"({c[k]['diff']:+.3f} [{c[k]['ci95'][0]:+.3f},{c[k]['ci95'][1]:+.3f}])"
                                         for k in ("centreline_f1_3px", "centreline_precision", "centreline_recall")))
r = json.loads((DT / f"cubit_eval_{name}_v1/results.json").read_text())["results"]
print(f"CUBIT spalling photos with false crack {r['spalling_photos_with_false_crack']:.3f}")
b = json.loads((DT / f"buildcrack_{name}_v1/results.json").read_text())
for m, v in b["models"].items():
    s = v["mean_of_seeds"]
    print(f"BuildCrack {m:12} F1 {s['centreline_f1_3px']:.3f} P {s['centreline_precision']:.3f} R {s['centreline_recall']:.3f}"
          f"  vote F1 {v['ensemble']['centreline_f1_3px']:.3f}  empty alarm {v['empty_tile_alarm_rate']:.3f}")
for k, v in b["differences"].items():
    if k.startswith(name):
        f, vf = v["mean_of_seeds"]["centreline_f1_3px"], v["ensemble"]["centreline_f1_3px"]
        print(f"BuildCrack {k:22} seed F1 {f['diff']:+.3f} [{f['ci95'][0]:+.3f},{f['ci95'][1]:+.3f}]  vote F1 {vf['diff']:+.3f} [{vf['ci95'][0]:+.3f},{vf['ci95'][1]:+.3f}]")
z = np.load(DT / f"buildcrack_{name}_v1/counts.npz")
dji = np.array([s.startswith("DJI") for s in z["stems"]])
for m in b["models"]:
    out = []
    for label, sel in (("drone", dji), ("other", ~dji)):
        sc = [centreline_scores(z[m][s][sel].sum(0)[3:7]) for s in range(z[m].shape[0])]
        out.append(f"{label} P {np.mean([x['precision'] for x in sc]):.3f} R {np.mean([x['recall'] for x in sc]):.3f} F1 {np.mean([x['f1'] for x in sc]):.3f}")
    print(f"BuildCrack {m:12} " + " | ".join(out))
