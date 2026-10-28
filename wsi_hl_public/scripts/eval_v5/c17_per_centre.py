#!/usr/bin/env python3
"""Per-centre CAMELYON17-LOCO results from the per-fold metrics stored in each probe_c17_loco.json.
For every held-out centre: teacher AUROC range, and over the 36 teacher-student pairs (seed-mean d)
the median, min and max of d and the number of pairs with |d| < 0.05. Each centre also gets a paired slide-cluster interval and a verdict at 0.05. Writes outputs/v5/secondary/c17_per_centre.json."""
import glob, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np

recs = defaultdict(list)                       # (teacher, student) -> list of per-fold dicts per seed
for f in glob.glob("outputs/v5/*/*/seed*/probe_c17_loco.json"):
    p = Path(f).parent; r = json.load(open(f))
    recs[(p.parent.parent.name, p.parent.name)].append({x["fold"]: x for x in r["folds"]})
centres = sorted(next(iter(recs.values()))[0])
out = {}
for c in centres:
    t_m = {}; d = []
    for (t, s), seeds in recs.items():
        t_m[t] = seeds[0][c]["t_metric"]
        d.append(float(np.mean([x[c]["s_metric"] - x[c]["t_metric"] for x in seeds])))
    d = np.asarray(d); tv = np.asarray(list(t_m.values()))
    out[c] = {"n_test_tiles": next(iter(recs.values()))[0][c]["n_test"], "teacher_min": float(tv.min()),
              "teacher_max": float(tv.max()), "d_median": float(np.median(d)), "d_min": float(d.min()),
              "d_max": float(d.max()), "n_abs_d_below_margin": int((np.abs(d) < 0.05).sum()), "n_pairs": int(len(d))}
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[c].items()})
# per-centre paired slide-cluster interval (seed drawn per replicate, as in aggregate.py), verdict at 0.05
import sys
sys.path.insert(0, "scripts/eval_v5"); sys.path.insert(0, "scripts")
from aggregate import pooled_bootstrap  # noqa: E402
lab = dict(np.load(Path(os.environ.get("WSI_V5_ROOT", "/data/wsi_hl_v5")) / "features/camelyon17_annotated50/labels.npz", allow_pickle=True))
runs = defaultdict(list)
for f in glob.glob("outputs/v5/*/*/seed*/probe_c17_loco_preds.npz"):
    p = Path(f).parent; runs[(p.parent.parent.name, p.parent.name)].append(np.load(f, allow_pickle=True))
for c in centres:
    states, half = [], []
    for (t, s), zs in sorted(runs.items()):
        cen = np.asarray(lab["subgroup"])[zs[0]["te_idx"]]; m = cen == c
        r = pooled_bootstrap(zs[0]["y"][m], zs[0]["pt"][m], [z["ps"][m] for z in zs], zs[0]["clusters"][m],
                             zs[0]["classes"], "auroc", 0.05)
        states.append(r["state"]); half.append((r["hi"] - r["lo"]) / 2)
    out[c].update(n_equivalent=states.count("equivalent"), n_inconclusive=states.count("inconclusive"),
                  n_inequivalent=states.count("inequivalent"), halfwidth_median=float(np.median(half)),
                  halfwidth_max=float(np.max(half)))
    print(c, states.count("equivalent"), states.count("inconclusive"), states.count("inequivalent"),
          round(float(np.median(half)), 4), round(float(np.max(half)), 4))
Path("outputs/v5/secondary").mkdir(parents=True, exist_ok=True)
json.dump(out, open("outputs/v5/secondary/c17_per_centre.json", "w"), indent=1)
