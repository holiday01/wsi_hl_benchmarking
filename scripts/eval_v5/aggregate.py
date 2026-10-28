#!/usr/bin/env python3
"""Seed-pooled summary of the v5 probes (preregistration_v5.md section 3).

For every (task, teacher, student architecture) with k seeds:
  - per-seed d and state (copied from probe_<task>.json)
  - pooled d = mean over seeds of d
  - pooled 90% CI: cluster bootstrap in which each replicate resamples clusters with
    replacement and draws ONE seed's student predictions at random (seed and cluster
    uncertainty together); teacher predictions are the same across seeds
  - seed SD of d, cluster SE, and the pre-registered three-state decision at the margin
  - pannuke_seg: pooled over seeds from the per-image Dice arrays the same way
Then hierarchical BH (stats_v4.hierarchical_bh) over the pooled TOST p-values, teacher = family.
Outputs <out>/summary_<task>.csv/json and <out>/summary_all.json.
"""
import argparse, csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from run_probes import TASKS, metric  # noqa: E402
from stats_v4 import hierarchical_bh  # noqa: E402


def interval_from(dc, center, margin, interval="percentile"):
    """Verdict, 90% interval and TOST p from bootstrap replicates dc. percentile (pre-specified): quantiles of dc and
    uncentred tail fractions; basic (sensitivity): replicates centred on the estimate, 2*center - quantiles, and
    tail fractions of dc - center beyond center +/- margin (the test that inverts the basic interval)."""
    q05, q95 = float(np.quantile(dc, 0.05)), float(np.quantile(dc, 0.95)); n = len(dc)
    if interval == "basic":
        lo, hi = 2 * center - q95, 2 * center - q05; e = dc - center
        p_l = float((np.sum(e >= center + margin) + 1) / (n + 1)); p_u = float((np.sum(e <= center - margin) + 1) / (n + 1))
    else:
        lo, hi = q05, q95
        p_l = float((np.sum(dc <= -margin) + 1) / (n + 1)); p_u = float((np.sum(dc >= margin) + 1) / (n + 1))
    state = "equivalent" if (lo > -margin and hi < margin) else ("inequivalent" if (hi < -margin or lo > margin) else "inconclusive")
    return dict(lo=lo, hi=hi, state=state, p_lower=p_l, p_upper=p_u, p_tost=max(p_l, p_u))


def pooled_bootstrap(y, pt, ps_list, clusters, classes, kind, margin, n_boot=1000, seed=42, seed_mode="draw",
                     interval="percentile", keep=None):
    uniq = np.unique(clusters); by = {g: np.flatnonzero(clusters == g) for g in uniq}
    rng = np.random.default_rng(seed); dc = []
    for _ in range(n_boot):
        idx = np.concatenate([by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        if seed_mode == "mean":                 # sensitivity: interval for the mean of the seeds' d
            mt = metric(y[idx], pt[idx], classes, kind)
            v = float(np.mean([metric(y[idx], ps[idx], classes, kind) for ps in ps_list])) - mt
        else:
            ps = ps_list[rng.integers(0, len(ps_list))]
            v = metric(y[idx], ps[idx], classes, kind) - metric(y[idx], pt[idx], classes, kind)
        if np.isfinite(v):
            dc.append(v)
    if keep is not None:
        keep.append(np.asarray(dc, float))
    d_seeds = [metric(y, ps, classes, kind) - metric(y, pt, classes, kind) for ps in ps_list]
    out = {"d_pooled": float(np.mean(d_seeds)), "d_seed_sd": float(np.std(d_seeds, ddof=1)) if len(d_seeds) > 1 else 0.0,
           "n_seeds": len(ps_list), "n_clusters": int(len(uniq))}
    if len(dc) < 50:
        out.update(lo=float("nan"), hi=float("nan"), state="undefined", p_tost=float("nan")); return out
    dc = np.asarray(dc)
    out.update(interval_from(dc, out["d_pooled"], margin, interval), se_cluster_seed=float(dc.std(ddof=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v5")
    ap.add_argument("--out", default="outputs/v5/summary")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed_mode", default="draw", choices=["draw", "mean"],
                    help="draw: one seed per replicate (pre-specified); mean: mean of the seeds' d (sensitivity)")
    ap.add_argument("--interval", default="percentile", choices=["percentile", "basic"],
                    help="percentile (pre-specified) or basic, centred on the estimate (sensitivity)")
    ap.add_argument("--tasks", default=None, help="comma-separated subset of tasks")
    ap.add_argument("--save_replicates", default=None, help="also write every pair's bootstrap d replicates to this .npz")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = sorted(glob.glob(f"{args.root}/*/*/seed*"))
    groups = defaultdict(list)                                  # (teacher, student) -> run dirs
    for r in runs:
        p = Path(r); groups[(p.parent.parent.name, p.parent.name)].append(r)
    all_rows = {}; reps = []                                   # (task, teacher, student, replicates)
    for task in (args.tasks.split(",") if args.tasks else list(TASKS) + ["pannuke_seg"]):
        rows = []
        for (teacher, student), dirs in sorted(groups.items()):
            recs = [(d, json.load(open(f"{d}/probe_{task}.json"))) for d in dirs if Path(f"{d}/probe_{task}.json").exists()]
            if not recs:
                continue
            row = {"task": task, "teacher": teacher, "student": student, "seeds": [Path(d).name for d, _ in recs]}
            if task == "pannuke_seg":
                per_seed = [r["mDice_fg"] for _, r in recs]
                row["d_per_seed"] = [x["d"] for x in per_seed]; row["state_per_seed"] = [x["state"] for x in per_seed]
                dice = [np.load(f"{d}/probe_pannuke_seg_dice.npz") for d, _ in recs]
                t = np.nanmean(dice[0]["teacher"][:, :5], 1); s_list = [np.nanmean(z["student"][:, :5], 1) for z in dice]
                n = len(t); rng = np.random.default_rng(42); dc = []
                for _ in range(args.n_boot):
                    idx = rng.integers(0, n, n)
                    if args.seed_mode == "mean":
                        dc.append(float(np.mean([np.nanmean(s[idx] - t[idx]) for s in s_list])))
                    else:
                        s = s_list[rng.integers(0, len(s_list))]
                        dc.append(np.nanmean(s[idx] - t[idx]))
                dc = np.asarray(dc); reps.append((task, teacher, student, dc.astype(float))); d_pooled = float(np.mean([np.nanmean(s - t) for s in s_list]))
                row.update(interval_from(dc, d_pooled, 0.05, args.interval), d_pooled=d_pooled,
                           d_seed_sd=float(np.std([np.nanmean(s - t) for s in s_list], ddof=1)) if len(s_list) > 1 else 0.0,
                           n_seeds=len(s_list), t_metric=float(np.nanmean(t)), s_metric_mean=float(np.mean([np.nanmean(s) for s in s_list])))
            else:
                spec = TASKS[task]
                preds = [np.load(f"{d}/probe_{task}_preds.npz", allow_pickle=True) for d, _ in recs]
                y, pt, cl, classes = preds[0]["y"], preds[0]["pt"], preds[0]["clusters"], preds[0]["classes"]
                for z in preds[1:]:
                    assert np.array_equal(z["te_idx"], preds[0]["te_idx"]) and np.allclose(z["pt"], pt, atol=1e-5)
                keep = []
                res = pooled_bootstrap(y, pt, [z["ps"] for z in preds], cl, classes, spec["metric"], spec["margin"], args.n_boot,
                                       seed_mode=args.seed_mode, interval=args.interval, keep=keep)
                reps.append((task, teacher, student, keep[0]))
                row.update(res, d_per_seed=[r["result"]["d"] for _, r in recs], state_per_seed=[r["result"]["state"] for _, r in recs],
                           inflation_per_seed=[r["result"].get("inflation") for _, r in recs],
                           t_metric=recs[0][1]["result"]["t_metric"],
                           s_metric_mean=float(np.mean([r["result"]["s_metric"] for _, r in recs])))
            rows.append(row)
        if not rows:
            continue
        fam = defaultdict(list)
        for r in rows:
            fam[r["teacher"]].append(r["p_tost"])
        # stage-1 BH within each teacher family on the pooled TOST p-values (NaN -> 1)
        bh = hierarchical_bh({t: [1.0 if not np.isfinite(p) else p for p in v] for t, v in fam.items()}, alpha=0.05)
        counters = defaultdict(int)
        for r in rows:                      # rows are grouped by teacher in insertion order
            k = counters[r["teacher"]]; counters[r["teacher"]] += 1
            r["bh_reject"] = bool(bh["rejected"][r["teacher"]][k])
        json.dump(rows, open(out / f"summary_{task}.json", "w"), indent=1, default=float)
        keys = ["task", "teacher", "student", "n_seeds", "t_metric", "s_metric_mean", "d_pooled", "lo", "hi", "d_seed_sd", "state", "p_tost", "bh_reject"]
        with open(out / f"summary_{task}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        from collections import Counter
        print(f"[{task}] {len(rows)} pairs: {dict(Counter(r['state'] for r in rows))}", flush=True)
        all_rows[task] = rows
    json.dump(all_rows, open(out / "summary_all.json", "w"), indent=1, default=float)
    if args.save_replicates:
        # one row per pair, NaN-padded (replicates with an undefined metric are dropped, as in the interval)
        n = max(len(r[3]) for r in reps); D = np.full((len(reps), n), np.nan)
        for i, r in enumerate(reps):
            D[i, :len(r[3])] = r[3]
        np.savez_compressed(args.save_replicates, d_replicates=D, n_valid=np.array([len(r[3]) for r in reps]),
                            task=np.array([r[0] for r in reps]), teacher=np.array([r[1] for r in reps]),
                            student=np.array([r[2] for r in reps]), bootstrap_seed=42, n_boot=args.n_boot,
                            seed_mode=args.seed_mode, interval=args.interval)


if __name__ == "__main__":
    main()
