#!/usr/bin/env python3
"""Additional teacher-only, CPU-only descriptive analyses (preregistration_v5.md §6, 2026-10-19 entry).
Uses the cached OOF teacher predictions of teacher_side_secondary.py and the cached teacher features; no student,
no verdict. All outputs are descriptive.

  teacher_pair_equivalence.csv  of the 66 teacher pairs per task, how many are equivalent / inequivalent / inconclusive
                                at margins 0.01-0.10 under tile vs slide-cluster 90% intervals (from ranking_pairs.csv)
  cluster_size_sensitivity.csv  teacher metric on all tiles vs slide-balanced subsamples (<= median tiles per slide);
                                Spearman between slide size and the slide's mean correct-class probability (informative cluster size)
  probe_sensitivity.csv         teacher OOF metric with logistic C in {0.1, 1, 10} and the Kendall tau of the teacher ranking vs C=1
"""
import argparse, csv, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_probes import FEAT_ROOT, TASKS, load_feats, metric, probe, split_for  # noqa: E402
from secondary_analyses import write_csv  # noqa: E402
from teacher_side_secondary import task_data, boot_metric  # noqa: E402

MARGINS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10]


def pair_equivalence(pairs_csv):
    rows = list(csv.DictReader(open(pairs_csv)))
    out = []
    for task in sorted({r["task"] for r in rows}, key=lambda t: ["c16", "msi", "center", "crc7k"].index(t)):
        rs = [r for r in rows if r["task"] == task]
        for m in MARGINS:
            rec = {"task": task, "margin": m, "n_pairs": len(rs)}
            for mode in ("tile", "cluster"):
                lo = np.array([float(r[f"ci90_{mode}_lo"]) for r in rs]); hi = np.array([float(r[f"ci90_{mode}_hi"]) for r in rs])
                eq = (lo > -m) & (hi < m); ineq = (hi < -m) | (lo > m)
                rec[f"equivalent_{mode}"] = int(eq.sum()); rec[f"inequivalent_{mode}"] = int(ineq.sum())
                rec[f"inconclusive_{mode}"] = int((~eq & ~ineq).sum())
            out.append(rec)
    return out


def cluster_size(task, teacher, preds_dir, n_rep, seed):
    z = np.load(preds_dir / f"{task}__{teacher}.npz", allow_pickle=True)
    y, pt, cl, classes = z["y"], z["pt"], z["clusters"].astype(str), list(z["classes"]); kind = TASKS[task]["metric"]
    uniq, inv, cnt = np.unique(cl, return_inverse=True, return_counts=True)
    full = metric(y, pt, classes, kind)
    rec = {"task": task, "teacher": teacher, "n_clusters": int(len(uniq)), "size_min": int(cnt.min()), "size_median": float(np.median(cnt)),
           "size_max": int(cnt.max()), "metric_all_tiles": full}
    if len(uniq) < len(cl):
        cap = int(np.median(cnt)); rng = np.random.default_rng(seed); by = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        vals = []
        for _ in range(n_rep):
            idx = np.concatenate([rng.choice(b, min(len(b), cap), replace=False) for b in by])
            v = metric(y[idx], pt[idx], classes, kind)
            if np.isfinite(v):
                vals.append(v)
        p_true = pt[np.arange(len(y)), [classes.index(c) for c in y]]
        mean_p = np.bincount(inv, weights=p_true) / cnt
        from scipy.stats import spearmanr
        rho, pval = spearmanr(cnt, mean_p)
        rec.update({"cap_tiles_per_slide": cap, "metric_slide_balanced_mean": float(np.mean(vals)), "metric_slide_balanced_sd": float(np.std(vals, ddof=1)),
                    "delta_balanced_minus_all": float(np.mean(vals) - full), "spearman_size_vs_mean_ptrue": float(rho), "spearman_p": float(pval)})
    else:
        rec.update({"cap_tiles_per_slide": np.nan, "metric_slide_balanced_mean": np.nan, "metric_slide_balanced_sd": np.nan,
                    "delta_balanced_minus_all": np.nan, "spearman_size_vs_mean_ptrue": np.nan, "spearman_p": np.nan})
    return rec


def probe_c(task, teacher, C, n_boot, seed):
    t0 = time.time()
    spec, ds, ds_dir, lab, y, clusters = task_data(task); kind = spec["metric"]; folds = split_for(task, lab, seed); classes = np.unique(y)
    feat, meta = load_feats(ds, f"teacher__{teacher}")
    p_all = np.zeros((len(y), len(classes))); covered = np.zeros(len(y), bool)
    for tr, te, name in folds:
        p, cl = probe(feat[tr], y[tr], feat[te], seed, C=C)
        for j, c in enumerate(cl):
            p_all[te, list(classes).index(c)] = p[:, j]
        covered[te] = True
    te = np.flatnonzero(covered); yt, pt, ct = y[te], p_all[te], clusters[te]
    m = metric(yt, pt, classes, kind)
    has_cluster = len(np.unique(ct)) < len(ct)
    se, lo, hi = boot_metric(yt, pt, ct, classes, kind, n_boot, seed, has_cluster)
    print(f"[probe] {task} {teacher:14s} C={C:<4} {kind}={m:.4f} [{lo:.4f},{hi:.4f}] {time.time()-t0:.0f}s", flush=True)
    return {"task": task, "teacher": teacher, "C": C, "metric": m, "ci90_lo": lo, "ci90_hi": hi, "se_cluster": se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v5/teacher_side_extra")
    ap.add_argument("--preds", default="outputs/v5/teacher_side_secondary/preds")
    ap.add_argument("--pairs", default="research/teacher_side_secondary_v5/ranking_pairs.csv")
    ap.add_argument("--tasks", nargs="+", default=["c16", "msi", "center", "crc7k"])
    ap.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--n_rep", type=int, default=200)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); preds = Path(a.preds); t0 = time.time()
    write_csv(out / "teacher_pair_equivalence.csv", pair_equivalence(a.pairs)); print("[A] pair equivalence written", flush=True)
    teachers = {t: sorted(f.stem.split("__", 1)[1] for f in preds.glob(f"{t}__*.npz")) for t in a.tasks}
    rows = [cluster_size(t, tc, preds, a.n_rep, a.seed) for t in a.tasks for tc in teachers[t]]
    write_csv(out / "cluster_size_sensitivity.csv", rows); print("[B] cluster-size sensitivity written", flush=True)
    jobs = [(t, tc, C) for t in a.tasks for tc in teachers[t] for C in a.Cs]
    with ProcessPoolExecutor(a.workers) as ex:
        prow = list(ex.map(probe_c, [j[0] for j in jobs], [j[1] for j in jobs], [j[2] for j in jobs], [a.n_boot] * len(jobs), [a.seed] * len(jobs)))
    from scipy.stats import kendalltau
    taus = []
    for t in a.tasks:
        base = {r["teacher"]: r["metric"] for r in prow if r["task"] == t and r["C"] == 1.0}
        for C in a.Cs:
            alt = {r["teacher"]: r["metric"] for r in prow if r["task"] == t and r["C"] == C}
            ks = sorted(base); tau = kendalltau([base[k] for k in ks], [alt[k] for k in ks])[0]
            taus.append({"task": t, "C": C, "kendall_tau_vs_C1": float(tau), "n_teachers": len(ks),
                         "metric_median": float(np.median([alt[k] for k in ks]))})
    write_csv(out / "probe_sensitivity.csv", prow); write_csv(out / "probe_sensitivity_ranking.csv", taus)
    json.dump({"tasks": a.tasks, "Cs": a.Cs, "n_rep": a.n_rep, "n_boot": a.n_boot, "seed": a.seed, "seconds": round(time.time() - t0, 1)},
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
