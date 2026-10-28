#!/usr/bin/env python3
"""Teacher-only parts of the pre-registered secondary analyses (preregistration_v5.md §6 item 3), run on CPU
while S4 occupies the GPU. Uses only cached teacher features (CLS, correct preprocessing), the same probe,
split and cluster definitions as run_probes.py / teacher_reference.py. No student is involved and nothing
here feeds a verdict.

Per task (c16, msi, center, crc7k) and teacher:
  teacher_metric.csv   metric, cluster and tile bootstrap SE, inflation = SE_cluster / SE_tile   (proxy for H1 on the teacher metric)
  icc.csv              ICC(1) of the teacher's correct-class probability within clusters, design effect   (item 3g, exact)
  ranking_resolution.csv  teacher pairs significant under tile vs cluster 90% interval                   (item 3e teacher part, exact)
  ranking_pairs.csv    every teacher pair: difference, tile CI, cluster CI
  cluster_count_curve_teacher.csv  CI half-width of the teacher metric vs number of clusters (cluster vs tile)   (proxy for item 3d)
Predictions are cached in <out>/preds/<task>__<teacher>.npz for the later student-side analyses.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_probes import FEAT_ROOT, TASKS, load_feats, metric, split_for  # noqa: E402
from teacher_reference import oof_predictions  # noqa: E402
from secondary_analyses import _icc1, _boot_diff, write_csv  # noqa: E402

TASK_LIST = ["c16", "msi", "center", "crc7k"]


def task_data(task):
    spec = TASKS[task]; ds = spec["dataset"]; ds_dir = FEAT_ROOT / ds
    lab = dict(np.load(ds_dir / "labels.npz", allow_pickle=True)); y = lab["labels"]
    if task == "center":
        y = np.unique(lab["subgroup"], return_inverse=True)[1]
    clusters = np.asarray(lab["groups"]).astype(str)
    return spec, ds, ds_dir, lab, y, clusters


def boot_metric(y, p, groups, classes, kind, n_boot, seed, cluster):
    rng = np.random.default_rng(seed); vals = []
    if cluster:
        uniq = np.unique(groups); by = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        idx = (np.concatenate([by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))]) if cluster
               else rng.integers(0, len(y), len(y)))
        v = metric(y[idx], p[idx], classes, kind)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) < 50:
        return np.nan, np.nan, np.nan
    return float(vals.std(ddof=1)), float(np.quantile(vals, .05)), float(np.quantile(vals, .95))


def one_teacher(task, teacher, out, n_boot, n_rep, n_boot_curve, seed):
    t0 = time.time()
    spec, ds, ds_dir, lab, y, clusters = task_data(task)
    kind = spec["metric"]; folds = split_for(task, lab, seed); classes = np.unique(y)
    feat, meta = load_feats(ds, f"teacher__{teacher}")
    p, te, per_fold = oof_predictions(feat, y, folds, classes, seed)
    yt, pt, ct = y[te], p[te], clusters[te]
    pred_dir = out / "preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"{task}__{teacher}.npz", te_idx=te, y=yt, pt=pt, clusters=ct, classes=classes)
    has_cluster = len(np.unique(ct)) < len(ct)
    m = metric(yt, pt, classes, kind)
    se_n, lo_n, hi_n = boot_metric(yt, pt, ct, classes, kind, n_boot, seed + 1, False)
    se_c, lo_c, hi_c = boot_metric(yt, pt, ct, classes, kind, n_boot, seed, True) if has_cluster else (se_n, lo_n, hi_n)
    row = {"task": task, "teacher": teacher, "metric": m, "n_test": int(len(yt)), "n_clusters": int(len(np.unique(ct))),
           "se_tile": se_n, "ci90_tile_lo": lo_n, "ci90_tile_hi": hi_n, "se_cluster": se_c, "ci90_cluster_lo": lo_c,
           "ci90_cluster_hi": hi_c, "inflation": (se_c / se_n if se_n and se_n > 0 else np.nan)}
    # ICC(1) of the correct-class probability
    p_true = pt[np.arange(len(yt)), [list(classes).index(c) for c in yt]].astype(np.float64)
    icc, n0 = _icc1(p_true, ct) if has_cluster else (np.nan, np.nan)
    icc_row = {"task": task, "teacher": teacher, "n_clusters": row["n_clusters"], "n_tiles": int(len(yt)), "icc1": icc,
               "mean_cluster_size": n0, "design_effect": 1 + (n0 - 1) * icc if np.isfinite(icc) else np.nan}
    # cluster-count curve of the teacher metric
    curve = []
    if has_cluster:
        uniq = np.unique(ct); K = len(uniq); by = {g: np.flatnonzero(ct == g) for g in uniq}
        grid = sorted({k for k in (10, 20, 30, 50, 75, 100, 150, 200, 300, K) if k <= K})
        rng = np.random.default_rng(seed)
        for k in grid:
            hw = {"cluster": [], "tile": []}
            for _ in range(n_rep if k < K else 1):
                sel = rng.choice(uniq, k, replace=False) if k < K else uniq
                idx = np.concatenate([by[g] for g in sel])
                for mode in hw:
                    s, lo, hi = boot_metric(yt[idx], pt[idx], ct[idx], classes, kind, n_boot_curve, int(rng.integers(1 << 30)), mode == "cluster")
                    if np.isfinite(lo):
                        hw[mode].append((hi - lo) / 2)
            curve.append({"task": task, "teacher": teacher, "n_clusters": k, "n_reps": len(hw["cluster"]),
                          "half_width_cluster": float(np.mean(hw["cluster"])) if hw["cluster"] else np.nan,
                          "half_width_tile": float(np.mean(hw["tile"])) if hw["tile"] else np.nan})
    print(f"[{task}] {teacher:14s} {kind}={m:.4f} SE tile={se_n:.4f} cluster={se_c:.4f} infl={row['inflation']:.2f} ICC={icc:.3f} {time.time()-t0:.0f}s", flush=True)
    return row, icc_row, curve


def ranking(task, out, n_boot, seed):
    spec = TASKS[task]; kind = spec["metric"]
    files = sorted((out / "preds").glob(f"{task}__*.npz"))
    preds = {f.stem.split("__", 1)[1]: np.load(f, allow_pickle=True) for f in files}
    pair_rows = []; sig = {"tile": 0, "cluster": 0}
    for a, b in combinations(sorted(preds), 2):
        za, zb = preds[a], preds[b]
        if not np.array_equal(za["te_idx"], zb["te_idx"]):
            continue
        y, cl, classes = za["y"], za["clusters"].astype(str), list(za["classes"])
        has_cluster = len(np.unique(cl)) < len(cl)
        d = metric(y, za["pt"], classes, kind) - metric(y, zb["pt"], classes, kind)
        lo_n, hi_n = _boot_diff(y, za["pt"], zb["pt"], cl, classes, kind, n_boot, seed, False)
        lo_c, hi_c = _boot_diff(y, za["pt"], zb["pt"], cl, classes, kind, n_boot, seed, True) if has_cluster else (lo_n, hi_n)
        s_n = bool(np.isfinite(lo_n) and (lo_n > 0 or hi_n < 0)); s_c = bool(np.isfinite(lo_c) and (lo_c > 0 or hi_c < 0))
        sig["tile"] += s_n; sig["cluster"] += s_c
        pair_rows.append({"task": task, "teacher_a": a, "teacher_b": b, "d": d, "ci90_tile_lo": lo_n, "ci90_tile_hi": hi_n,
                          "ci90_cluster_lo": lo_c, "ci90_cluster_hi": hi_c, "sig_tile": s_n, "sig_cluster": s_c})
    res = {"task": task, "n_teachers": len(preds), "n_teacher_pairs": len(pair_rows),
           "significant_pairs_tile": sig["tile"], "significant_pairs_cluster": sig["cluster"]}
    print(f"[rank] {task}: {res}", flush=True)
    return res, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/v5/teacher_side_secondary")
    ap.add_argument("--tasks", nargs="+", default=TASK_LIST)
    ap.add_argument("--teachers", nargs="+", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_rep", type=int, default=20)
    ap.add_argument("--n_boot_curve", type=int, default=200)
    ap.add_argument("--n_boot_rank", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = []
    for task in a.tasks:
        ds_dir = FEAT_ROOT / TASKS[task]["dataset"]
        teachers = a.teachers or sorted(d.name.split("__", 1)[1] for d in ds_dir.iterdir()
                                        if d.is_dir() and d.name.startswith("teacher__") and (d / "meta.json").exists())
        jobs += [(task, t) for t in teachers]
    rows, icc_rows, curve_rows = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one_teacher, task, t, out, a.n_boot, a.n_rep, a.n_boot_curve, a.seed): (task, t) for task, t in jobs}
        for f in as_completed(futs):
            r, ir, cv = f.result(); rows.append(r); icc_rows.append(ir); curve_rows += cv
    key = lambda r: (a.tasks.index(r["task"]), r["teacher"])
    write_csv(out / "teacher_metric.csv", sorted(rows, key=key)); write_csv(out / "icc.csv", sorted(icc_rows, key=key))
    write_csv(out / "cluster_count_curve_teacher.csv", sorted(curve_rows, key=lambda r: (a.tasks.index(r["task"]), r["teacher"], r["n_clusters"])))
    res_rows, pair_rows = [], []
    with ProcessPoolExecutor(min(a.workers, len(a.tasks))) as ex:
        for res, pr in ex.map(ranking, a.tasks, [out] * len(a.tasks), [a.n_boot_rank] * len(a.tasks), [a.seed] * len(a.tasks)):
            res_rows.append(res); pair_rows += pr
    write_csv(out / "ranking_resolution.csv", res_rows); write_csv(out / "ranking_pairs.csv", pair_rows)
    json.dump({"tasks": a.tasks, "n_boot": a.n_boot, "n_rep": a.n_rep, "n_boot_curve": a.n_boot_curve, "n_boot_rank": a.n_boot_rank,
               "seed": a.seed, "seconds": round(time.time() - t0, 1), "n_jobs": len(jobs)}, open(out / "meta.json", "w"), indent=1)
    print(f"[done] {len(jobs)} (task, teacher) jobs in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
