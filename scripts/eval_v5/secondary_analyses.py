#!/usr/bin/env python3
"""Secondary descriptive analyses registered in research/preregistration_v5.md §6 (2026-10-17).

They are computed from the S7/S8 outputs only and change no primary decision:
  delta_min     per pair, delta_min = max(|lo|, |hi|) of the pooled 90% CI (and per seed)
  margin_sweep  fraction of equivalent pairs per task as the margin varies over [0.01, 0.10]
  three_state   per task: equivalent / inequivalent / inconclusive counts with |d| and CI half-width
  headroom      d and margin relative to the teacher's headroom (1 - metric_T); margin vs
                between-teacher dispersion of metric_T
  icc           ICC(1) of the teacher's correct-class probability within cluster, design effect
  calibration   simulated false-equivalence rate at true d = +margin with naive vs cluster SE
  ranking       pairwise teacher differences significant under naive vs cluster 90% CI;
                Kendall tau between teacher and student rankings
  cluster_curve CI half-width of d versus number of clusters (cluster vs naive bootstrap)

Usage: secondary_analyses.py --root outputs/v5 --summary outputs/v5/summary --out outputs/v5/secondary
       [--analyses delta_min,margin_sweep,...] [--curve_pairs 12] [--curve_boot 200]
"""
import argparse, csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_probes import TASKS, metric  # noqa: E402

ALL = ["delta_min", "margin_sweep", "three_state", "headroom", "icc", "calibration", "ranking", "cluster_curve"]


def load_summary(summary):
    rows = {}
    for f in glob.glob(f"{summary}/summary_*.json"):
        task = Path(f).stem[len("summary_"):]
        if task == "all":
            continue
        rows[task] = json.load(open(f))
    return rows


def run_records(root):
    """(task, teacher, student, seed_dir) -> probe JSON for every per-run probe file."""
    out = {}
    for f in glob.glob(f"{root}/*/*/seed*/probe_*.json"):
        p = Path(f); task = p.stem[len("probe_"):]
        if task.endswith("_preds"):
            continue
        out[(task, p.parent.parent.parent.name, p.parent.parent.name, p.parent.name)] = json.load(open(f))
    return out


def write_csv(path, rows, keys=None):
    if not rows:
        return
    keys = keys or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)


def a_delta_min(S, R, out):
    rows = []
    for task, rs in S.items():
        for r in rs:
            if not (np.isfinite(r.get("lo", np.nan)) and np.isfinite(r.get("hi", np.nan))):
                continue
            rows.append({"task": task, "teacher": r["teacher"], "student": r["student"], "d_pooled": r["d_pooled"],
                         "lo": r["lo"], "hi": r["hi"], "delta_min": max(abs(r["lo"]), abs(r["hi"])),
                         "half_width": (r["hi"] - r["lo"]) / 2, "state": r["state"]})
    per_seed = []
    for (task, t, s, seed), rec in R.items():
        res = rec.get("result") or rec.get("mDice_fg")
        if res and np.isfinite(res.get("lo", np.nan)):
            per_seed.append({"task": task, "teacher": t, "student": s, "seed": seed, "d": res["d"],
                             "delta_min": max(abs(res["lo"]), abs(res["hi"])), "state": res["state"]})
    write_csv(out / "delta_min_pooled.csv", rows); write_csv(out / "delta_min_per_seed.csv", per_seed)
    return {"pooled": rows, "per_seed": per_seed}


def a_margin_sweep(S, R, out):
    grid = np.round(np.arange(0.01, 0.1001, 0.005), 3); rows = []
    for task, rs in S.items():
        ok = [r for r in rs if np.isfinite(r.get("lo", np.nan))]
        for m in grid:
            eq = sum(1 for r in ok if r["lo"] > -m and r["hi"] < m)
            ineq = sum(1 for r in ok if r["hi"] < -m or r["lo"] > m)
            rows.append({"task": task, "margin": float(m), "n_pairs": len(ok), "equivalent": eq, "inequivalent": ineq,
                         "inconclusive": len(ok) - eq - ineq, "frac_equivalent": eq / len(ok) if ok else np.nan})
    write_csv(out / "margin_sweep.csv", rows)
    return rows


def a_three_state(S, R, out):
    rows = []
    for task, rs in S.items():
        ok = [r for r in rs if np.isfinite(r.get("lo", np.nan))]
        c = defaultdict(int)
        for r in ok:
            c[r["state"]] += 1
        absd = [abs(r["d_pooled"]) for r in ok]; hw = [(r["hi"] - r["lo"]) / 2 for r in ok]
        rows.append({"task": task, "n_pairs": len(ok), **{k: c[k] for k in ("equivalent", "inequivalent", "inconclusive")},
                     "abs_d_median": float(np.median(absd)) if absd else np.nan, "abs_d_max": float(np.max(absd)) if absd else np.nan,
                     "half_width_median": float(np.median(hw)) if hw else np.nan,
                     "inconclusive_with_small_d": sum(1 for r in ok if r["state"] == "inconclusive" and abs(r["d_pooled"]) < 0.02)})
    write_csv(out / "three_state.csv", rows)
    return rows


def a_headroom(S, R, out):
    rows, disp = [], []
    for task, rs in S.items():
        tm = {r["teacher"]: r["t_metric"] for r in rs}
        if tm:
            v = np.array(list(tm.values()))
            disp.append({"task": task, "n_teachers": len(v), "t_metric_mean": float(v.mean()), "t_metric_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                         "t_metric_range": float(v.max() - v.min()), "margin": rs[0].get("margin", 0.05) if "margin" in rs[0] else 0.05,
                         "margin_over_range": (0.05 / (v.max() - v.min())) if v.max() > v.min() else np.inf})
        for r in rs:
            h = 1.0 - r["t_metric"]
            rows.append({"task": task, "teacher": r["teacher"], "student": r["student"], "t_metric": r["t_metric"], "headroom": h,
                         "d_pooled": r["d_pooled"], "d_over_headroom": r["d_pooled"] / h if h > 0 else np.nan,
                         "margin_over_headroom": 0.05 / h if h > 0 else np.nan, "state": r["state"]})
    write_csv(out / "headroom_pairs.csv", rows); write_csv(out / "headroom_teacher_dispersion.csv", disp)
    return {"pairs": rows, "dispersion": disp}


def _icc1(x, g):
    """One-way random-effects ICC(1) from the ANOVA estimator."""
    uniq, inv = np.unique(g, return_inverse=True); k = len(uniq)
    n_i = np.bincount(inv); N = len(x)
    if k < 2 or N <= k:
        return np.nan, np.nan
    m = x.mean(); mi = np.bincount(inv, weights=x) / n_i
    ssb = float(np.sum(n_i * (mi - m) ** 2)); ssw = float(np.sum((x - mi[inv]) ** 2))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(icc), float(n0)


def a_icc(S, R, out):
    rows = []; seen = set()
    for (task, t, s, seed), rec in R.items():
        if task == "pannuke_seg" or (task, t) in seen:
            continue
        f = Path(glob.glob(f"{args_root}/{t}/{s}/{seed}/probe_{task}_preds.npz")[0]) if glob.glob(f"{args_root}/{t}/{s}/{seed}/probe_{task}_preds.npz") else None
        if f is None:
            continue
        z = np.load(f, allow_pickle=True); y, pt, cl, classes = z["y"], z["pt"], z["clusters"].astype(str), list(z["classes"])
        p_true = pt[np.arange(len(y)), [classes.index(c) for c in y]]
        icc, n0 = _icc1(p_true.astype(np.float64), cl)
        rows.append({"task": task, "teacher": t, "n_clusters": int(len(np.unique(cl))), "n_tiles": int(len(y)), "icc1": icc,
                     "mean_cluster_size": n0, "design_effect": 1 + (n0 - 1) * icc if np.isfinite(icc) else np.nan})
        seen.add((task, t))
    write_csv(out / "icc.csv", rows)
    return rows


def a_calibration(S, R, out, n_sim=200000, seed=42):
    rng = np.random.default_rng(seed); rows = []
    by = defaultdict(lambda: {"c": [], "n": []})
    for (task, t, s, sd), rec in R.items():
        res = rec.get("result")
        if res and np.isfinite(res.get("se_cluster", np.nan)) and np.isfinite(res.get("se_naive", np.nan)):
            by[task]["c"].append(res["se_cluster"]); by[task]["n"].append(res["se_naive"])
    for task, v in by.items():
        se_c, se_n, m = float(np.median(v["c"])), float(np.median(v["n"])), 0.05
        for true_d in (0.0, m / 2, m):
            dh = rng.normal(true_d, se_c, n_sim)           # the truth has cluster-level noise
            fe_naive = float(np.mean(np.abs(dh) + 1.645 * se_n < m))
            fe_clust = float(np.mean(np.abs(dh) + 1.645 * se_c < m))
            rows.append({"task": task, "true_d": true_d, "se_cluster_median": se_c, "se_naive_median": se_n, "n_runs": len(v["c"]),
                         "equiv_rate_naive": fe_naive, "equiv_rate_cluster": fe_clust})
    write_csv(out / "calibration_simulation.csv", rows)
    return rows


def _boot_diff(y, pa, pb, groups, classes, kind, n_boot, seed, cluster):
    rng = np.random.default_rng(seed); d = []
    if cluster:
        uniq = np.unique(groups); by = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        idx = (np.concatenate([by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))]) if cluster
               else rng.integers(0, len(y), len(y)))
        v = metric(y[idx], pa[idx], classes, kind) - metric(y[idx], pb[idx], classes, kind)
        if np.isfinite(v):
            d.append(v)
    d = np.asarray(d)
    return (float(np.quantile(d, .05)), float(np.quantile(d, .95))) if len(d) >= 50 else (np.nan, np.nan)


def a_ranking(S, R, out, n_boot=500):
    from itertools import combinations
    rows, tau_rows = [], []
    for task, rs in S.items():
        if task == "pannuke_seg":
            continue
        # one preds file per teacher (teacher predictions are identical across its runs)
        preds = {}
        for r in rs:
            fs = glob.glob(f"{args_root}/{r['teacher']}/{r['student']}/seed*/probe_{task}_preds.npz")
            if fs and r["teacher"] not in preds:
                preds[r["teacher"]] = np.load(fs[0], allow_pickle=True)
        kind = TASKS[task]["metric"]
        sig = {"naive": 0, "cluster": 0}; n_pairs = 0
        for a, b in combinations(sorted(preds), 2):
            za, zb = preds[a], preds[b]
            if not np.array_equal(za["te_idx"], zb["te_idx"]):
                continue
            y, cl, classes = za["y"], za["clusters"].astype(str), list(za["classes"])
            n_pairs += 1
            for mode in ("naive", "cluster"):
                lo, hi = _boot_diff(y, za["pt"], zb["pt"], cl, classes, kind, n_boot, 42, mode == "cluster")
                if np.isfinite(lo) and (lo > 0 or hi < 0):
                    sig[mode] += 1
        rows.append({"task": task, "n_teachers": len(preds), "n_teacher_pairs": n_pairs,
                     "significant_pairs_naive": sig["naive"], "significant_pairs_cluster": sig["cluster"]})
        # Kendall tau teacher ranking vs student ranking, per student size
        from scipy.stats import kendalltau
        for stu in sorted({r["student"] for r in rs}):
            sub = [r for r in rs if r["student"] == stu]
            if len(sub) >= 3:
                tau, p = kendalltau([r["t_metric"] for r in sub], [r["s_metric_mean"] for r in sub])
                tau_rows.append({"task": task, "student": stu, "n_teachers": len(sub), "kendall_tau": float(tau), "p": float(p)})
    write_csv(out / "ranking_resolution.csv", rows); write_csv(out / "ranking_kendall_tau.csv", tau_rows)
    return {"resolution": rows, "tau": tau_rows}


def a_cluster_curve(S, R, out, max_pairs=12, n_rep=20, n_boot=200, seed=42):
    rows = []
    for task in ("c16", "panda", "c17_loco", "msi", "center", "panda_biopsy"):
        files = sorted(glob.glob(f"{args_root}/*/*/seed42/probe_{task}_preds.npz"))
        files = [f for f in files if "vit-base" in f or "vit_base" in f][:max_pairs] or files[:max_pairs]
        kind = TASKS[task]["metric"] if task in TASKS else None
        for f in files:
            z = np.load(f, allow_pickle=True); y, pt, ps, cl, classes = z["y"], z["pt"], z["ps"], z["clusters"].astype(str), list(z["classes"])
            uniq = np.unique(cl); K = len(uniq); by = {g: np.flatnonzero(cl == g) for g in uniq}
            grid = sorted({k for k in (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, K) if k <= K})
            rng = np.random.default_rng(seed)
            teacher = Path(f).parts[-4]
            for k in grid:
                hw_c, hw_n = [], []
                for _ in range(n_rep if k < K else 1):
                    sel = rng.choice(uniq, k, replace=False) if k < K else uniq
                    idx = np.concatenate([by[g] for g in sel])
                    for mode, store in (("cluster", hw_c), ("naive", hw_n)):
                        lo, hi = _boot_diff(y[idx], ps[idx], pt[idx], cl[idx], classes, kind, n_boot, int(rng.integers(1 << 30)), mode == "cluster")
                        if np.isfinite(lo):
                            store.append((hi - lo) / 2)
                rows.append({"task": task, "teacher": teacher, "file": f, "n_clusters": k, "n_reps": len(hw_c),
                             "half_width_cluster": float(np.mean(hw_c)) if hw_c else np.nan,
                             "half_width_naive": float(np.mean(hw_n)) if hw_n else np.nan,
                             "n_tiles_mean": float(len(idx))})
            print(f"  [curve] {task} {teacher} done", flush=True)
    write_csv(out / "cluster_count_curve.csv", rows)
    return rows


def main():
    global args_root
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/v5"); ap.add_argument("--summary", default="outputs/v5/summary")
    ap.add_argument("--out", default="outputs/v5/secondary"); ap.add_argument("--analyses", default=",".join(ALL))
    ap.add_argument("--curve_pairs", type=int, default=12); ap.add_argument("--curve_boot", type=int, default=200)
    args = ap.parse_args(); args_root = args.root
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    S, R = load_summary(args.summary), run_records(args.root)
    done = {}
    for a in args.analyses.split(","):
        fn = globals()[f"a_{a}"]
        kw = {"max_pairs": args.curve_pairs, "n_boot": args.curve_boot} if a == "cluster_curve" else {}
        done[a] = fn(S, R, out, **kw)
        print(f"[secondary] {a} done", flush=True)
    json.dump(done, open(out / "secondary_all.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
