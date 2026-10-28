#!/usr/bin/env python3
"""Targeted PanNuke-exposure ablation (analysis plan section 7, written before any ablation result).

For each ablated run outputs/v5_ablation/no_pannuke/<teacher>/vit-base/seed42 and its original
counterpart outputs/v5/<teacher>/vit-base/seed42 (same teacher, architecture, seed, recipe):
  primary   PanNuke foreground Dice, d_abl = Dice(ablated) - Dice(original), paired per-image
            bootstrap (B=1000, seed 42), 90% percentile interval, TOST at 0.05
  controls  every other task, d_abl = metric(ablated) - metric(original) on the same test tiles,
            paired cluster bootstrap exactly as run_probes.cluster_tost, TOST at 0.05
  also      the ablated student's own teacher-student verdict per task (single seed, from its probe JSON)
Writes outputs/v5_ablation/no_pannuke/compare.json."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval_v5")); sys.path.insert(0, str(ROOT / "scripts"))
from run_probes import TASKS, cluster_tost  # noqa: E402

A = ROOT / "outputs/v5_ablation/no_pannuke"
MARGIN = 0.05


def verdict(lo, hi, m=MARGIN):
    return "equivalent" if (lo > -m and hi < m) else ("inequivalent" if (hi < -m or lo > m) else "inconclusive")


def main():
    out = {}
    for run in sorted(A.glob("*/vit-base/seed42")):
        t = run.parent.parent.name; orig = ROOT / "outputs/v5" / t / "vit-base/seed42"; res = {}
        if not (run / "probe_pannuke_seg_dice.npz").exists():
            print(t, "not evaluated yet; skipped"); continue
        a = np.load(run / "probe_pannuke_seg_dice.npz"); o = np.load(orig / "probe_pannuke_seg_dice.npz")
        sa, so = np.nanmean(a["student"][:, :5], 1), np.nanmean(o["student"][:, :5], 1)
        ok = np.isfinite(sa) & np.isfinite(so); diff = sa[ok] - so[ok]
        rng = np.random.default_rng(42); n = len(diff)
        bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(1000)])
        lo, hi = float(np.quantile(bs, .05)), float(np.quantile(bs, .95))
        res["pannuke_seg"] = {"d_abl_vs_orig": float(diff.mean()), "lo": lo, "hi": hi, "state": verdict(lo, hi),
                              "dice_ablated": float(np.nanmean(sa)), "dice_original": float(np.nanmean(so)),
                              "teacher_verdict_ablated": json.load(open(run / "probe_pannuke_seg.json"))["mDice_fg"]["state"]}
        for task, spec in TASKS.items():
            fa, fo = run / f"probe_{task}_preds.npz", orig / f"probe_{task}_preds.npz"
            if not (fa.exists() and fo.exists()):
                continue
            za, zo = np.load(fa, allow_pickle=True), np.load(fo, allow_pickle=True)
            assert np.array_equal(za["te_idx"], zo["te_idx"]) and np.array_equal(za["y"], zo["y"])
            r = cluster_tost(za["y"], zo["ps"], za["ps"], za["clusters"], za["classes"], spec["metric"], MARGIN)
            res[task] = {"d_abl_vs_orig": r["d"], "lo": r["lo"], "hi": r["hi"], "state": r["state"],
                         "metric_ablated": r["s_metric"], "metric_original": r["t_metric"],
                         "teacher_verdict_ablated": json.load(open(run / f"probe_{task}.json"))["result"]["state"]}
        out[t] = res
        for k, v in res.items():
            print(t, k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()})
    json.dump(out, open(A / "compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
