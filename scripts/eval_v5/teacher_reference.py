#!/usr/bin/env python3
"""Teacher-only reference probes on the pre-registered splits (no student involved).

For one task, every cached teacher feature set under /data/wsi_hl_v5/features/<dataset>/
(teacher__<t> = correct preprocessing, teacher_legacy__<t> = bug-1 preprocessing with
ImageNet statistics for every teacher) is probed with the same split rule, linear probe and
metric as scripts/eval_v5/run_probes.py. Reported per (teacher, embedding):
  metric with a cluster-bootstrap 90% percentile CI,
  CLS vs CLS++mean-patch-token ("concat", the Midnight-style embedding), paired,
  correct vs legacy preprocessing, paired (how much bug 1 moved the teacher itself).
Descriptive only: it shows which tasks sit at the teachers' ceiling and the size of the
bug-1 effect on the teachers. It makes no student/teacher equivalence claim and does not
touch any pre-registered margin.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_probes import FEAT_ROOT, TASKS, cluster_tost, load_feats, metric, probe, split_for  # noqa: E402


def oof_predictions(feat, y, folds, classes_ref, seed):
    p_all = np.zeros((len(y), len(classes_ref))); covered = np.zeros(len(y), bool); per_fold = []
    for tr, te, name in folds:
        p, cl = probe(feat[tr], y[tr], feat[te], seed)
        for j, c in enumerate(cl):
            p_all[te, list(classes_ref).index(c)] = p[:, j]
        covered[te] = True
        per_fold.append({"fold": name, "n_test": int(len(te)), "metric": metric(y[te], p, cl, "auroc" if len(classes_ref) == 2 else "macro_auroc")})
    return p_all, np.flatnonzero(covered), per_fold


def cluster_ci(y, p, groups, classes, kind, n_boot, seed):
    uniq = np.unique(groups); by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        v = metric(y[idx], p[idx], classes, kind)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out_dir", default="outputs/v5/teacher_reference")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    spec = TASKS[args.task]; ds = spec["dataset"]; kind = spec["metric"]
    ds_dir = FEAT_ROOT / ds
    if not (ds_dir / "labels.npz").exists():
        raise SystemExit(f"no features yet for {ds}")
    lab = dict(np.load(ds_dir / "labels.npz", allow_pickle=True))
    y = lab["labels"]
    if args.task == "center":
        y = np.unique(lab["subgroup"], return_inverse=True)[1]
    ds_fp = json.load(open(ds_dir / "dataset.json"))["fingerprint"]
    clusters = np.asarray(lab["groups"])
    folds = split_for(args.task, lab, args.seed)
    classes_ref = np.unique(y)

    keys = sorted(d.name for d in ds_dir.iterdir()
                  if d.is_dir() and d.name.startswith(("teacher__", "teacher_legacy__")) and (d / "meta.json").exists())
    rows, preds = [], {}
    t0 = time.time()
    for key in keys:
        feat, meta = load_feats(ds, key)
        if meta.get("dataset_fingerprint") != ds_fp:
            print(f"[stale] {key}: skipped"); continue
        nm = np.load(ds_dir / key / "native_mean.npy").astype(np.float32)
        legacy = key.startswith("teacher_legacy__")
        teacher = key.split("__", 1)[1]
        for emb, f in (("cls", feat), ("concat", np.concatenate([feat, nm], 1))):
            p, te, per_fold = oof_predictions(f, y, folds, classes_ref, args.seed)
            m = metric(y[te], p[te], classes_ref, kind)
            lo, hi = cluster_ci(y[te], p[te], clusters[te], classes_ref, kind, args.n_boot, args.seed)
            preds[(teacher, emb, legacy)] = (p, te)
            rows.append({"teacher": teacher, "embedding": emb, "preprocessing": "legacy" if legacy else "correct",
                         "dim": int(f.shape[1]), "metric": m, "ci90": [lo, hi], "n_test": int(len(te)),
                         "n_clusters": int(len(np.unique(clusters[te]))), "folds": per_fold})
            print(f"[{args.task}] {teacher:14s} {emb:7s} {'legacy ' if legacy else 'correct'} {kind}={m:.4f} CI90=[{lo:.4f},{hi:.4f}]", flush=True)

    paired = []
    for (teacher, emb, legacy), (p, te) in preds.items():
        # correct - legacy (same embedding)
        if not legacy and (teacher, emb, True) in preds:
            pl, _ = preds[(teacher, emb, True)]
            r = cluster_tost(y[te], pl[te], p[te], clusters[te], classes_ref, kind, spec["margin"], args.n_boot, args.seed)
            paired.append({"teacher": teacher, "embedding": emb, "contrast": "correct_minus_legacy",
                           "d": r["d"], "ci90": [r.get("lo"), r.get("hi")], "legacy_metric": r["t_metric"], "correct_metric": r["s_metric"]})
        # concat - cls (same preprocessing)
        if emb == "concat" and (teacher, "cls", legacy) in preds:
            pc, _ = preds[(teacher, "cls", legacy)]
            r = cluster_tost(y[te], pc[te], p[te], clusters[te], classes_ref, kind, spec["margin"], args.n_boot, args.seed)
            paired.append({"teacher": teacher, "preprocessing": "legacy" if legacy else "correct", "contrast": "concat_minus_cls",
                           "d": r["d"], "ci90": [r.get("lo"), r.get("hi")], "cls_metric": r["t_metric"], "concat_metric": r["s_metric"]})
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"task": args.task, "dataset": ds, "metric": kind, "seed": args.seed, "n_boot": args.n_boot,
               "dataset_fingerprint": ds_fp, "rows": rows, "paired": paired, "seconds": round(time.time() - t0, 1)},
              open(out / f"{args.task}.json", "w"), indent=1)
    print(f"[done] {args.task}: {len(rows)} rows, {len(paired)} paired contrasts in {time.time()-t0:.0f}s -> {out / (args.task + '.json')}", flush=True)


if __name__ == "__main__":
    main()
