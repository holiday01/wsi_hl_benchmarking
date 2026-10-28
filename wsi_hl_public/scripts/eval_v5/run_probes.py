#!/usr/bin/env python3
"""Probe + equivalence statistics on cached features (scripts/eval_v5/extract_features.py).

For one task and one (teacher, student) pair:
  1. fixed split (pre-registered rule per task, see TASKS)
  2. linear probe (logistic regression) on teacher features and on student features
  3. paired metric difference d = metric(student) - metric(teacher) with a
     cluster bootstrap over slides/biopsies, TOST at the task's margin
     (90% percentile CI inside [-margin, +margin]), and the cluster-vs-naive SE ratio.
Output: one JSON per (task, teacher, student run).

Tasks:
  c16        c16_annotated_270       binary AUROC; slide-stratified 50/50 split (seed 42)
  c17_loco   camelyon17_annotated50  binary AUROC; leave-one-centre-out, pooled out-of-fold
  msi        kather_msi              binary AUROC; the dataset's is_test split
  panda      panda_radboud           macro one-vs-rest AUROC (4 classes); biopsy split by sha256 (50/50)
  crc7k      crc_val_7k              macro AUROC (9 classes); tile split by sha256 (50/50), no clusters
  center     tcga_brca_center        centre probe (TSS, >=3 slides), out-of-fold macro AUROC
  panda_biopsy panda_radboud         biopsy-level: mean-pooled tile features per biopsy, ISUP 0-5
                                     macro OvR AUROC, same biopsy sha256 split as `panda`;
                                     cluster = biopsy = unit, so the cluster and naive bootstraps coincide
                                     (secondary; registered 2026-10-17 in preregistration_v5 §6)
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from eval_c16_equivalence import stratified_slide_split  # noqa: E402

FEAT_ROOT = Path(os.environ.get("WSI_V5_ROOT", "/data/wsi_hl_v5")) / "features"
TASKS = {
    "c16":      {"dataset": "c16_annotated_270", "metric": "auroc", "margin": 0.05},
    "c17_loco": {"dataset": "camelyon17_annotated50", "metric": "auroc", "margin": 0.05},
    "msi":      {"dataset": "kather_msi", "metric": "auroc", "margin": 0.05},
    "panda":    {"dataset": "panda_radboud", "metric": "macro_auroc", "margin": 0.05},
    "crc7k":    {"dataset": "crc_val_7k", "metric": "macro_auroc", "margin": 0.05},
    "center":   {"dataset": "tcga_brca_center", "metric": "macro_auroc", "margin": 0.05},
    "panda_biopsy": {"dataset": "panda_radboud", "metric": "macro_auroc", "margin": 0.05, "level": "biopsy"},
    # whole-biopsy variant: up to 120 tissue tiles per biopsy chosen without the Gleason mask
    "panda_biopsy_tissue": {"dataset": "panda_radboud_tissue", "metric": "macro_auroc", "margin": 0.05, "level": "biopsy"},
}
PANDA_MANIFEST = HERE.parent.parent / "data" / "manifests" / "panda_radboud_v5.json"


def to_biopsy_level(feats, lab):
    """Mean-pool tile features per biopsy (the tile set is the class-capped sample written by
    extract_patches_v5) and attach the biopsy's ISUP grade from the manifest.
    Returns pooled feature arrays (one per input) and a label bundle with groups = biopsy ids."""
    g = np.asarray(lab["groups"]).astype(str)
    uniq, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float32)
    pooled = []
    for f in feats:
        out = np.zeros((len(uniq), f.shape[1]), np.float32)
        np.add.at(out, inv, f.astype(np.float32))
        pooled.append(out / cnt[:, None])
    man = json.load(open(PANDA_MANIFEST))["slides"]
    isup = {}
    for sl in man:
        for k in (sl["group"], sl["name"], sl["image_id"]):
            isup[str(k)] = int(sl["isup"])
    def look(u):
        for k in (u, u.rsplit(".", 1)[0]):
            if k in isup:
                return isup[k]
        raise KeyError(f"biopsy {u} not in {PANDA_MANIFEST}")
    y = np.array([look(u) for u in uniq], dtype=np.int64)
    return pooled, {"labels": y, "groups": uniq, "n_tiles_per_biopsy": cnt.astype(np.int64)}


def h01(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) / float(1 << 256)


def load_feats(dataset, key):
    d = FEAT_ROOT / dataset / key
    f = np.load(d / "feat.npy").astype(np.float32)
    return f, json.load(open(d / "meta.json"))


def probe(x_tr, y_tr, x_te, seed=42, C=1.0):
    """Logistic-regression probe on standardised features -> class probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(sc.transform(x_tr), y_tr)
    return clf.predict_proba(sc.transform(x_te)), clf.classes_


def metric(y, p, classes, kind):
    from sklearn.metrics import roc_auc_score
    if kind == "auroc":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p[:, list(classes).index(1)]))
    present = [c for c in classes if (y == c).any() and (y != c).any()]
    if len(present) < 2:
        return float("nan")
    return float(np.mean([roc_auc_score((y == c).astype(int), p[:, list(classes).index(c)]) for c in present]))


def cluster_tost(y, pt, ps, groups, classes, kind, margin, n_boot=1000, seed=42):
    y = np.asarray(y); groups = np.asarray(groups)
    uniq = np.unique(groups); by = {g: np.flatnonzero(groups == g) for g in uniq}
    d_hat = metric(y, ps, classes, kind) - metric(y, pt, classes, kind)
    rng = np.random.default_rng(seed); dc = []
    for _ in range(n_boot):
        idx = np.concatenate([by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        v = metric(y[idx], ps[idx], classes, kind) - metric(y[idx], pt[idx], classes, kind)
        if np.isfinite(v):
            dc.append(v)
    rng2 = np.random.default_rng(seed + 1); dn = []
    for _ in range(n_boot):
        idx = rng2.integers(0, len(y), len(y))
        v = metric(y[idx], ps[idx], classes, kind) - metric(y[idx], pt[idx], classes, kind)
        if np.isfinite(v):
            dn.append(v)
    out = {"d": float(d_hat), "t_metric": metric(y, pt, classes, kind), "s_metric": metric(y, ps, classes, kind),
           "n_clusters": int(len(uniq)), "n_tiles": int(len(y)), "margin": margin}
    if len(dc) < 50:
        out.update(lo=float("nan"), hi=float("nan"), state="undefined"); return out
    dc = np.asarray(dc); lo, hi = float(np.quantile(dc, 0.05)), float(np.quantile(dc, 0.95))
    se_c = float(dc.std(ddof=1)); se_n = float(np.std(dn, ddof=1)) if len(dn) >= 50 else float("nan")
    state = "equivalent" if (lo > -margin and hi < margin) else ("inequivalent" if (hi < -margin or lo > margin) else "inconclusive")
    out.update(lo=lo, hi=hi, se_cluster=se_c, se_naive=se_n,
               inflation=(se_c / se_n if se_n and se_n > 0 else float("nan")),
               p_lower=float((np.sum(dc <= -margin) + 1) / (len(dc) + 1)),
               p_upper=float((np.sum(dc >= margin) + 1) / (len(dc) + 1)), state=state)
    out["p_tost"] = max(out["p_lower"], out["p_upper"])
    return out


def split_for(task, lab, seed=42, center_group="slide"):
    """Returns list of (train_idx, test_idx, fold_name). Out-of-fold predictions are pooled."""
    y, g = lab["labels"], lab["groups"]
    n = len(y)
    if task == "c16":
        slide_int = np.unique(g, return_inverse=True)[1]          # stable, unlike hash()
        tr, te, _, _ = stratified_slide_split(slide_int, y, np.arange(n), 0.5, seed=seed)
        return [(tr, te, "split")]
    if task == "c17_loco":
        cen = np.asarray(lab["subgroup"])                         # centre0..4 per tile
        folds = []
        for c in sorted(np.unique(cen)):
            te = np.flatnonzero(cen == c); tr = np.flatnonzero(cen != c)
            folds.append((tr, te, str(c)))
        return folds
    if task == "msi":
        it = lab["is_test"]
        return [(np.flatnonzero(it == 0), np.flatnonzero(it == 1), "is_test")]
    if task in ("panda", "crc7k"):
        key = g
        b = np.array([h01(f"{task}:{seed}:{k}") for k in key])
        return [(np.flatnonzero(b < 0.5), np.flatnonzero(b >= 0.5), "sha_half")]
    if task == "center":
        # 3-fold grouped by slide (or, for the sensitivity analysis, by patient), over TSS codes with >= 3 slides
        tss = lab["subgroup"]; slides = g
        unit = np.asarray(lab["patient"]) if center_group == "patient" else slides
        keep_codes = [c for c in np.unique(tss) if len(np.unique(slides[tss == c])) >= 3]
        keep = np.isin(tss, keep_codes)
        folds = []
        fold_of_slide = {}
        for c in keep_codes:                                   # rank slides within the TSS by sha256
            ss = sorted(np.unique(unit[tss == c]), key=lambda s: h01(f"center:{seed}:{s}"))
            for r, s in enumerate(ss):
                fold_of_slide.setdefault(s, r % 3)              # a patient seen under two TSS codes keeps its first fold
        fo = np.array([fold_of_slide.get(s, -1) for s in unit])
        for k in range(3):
            folds.append((np.flatnonzero(keep & (fo != k)), np.flatnonzero(keep & (fo == k)), f"fold{k}"))
        return folds
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student_key", required=True, help="student feature dir name under the dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--teacher_embedding", default="cls", choices=["cls", "concat_native_mean"])
    ap.add_argument("--C", type=float, default=1.0, help="probe inverse regularisation (sensitivity analysis only)")
    ap.add_argument("--center_group", default="slide", choices=["slide", "patient"], help="site-task fold unit (sensitivity only)")
    args = ap.parse_args()
    spec = TASKS[args.task]
    lab = dict(np.load(FEAT_ROOT / spec["dataset"] / "labels.npz", allow_pickle=True))
    y = lab["labels"]
    if args.task == "center":
        y = np.unique(lab["subgroup"], return_inverse=True)[1]   # TSS code -> class id
    t_feat, t_meta = load_feats(spec["dataset"], f"teacher__{args.teacher}")
    if args.teacher_embedding == "concat_native_mean":
        nm = np.load(FEAT_ROOT / spec["dataset"] / f"teacher__{args.teacher}" / "native_mean.npy").astype(np.float32)
        t_feat = np.concatenate([t_feat, nm], 1)
    s_feat, s_meta = load_feats(spec["dataset"], args.student_key)
    ds_fp = json.load(open(FEAT_ROOT / spec["dataset"] / "dataset.json"))["fingerprint"]
    for who, m in (("teacher", t_meta), ("student", s_meta)):
        if m.get("dataset_fingerprint") != ds_fp:
            raise SystemExit(f"{who} features for {spec['dataset']} were extracted from a different tile set; re-extract")
    assert len(t_feat) == len(y) == len(s_feat)
    split_task = args.task
    if spec.get("level") == "biopsy":
        (t_feat, s_feat), lab = to_biopsy_level((t_feat, s_feat), lab)
        y = lab["labels"]; split_task = "panda"            # same biopsy sha256 halves as the tile task
    clusters = lab["groups"]   # slide / biopsy / tile; the bootstrap resamples these

    folds = split_for(split_task, lab, args.seed, args.center_group)
    t0 = time.time()
    pt_all = np.zeros((len(y), len(np.unique(y)))); ps_all = np.zeros_like(pt_all); covered = np.zeros(len(y), bool)
    classes_ref = np.unique(y)
    per_fold = []
    for tr, te, name in folds:
        pt, ct = probe(t_feat[tr], y[tr], t_feat[te], args.seed, args.C)
        ps, cs = probe(s_feat[tr], y[tr], s_feat[te], args.seed, args.C)
        assert list(ct) == list(cs)
        # align to the reference class order
        for arr, src in ((pt_all, pt), (ps_all, ps)):
            for j, c in enumerate(ct):
                arr[te, list(classes_ref).index(c)] = src[:, j]
        covered[te] = True
        per_fold.append({"fold": name, "n_test": int(len(te)),
                         "t_metric": metric(y[te], pt, ct, spec["metric"]),
                         "s_metric": metric(y[te], ps, cs, spec["metric"])})
    te_all = np.flatnonzero(covered)
    res = cluster_tost(y[te_all], pt_all[te_all], ps_all[te_all], np.asarray(clusters)[te_all],
                       classes_ref, spec["metric"], spec["margin"], args.n_boot, args.seed)
    rec = {"task": args.task, "dataset": spec["dataset"], "metric": spec["metric"], "teacher": args.teacher,
           "teacher_embedding": args.teacher_embedding, "student_key": args.student_key,
           "C": args.C, "center_group": args.center_group,
           "student_meta": {k: s_meta.get(k) for k in ("run", "student", "seed", "ckpt_sha256", "epoch")},
           "folds": per_fold, "result": res, "seconds": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1)
    np.savez_compressed(str(args.out).replace(".json", "_preds.npz"), te_idx=te_all, y=y[te_all],
                        pt=pt_all[te_all].astype(np.float32), ps=ps_all[te_all].astype(np.float32),
                        clusters=np.asarray(clusters)[te_all], classes=classes_ref)
    print(f"[{args.task}] {args.teacher} vs {args.student_key}: T={res['t_metric']:.4f} S={res['s_metric']:.4f} "
          f"d={res['d']:+.4f} CI90=[{res.get('lo', float('nan')):+.4f},{res.get('hi', float('nan')):+.4f}] "
          f"{res['state']} (clusters={res['n_clusters']}, inflation={res.get('inflation', float('nan')):.2f})", flush=True)


if __name__ == "__main__":
    main()
