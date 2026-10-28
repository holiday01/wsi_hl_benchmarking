#!/usr/bin/env python3
"""PanNuke segmentation linear probe on cached patch tokens (v5).

Same protocol as scripts/evaluate_pannuke_seg.py (14x14 token grid, majority-class
block labels, linear D->6 probe, per-image pixel Dice, image-cluster bootstrap) but
patch tokens are cached per model so each teacher's forward over the 3 folds runs once.

  # cache tokens (teacher or student), folds 1,2 = probe training, 3 = test
  python scripts/eval_v5/pannuke_seg.py cache --model teacher:h-optimus-0
  python scripts/eval_v5/pannuke_seg.py cache --model student:outputs/v5/h-optimus-0/vit-small/seed42
  # probe + statistics for one pair
  python scripts/eval_v5/pannuke_seg.py probe --teacher h-optimus-0 \
      --student_key student__outputs_v5_h-optimus-0_vit-small_seed42 --out <json>

Token cache: /data/wsi_hl_v5/features/pannuke/<model_key>/tokens_fold{1,2,3}.npy fp16 [N,196,D];
labels: /data/wsi_hl_v5/features/pannuke/labels_fold{k}.npz (cls_grid [N,14,14], cls_map [N,256,256] uint8).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from evaluate_pannuke_seg import (CLASS_NAMES, PanNukeFold, linear_seg_probe,  # noqa: E402
                                  per_image_dice)


def upsample_grid_to_pixel(grid_cls, target=256):
    """(g,g) int -> (target,target) nearest neighbour; pixel r maps to grid row floor(r*g/target),
    so every pixel gets a prediction (evaluate_pannuke_seg's np.repeat version yields 252x252 for g=14)."""
    g = grid_cls.shape[0]
    idx = (np.arange(target) * g) // target
    return grid_cls[idx][:, idx]
from extract_features import load_model  # noqa: E402
from stats_v4 import slide_cluster_bootstrap  # noqa: E402

PN_ROOT = Path(os.environ.get("WSI_EXTERNAL_DIR", "/mnt/nas1/wsi_datasets/external")) / "pannuke"
OUT_ROOT = Path(os.environ.get("WSI_V5_ROOT", "/data/wsi_hl_v5")) / "features/pannuke"
FOLDS = {1: "fold_1_unz", 2: "fold_2_unz", 3: "fold_3_unz"}


def cache(args):
    device = "cuda"
    model, key, info, is_teacher = load_model(args.model, device)
    out = OUT_ROOT / key
    if (out / "meta.json").exists() and not args.overwrite:
        print(f"[skip] {out}"); return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); shapes = {}
    for k, sub in FOLDS.items():
        ds = PanNukeFold(PN_ROOT / sub)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        toks, grids, maps = [], [], []
        with torch.no_grad():
            for x, lab, pm, _ in ld:
                o = model(x.to(device), return_patches=True)
                toks.append(o["patch_feats"].half().cpu().numpy())
                if not (OUT_ROOT / f"labels_fold{k}.npz").exists():
                    grids.append(lab.numpy().astype(np.uint8)); maps.append(pm.numpy().astype(np.uint8))
        tok = np.concatenate(toks); np.save(out / f"tokens_fold{k}.npy", tok); shapes[k] = list(tok.shape)
        if grids:
            np.savez(OUT_ROOT / f"labels_fold{k}.npz", cls_grid=np.concatenate(grids), cls_map=np.concatenate(maps))
        print(f"  fold {k}: {tok.shape} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"model": args.model, "shapes": shapes, "seconds": round(time.time() - t0, 1), **info},
              open(out / "meta.json", "w"), indent=1)
    print(f"[done] {key} -> {out}", flush=True)


def _load(key, folds):
    return torch.from_numpy(np.concatenate([np.load(OUT_ROOT / key / f"tokens_fold{k}.npy") for k in folds]).astype(np.float32))


def probe(args):
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    lab_tr = [np.load(OUT_ROOT / f"labels_fold{k}.npz") for k in (1, 2)]
    lab_te = np.load(OUT_ROOT / "labels_fold3.npz")
    y_tr = torch.from_numpy(np.concatenate([l["cls_grid"] for l in lab_tr]).astype(np.int64)).reshape(-1, 196)
    y_te = torch.from_numpy(lab_te["cls_grid"].astype(np.int64)).reshape(-1, 196)
    pix_te = lab_te["cls_map"]
    res = {}
    for who, key in (("teacher", f"teacher__{args.teacher}"), ("student", args.student_key)):
        tr, te = _load(key, (1, 2)), _load(key, (3,))
        pred = linear_seg_probe(tr, y_tr, te, y_te, n_classes=6, device=device)   # (N, 196)
        pred = np.asarray(pred).reshape(-1, 14, 14)
        dice = np.zeros((len(pix_te), 6))
        for i in range(len(pix_te)):
            dice[i] = per_image_dice(upsample_grid_to_pixel(pred[i]), pix_te[i])
        res[who] = dice
    t_d, s_d = res["teacher"], res["student"]
    ids = np.array([f"img{i}" for i in range(len(t_d))])
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        d = s_d[:, c] - t_d[:, c]; ok = np.isfinite(d)
        b = slide_cluster_bootstrap(d[ok], ids[ok], np.mean, n_boot=args.n_boot, seed=args.seed, ci=0.90)
        per_class[name] = {"teacher_dice": float(np.nanmean(t_d[:, c])), "student_dice": float(np.nanmean(s_d[:, c])),
                           "delta": float(np.nanmean(d)), "ci90": [b["lo"], b["hi"]], "n_images": int(ok.sum())}
    fg_t, fg_s = np.nanmean(t_d[:, :5], 1), np.nanmean(s_d[:, :5], 1)
    d = fg_s - fg_t; ok = np.isfinite(d)
    b = slide_cluster_bootstrap(d[ok], ids[ok], np.mean, n_boot=args.n_boot, seed=args.seed, ci=0.90)
    m = args.margin
    state = "equivalent" if (b["lo"] > -m and b["hi"] < m) else ("inequivalent" if (b["hi"] < -m or b["lo"] > m) else "inconclusive")
    rec = {"task": "pannuke_seg", "teacher": args.teacher, "student_key": args.student_key,
           "mDice_fg": {"teacher": float(np.nanmean(fg_t)), "student": float(np.nanmean(fg_s)), "d": float(np.nanmean(d)),
                        "lo": b["lo"], "hi": b["hi"], "se_cluster": b["se"], "margin": m, "state": state,
                        "retention": float(np.nanmean(fg_s) / np.nanmean(fg_t))},
           "per_class": per_class, "n_test_images": int(len(t_d))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=1)
    np.savez(str(args.out).replace(".json", "_dice.npz"), teacher=t_d, student=s_d)
    print(f"[pannuke] {args.teacher} vs {args.student_key}: mDice T={rec['mDice_fg']['teacher']:.4f} "
          f"S={rec['mDice_fg']['student']:.4f} d={rec['mDice_fg']['d']:+.4f} CI90=[{b['lo']:+.4f},{b['hi']:+.4f}] {state}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--model", required=True); c.add_argument("--batch_size", type=int, default=32)
    c.add_argument("--num_workers", type=int, default=6); c.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("probe"); p.add_argument("--teacher", required=True); p.add_argument("--student_key", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=1000); p.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()
    (cache if args.cmd == "cache" else probe)(args)


if __name__ == "__main__":
    main()
