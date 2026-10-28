#!/usr/bin/env python3
"""Quantify, on real evaluation tiles, what bugs 1 and 2 did to the distillation targets.

For every teacher, on N tiles of CRC-VAL-HE-7K (public, never used for distillation):
  bug 2 (patch-14 teachers only): cosine between the correct 14x14 target (16x16 grid
         bilinearly resampled, prefix tokens removed) and the old target (last 196 of the
         256+prefix tokens), per 14x14 cell -> a misalignment heat-map, plus the mean.
  bug 1 (teachers with non-ImageNet statistics only): cosine between the CLS token with the
         teacher's own normalisation and the CLS token under ImageNet normalisation, and the
         same for the aligned patch tokens.
  both:  cosine between the correct target and the old (wrong norm + truncated) target.
Output: one JSON (summaries) and one NPZ (14x14 heat-maps) under --out.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from datasets import TileSet  # noqa: E402
from corpus_v5 import normalize_u8  # noqa: E402
from distill_wsi_model import TeacherModel  # noqa: E402

TEACHERS = ["phikon", "phikon-v2", "uni", "uni2-h", "virchow", "virchow2", "h-optimus-0",
            "prov-gigapath", "hibou-b", "hibou-l", "conch", "midnight"]


def cos(a, b, dim=-1):
    return F.cosine_similarity(a.float(), b.float(), dim=dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="crc_val_7k")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--teachers", nargs="*", default=TEACHERS)
    ap.add_argument("--out", default="research/teacher_bug_impact_v5")
    args = ap.parse_args()
    device = "cuda"
    ts = TileSet(args.dataset)
    sel = np.linspace(0, ts.n - 1, args.n).round().astype(int)
    sel = np.unique(sel)
    print(f"[tiles] {args.dataset}: {len(sel)} of {ts.n}", flush=True)
    x_u8 = torch.from_numpy(ts.read(sel))

    summary, maps = {}, {}
    for t in args.teachers:
        t0 = time.time()
        m = TeacherModel(t, patch_grid=14).eval().to(device)
        has_bug2 = m.native_grid != 14
        has_bug1 = bool(m.renorm)
        acc = {"cls_norm": [], "pat_norm": [], "trunc_cell": [], "both_cell": [], "trunc_cls_shift": []}
        with torch.no_grad():
            for s in range(0, len(sel), args.batch_size):
                x = normalize_u8(x_u8[s:s + args.batch_size].to(device))
                seq = m.forward_tokens(x)                       # correct preprocessing
                correct = m.extract_patch_tokens(seq)           # (B,196,D)
                if has_bug2:
                    old = seq[:, -196:, :]                      # bug 2: last 196 tokens
                    acc["trunc_cell"].append(cos(correct, old).cpu())         # (B,196)
                if has_bug1:
                    m.renorm = False
                    seq_l = m.forward_tokens(x)
                    m.renorm = True
                    acc["cls_norm"].append(cos(seq[:, 0], seq_l[:, 0]).cpu())
                    acc["pat_norm"].append(cos(correct, m.extract_patch_tokens(seq_l)).mean(1).cpu())
                    old_both = seq_l[:, -196:, :] if has_bug2 else m.extract_patch_tokens(seq_l)
                    acc["both_cell"].append(cos(correct, old_both).cpu())
        rec = {"patch_size": m.patch_size, "n_prefix_tokens": m.n_prefix_tokens, "native_grid": m.native_grid,
               "affected_by_bug1": has_bug1, "affected_by_bug2": has_bug2, "n_tiles": int(len(sel))}
        if has_bug2:
            c = torch.cat(acc["trunc_cell"])                    # (N,196)
            grid = c.mean(0).view(14, 14).numpy()
            maps[f"{t}__bug2_cell_cos"] = grid
            rec.update(bug2_cos_mean=float(c.mean()), bug2_cos_min_cell=float(grid.min()), bug2_cos_max_cell=float(grid.max()),
                       bug2_frac_cells_below_0p9=float((grid < 0.9).mean()))
        if has_bug1:
            cc = torch.cat(acc["cls_norm"]); pc = torch.cat(acc["pat_norm"])
            rec.update(bug1_cls_cos_mean=float(cc.mean()), bug1_cls_cos_p05=float(cc.quantile(0.05)),
                       bug1_patch_cos_mean=float(pc.mean()))
            b = torch.cat(acc["both_cell"])
            maps[f"{t}__both_cell_cos"] = b.mean(0).view(14, 14).numpy()
            rec.update(both_cos_mean=float(b.mean()))
        rec["seconds"] = round(time.time() - t0, 1)
        summary[t] = rec
        print(f"[{t}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        del m; torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": args.dataset, "n_tiles": int(len(sel)), "tile_indices_sha256_of_list": None,
               "teachers": summary}, open(str(out) + ".json", "w"), indent=1)
    np.savez_compressed(str(out) + ".npz", **maps)
    print(f"[done] -> {out}.json / .npz", flush=True)


if __name__ == "__main__":
    main()
