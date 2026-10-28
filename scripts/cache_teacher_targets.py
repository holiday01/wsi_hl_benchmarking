#!/usr/bin/env python3
"""Run one frozen teacher once over the v5 corpus and cache its targets (fp16):
cls [N,D], patch14 [N,196,D] (prefix dropped, 16x16 -> 14x14 cell-centre bilinear),
native_mean [N,D]. fp32 forward, PyTorch default matmul precision (same as the
evaluation scripts), ImageNet-normalised input converted inside TeacherModel.

After writing, --verify_n random samples are re-run through TeacherModel.forward and
compared with the cache; the run fails if cosine < 0.9999 for any sample.

  python scripts/cache_teacher_targets.py --corpus /data/wsi_hl_v5/corpus \\
      --teacher h-optimus-0 --out_root /data/wsi_hl_v5/teacher_cache
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_v5 import Corpus, CorpusImages, normalize_u8  # noqa: E402
from distill_wsi_model import TeacherModel  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--verify_n", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    corpus = Corpus(args.corpus)
    out = Path(args.out_root) / args.teacher
    if (out / "meta.json").exists() and not args.overwrite:
        if json.load(open(out / "meta.json")).get("verified"):
            print(f"[skip] {out} exists and is verified"); return
        print(f"[rebuild] {out} exists but is not verified")
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tm = TeacherModel(args.teacher, patch_grid=14).eval().to(dev)
    D, N, g, npre = tm.embed_dim, corpus.n, tm.native_grid, tm.n_prefix_tokens

    cls_mm = np.lib.format.open_memmap(out / "cls.npy", "w+", np.float16, (N, D))
    pat_mm = np.lib.format.open_memmap(out / "patch14.npy", "w+", np.float16, (N, 196, D))
    nm_mm = np.lib.format.open_memmap(out / "native_mean.npy", "w+", np.float16, (N, D))

    ld = DataLoader(CorpusImages(corpus), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for bi, (x_u8, idx) in enumerate(ld):
            x = normalize_u8(x_u8.to(dev, non_blocking=True))
            seq = tm.forward_tokens(x)
            native = seq[:, npre:, :]
            if native.shape[1] != g * g:
                raise RuntimeError(f"token count {native.shape[1]} != {g}^2")
            p14 = tm.extract_patch_tokens(seq)
            i = idx.numpy()
            cls_mm[i] = seq[:, 0, :].half().cpu().numpy()
            pat_mm[i] = p14.half().cpu().numpy()
            nm_mm[i] = native.mean(1).half().cpu().numpy()
            done += len(i)
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"  {done}/{N}  {done/max(el,1e-6):.0f} img/s  eta {(N-done)/max(done/max(el,1e-6),1e-6)/60:.1f} min",
                      flush=True)
    for m in (cls_mm, pat_mm, nm_mm):
        m.flush()
    extract_s = time.time() - t0
    del cls_mm, pat_mm, nm_mm

    # ---- verification against the ordinary forward() path ----
    rng = np.random.default_rng(0)
    vidx = np.sort(rng.choice(N, size=min(args.verify_n, N), replace=False))
    cls_c = np.load(out / "cls.npy", mmap_mode="r"); pat_c = np.load(out / "patch14.npy", mmap_mode="r")
    cos_cls, cos_pat = [], []
    with torch.no_grad():
        for s in range(0, len(vidx), 64):
            ii = vidx[s:s + 64]
            x = normalize_u8(torch.from_numpy(np.stack([corpus.images[i] for i in ii])).to(dev))
            o = tm(x, return_patches=True)
            cc = torch.from_numpy(np.asarray(cls_c[ii], dtype=np.float32)).to(dev)
            pc = torch.from_numpy(np.asarray(pat_c[ii], dtype=np.float32)).to(dev)
            cos_cls += F.cosine_similarity(o["feat"], cc, dim=-1).tolist()
            cos_pat += F.cosine_similarity(o["patch_feats"], pc, dim=-1).mean(1).tolist()
    cos_cls, cos_pat = np.asarray(cos_cls), np.asarray(cos_pat)
    n_nan = int(np.isnan(cos_cls).sum() + np.isnan(cos_pat).sum())
    v = {"n": int(len(vidx)), "n_nan": n_nan,
         "cls_cos_min": float(np.nanmin(cos_cls)), "cls_cos_mean": float(np.nanmean(cos_cls)),
         "patch_cos_min": float(np.nanmin(cos_pat)), "patch_cos_mean": float(np.nanmean(cos_pat))}
    print("[verify]", v, flush=True)
    ok = n_nan == 0 and v["cls_cos_min"] > 0.9999 and v["patch_cos_min"] > 0.9999
    meta = {"teacher": args.teacher, "dim": D, "n": N, "native_grid": g, "n_prefix_tokens": npre,
            "patch_grid": 14, "input_mean": tm.input_mean.flatten().tolist(),
            "input_std": tm.input_std.flatten().tolist(),
            "corpus_dir": str(corpus.dir), "corpus_memmap_sha256": corpus.meta["memmap_sha256"],
            "corpus_manifest_sha256": corpus.meta["manifest_sha256"],
            "git_commit": git_commit(), "torch": torch.__version__,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "extract_seconds": round(extract_s, 1), "img_per_s": round(N / extract_s, 1),
            "verification": v, "verified": ok, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if not ok:
        raise SystemExit(f"[FAIL] cache verification failed for {args.teacher}: {v}")
    print(f"[done] {args.teacher}: N={N} D={D} in {extract_s/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
