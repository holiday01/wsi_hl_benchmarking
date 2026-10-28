#!/usr/bin/env python3
"""v5 distillation: same recipe as distill_v4.py (L_CLS 1-cos + L_PAT 1-cos, AdamW,
cosine LR, early stopping on val loss) but the teacher targets come from the fp16
cache written by cache_teacher_targets.py, so a run costs only the student's
forward/backward. Several runs can share one GPU.

Outputs (same layout as v4 so the evaluation scripts can consume them):
  best.pt            {"epoch", "student", "criterion", "val_loss", "config"}
  training_log.json  args, history, best_val/epoch, corpus + cache hashes
  splits.npz         train/val index arrays into the corpus
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_v5 import Corpus, DistillPairs, TeacherCache, normalize_u8  # noqa: E402
from distill_v4_loss import V4DistillLoss  # noqa: E402
from distill_wsi_model import StudentModel  # noqa: E402


def run_epoch(student, loader, optimizer, criterion, device, train, amp):
    student.train(train)
    tot, n = 0.0, 0
    torch.cuda.reset_peak_memory_stats(device)
    for x_u8, t_cls, t_pat in loader:
        x = normalize_u8(x_u8.to(device, non_blocking=True))
        t_out = {"feat": t_cls.to(device, non_blocking=True).float(),
                 "patch_feats": t_pat.to(device, non_blocking=True).float()}
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s_out = student(x, return_patches=True)
            loss = criterion(s_out, t_out)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        tot += loss.item(); n += 1
    return tot / max(n, 1), torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", default="vit_small_patch16_224")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam_cls", type=float, default=1.0)
    ap.add_argument("--lam_pat", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast for the student (off = fp32 like v4)")
    ap.add_argument("--exclude_tags", default="", help="comma-separated corpus source tags left out of train and val (ablation)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda"
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(args.corpus)
    cache = TeacherCache(Path(args.cache_root) / args.teacher, corpus)
    if not cache.meta.get("verified"):
        raise SystemExit(f"teacher cache for {args.teacher} is not verified")
    tr_idx, va_idx = corpus.indices("train"), corpus.indices("val")
    if args.exclude_tags:
        drop = np.isin(corpus.tag, args.exclude_tags.split(","))
        tr_idx, va_idx = tr_idx[~drop[tr_idx]], va_idx[~drop[va_idx]]
    print(f"[corpus] N={corpus.n} train={len(tr_idx)} val={len(va_idx)} "
          f"memmap={corpus.meta['memmap_sha256'][:12]}  [cache] {args.teacher} D={cache.dim}", flush=True)

    g = torch.Generator(); g.manual_seed(args.seed)
    tr_ld = DataLoader(DistillPairs(corpus, cache, tr_idx), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                       persistent_workers=args.num_workers > 0, generator=g)
    va_ld = DataLoader(DistillPairs(corpus, cache, va_idx), batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       persistent_workers=args.num_workers > 0)

    student = StudentModel(args.student, embed_dim=256, n_classes=0, pretrained=True).to(device)
    criterion = V4DistillLoss(student_cls_dim=student.embed_dim,
                              student_patch_dim=student.backbone.num_features,
                              teacher_dim=cache.dim, lam_cls=args.lam_cls, lam_pat=args.lam_pat,
                              use_mgd=False, use_dino=False).to(device)
    opt = torch.optim.AdamW(list(student.parameters()) + list(criterion.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"[student] {args.student} params={s_params:,}  loss=L_CLS({args.lam_cls})+L_PAT({args.lam_pat})"
          f"  amp={args.amp}", flush=True)

    history, best, best_ep, patience = [], float("inf"), -1, args.patience
    t0 = time.time(); start_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for epoch in range(1, args.epochs + 1):
        ep0 = time.time()
        tr_l, tr_mb = run_epoch(student, tr_ld, opt, criterion, device, True, args.amp)
        sch.step()
        va_l, va_mb = run_epoch(student, va_ld, opt, criterion, device, False, args.amp)
        ep_dt = time.time() - ep0
        if not (np.isfinite(tr_l) and np.isfinite(va_l)):
            raise SystemExit(f"non-finite loss at epoch {epoch}: train {tr_l} val {va_l}")
        history.append({"epoch": epoch, "train_loss": tr_l, "val_loss": va_l,
                        "elapsed_total_s": time.time() - t0, "elapsed_epoch_s": ep_dt,
                        "peak_vram_train_mb": tr_mb, "peak_vram_val_mb": va_mb})
        mark = ""
        if va_l < best - 1e-5:
            best, best_ep, patience, mark = va_l, epoch, args.patience, " *"
            torch.save({"epoch": epoch, "student": student.state_dict(),
                        "criterion": criterion.state_dict(), "val_loss": va_l,
                        "config": vars(args)}, out / "best.pt")
        else:
            patience -= 1
        print(f"  epoch {epoch:3d} | train {tr_l:.4f} | val {va_l:.4f}{mark} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {ep_dt:.0f}s | vram {tr_mb:.0f}MB", flush=True)
        with open(out / "training_log.json", "w") as f:   # written every epoch (resumable bookkeeping)
            json.dump({"args": vars(args), "history": history, "best_val": best, "best_epoch": best_ep,
                       "student_params": s_params, "teacher_dim": cache.dim,
                       "corpus_memmap_sha256": corpus.meta["memmap_sha256"],
                       "corpus_manifest_sha256": corpus.meta["manifest_sha256"],
                       "teacher_cache_meta": cache.meta, "start_ts": start_ts,
                       "end_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "finished": epoch == args.epochs or patience <= 0}, f, indent=1)
        if patience <= 0:
            print(f"  [early stop] no improvement for {args.patience} epochs", flush=True)
            break
    np.savez(out / "splits.npz", train=tr_idx, val=va_idx)
    print(f"[done] best val {best:.4f} @ ep {best_ep}  ({(time.time()-t0)/60:.1f} min) -> {out}", flush=True)


if __name__ == "__main__":
    main()
