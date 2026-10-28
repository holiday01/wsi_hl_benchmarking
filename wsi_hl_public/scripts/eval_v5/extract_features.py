#!/usr/bin/env python3
"""Extract and cache features of one model on one evaluation tile set.

  python scripts/eval_v5/extract_features.py --dataset c16_annotated_270 --model teacher:h-optimus-0
  python scripts/eval_v5/extract_features.py --dataset c16_annotated_270 \
      --model student:outputs/v5/h-optimus-0/vit-small/seed42

Writes /data/wsi_hl_v5/features/<dataset>/<model_key>/{feat.npy (fp16), meta.json}
(teachers also native_mean.npy for the Midnight concat embedding) and
/data/wsi_hl_v5/features/<dataset>/labels.npz once. Teacher = CLS token, student = the
256-d projection head output, both from ImageNet-normalised tiles (TeacherModel applies
the teacher's own statistics). fp32, PyTorch default matmul precision. Idempotent.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from datasets import TileSet  # noqa: E402
from corpus_v5 import normalize_u8  # noqa: E402
from distill_wsi_model import StudentModel, TeacherModel  # noqa: E402

FEAT_ROOT = Path(os.environ.get("WSI_V5_ROOT", "/data/wsi_hl_v5")) / "features"
STU_MAP = {"vit-tiny": "vit_tiny_patch16_224", "vit-small": "vit_small_patch16_224",
           "vit-base": "vit_base_patch16_224"}


def student_key(run_dir) -> str:
    """Feature-dir name for a student run: 'student__' + path relative to outputs/ with '/'->'_'.
    run_v5_pipeline.sh must build the same string: student__$(echo ${d#outputs/} | sed 's#/#_#g')."""
    r = str(run_dir).rstrip("/")
    for pfx in ("outputs/", "/home/holiday/wsi_hl/outputs/"):
        if r.startswith(pfx):
            r = r[len(pfx):]
    return "student__" + r.replace("/", "_")


def sha256_file(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def load_model(spec, device):
    kind, arg = spec.split(":", 1)
    if kind == "teacher":
        m = TeacherModel(arg).eval().to(device)
        return m, f"teacher__{arg}", {"teacher": arg, "input_mean": m.input_mean.flatten().tolist()}, True
    if kind == "teacher_legacy":
        # Bug-1 preprocessing (ImageNet statistics fed to every teacher, no renormalisation).
        # Only differs from teacher:<name> for teachers whose own statistics are not ImageNet.
        m = TeacherModel(arg).eval().to(device)
        m.renorm = False
        return m, f"teacher_legacy__{arg}", {"teacher": arg, "legacy_norm": True,
                                             "input_mean": [0.485, 0.456, 0.406]}, True
    if kind == "student":
        run = Path(arg)
        ck_path = run / "best.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        arch = cfg.get("student") or STU_MAP[run.name]
        m = StudentModel(arch, embed_dim=256, n_classes=0, pretrained=False)
        m.load_state_dict(ck["student"], strict=True)
        m = m.eval().to(device)
        key = student_key(run)
        return m, key, {"run": str(run), "student": arch, "ckpt_sha256": sha256_file(ck_path),
                        "teacher": cfg.get("teacher"), "seed": cfg.get("seed"), "epoch": ck.get("epoch")}, False
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="teacher:<name>, teacher_legacy:<name> (bug-1 preprocessing) or student:<run_dir>")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    ts = TileSet(args.dataset)
    ds_dir = FEAT_ROOT / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    ds_json = ds_dir / "dataset.json"
    if ds_json.exists() and json.load(open(ds_json))["fingerprint"] != ts.fingerprint:
        print(f"[stale] {args.dataset} tile set changed; labels.npz rewritten, old features are invalid")
        ds_json.unlink()
    if not ds_json.exists():
        np.savez(ds_dir / "labels.npz", **ts.label_bundle())
        json.dump({"dataset": args.dataset, "n": ts.n, "fingerprint": ts.fingerprint},
                  open(ds_json, "w"), indent=1)
    model, key, info, is_teacher = load_model(args.model, device)
    out = ds_dir / key
    if (out / "meta.json").exists() and not args.overwrite:
        if json.load(open(out / "meta.json")).get("dataset_fingerprint") == ts.fingerprint:
            print(f"[skip] {out}"); return
        print(f"[stale] {out}: features from an older tile set; re-extracting")
    out.mkdir(parents=True, exist_ok=True)
    feats, nmeans = [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, ts.n, args.batch_size):
            idx = np.arange(s, min(s + args.batch_size, ts.n))
            x = normalize_u8(torch.from_numpy(ts.read(idx)).to(device))
            if is_teacher:
                seq = model.forward_tokens(x)
                feats.append(seq[:, 0].half().cpu().numpy())
                nmeans.append(seq[:, model.n_prefix_tokens:].mean(1).half().cpu().numpy())
            else:
                feats.append(model(x)["feat"].half().cpu().numpy())
    feat = np.concatenate(feats)
    np.save(out / "feat.npy", feat)
    if is_teacher:
        np.save(out / "native_mean.npy", np.concatenate(nmeans))
    meta = {"dataset": args.dataset, "n": int(ts.n), "dim": int(feat.shape[1]), "model": args.model,
            "dataset_fingerprint": ts.fingerprint, "seconds": round(time.time() - t0, 1),
            "torch": torch.__version__, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **info}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"[done] {args.dataset} x {key}: {feat.shape} in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
