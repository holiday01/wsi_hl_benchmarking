#!/usr/bin/env python3
"""Build the v5 distillation corpus: one uint8 memmap [N,224,224,3] plus a manifest.

Sources are listed in a JSON config (see configs/corpus_v5.json):
  {"type": "hdf5",   "tag": ..., "path": ..., "max_samples": ..}
  {"type": "folder", "tag": ..., "root": ..., "patterns": [...], "max_samples": ..}
  {"type": "numpy",  "tag": ..., "paths": [...], "max_samples": ..}
Images are resized to 224x224 (PIL bilinear, as multi_source_dataset._to_tensor did) and
stored raw. Every sample gets a group id (slide for HDF5, sha256 bucket of the file name
for folders, file:index-bucket for numpy) and a train/val assignment by
sha256("<seed>:<group>") -- never Python's salted hash(). Subsampling (max_samples) is
also sha256-ordered so it is reproducible without an RNG state.

The manifest (JSON) records every sample's origin, the split, and sha256 digests of the
memmap and of the manifest body, so the corpus can be verified byte-for-byte later.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

P = 224


def h01(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) / float(1 << 256)


def to224(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] != (P, P):
        arr = np.asarray(Image.fromarray(arr).resize((P, P), Image.BILINEAR))
    return np.ascontiguousarray(arr)


def plan_hdf5(src, seed):
    with h5py.File(src["path"], "r") as h:
        n = h["patches"].shape[0]
        sid = h["slide_ids"][:]
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in h.attrs.get("slide_names", [])]
    items = [{"tag": src["tag"], "type": "hdf5", "path": src["path"], "index": int(i),
              "group": f"{src['tag']}:{names[int(sid[i])] if names else 'slide' + str(int(sid[i]))}"}
             for i in range(n)]
    return subsample(items, src.get("max_samples"), seed, key=lambda it: f"{it['path']}#{it['index']}")


def plan_folder(src, seed):
    root = Path(src["root"]).expanduser()
    paths = []
    for pat in src.get("patterns", ["*.tif", "*.png", "*.jpg"]):
        paths.extend(root.rglob(pat))
    paths = sorted(str(p) for p in paths)
    nb = int(src.get("group_buckets", 20))
    items = [{"tag": src["tag"], "type": "folder", "path": p, "index": 0,
              "group": f"{src['tag']}:b{int(h01('grp:' + os.path.basename(p)) * nb):03d}"}
             for p in paths]
    return subsample(items, src.get("max_samples"), seed, key=lambda it: it["path"])


def plan_numpy(src, seed):
    items = []
    nb = int(src.get("group_buckets", 20))
    for p in src["paths"]:
        a = np.load(p, mmap_mode="r")
        for i in range(len(a)):
            items.append({"tag": src["tag"], "type": "numpy", "path": p, "index": i,
                          "group": f"{src['tag']}:{os.path.basename(os.path.dirname(p))}:b{i % nb:03d}"})
    return subsample(items, src.get("max_samples"), seed, key=lambda it: f"{it['path']}#{it['index']}")


def subsample(items, k, seed, key):
    if k is None or k >= len(items):
        return items
    scored = sorted(items, key=lambda it: h01(f"sub:{seed}:{key(it)}"))
    return sorted(scored[:k], key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[plan]", flush=True)
    items = []
    for src in cfg["sources"]:
        t0 = time.time()
        f = {"hdf5": plan_hdf5, "folder": plan_folder, "numpy": plan_numpy}[src["type"]]
        its = f(src, args.seed)
        for it in its:
            it["split"] = "val" if h01(f"split:{args.seed}:{it['group']}") < args.val_frac else "train"
        items.extend(its)
        n_groups = len({it["group"] for it in its})
        n_val = sum(it["split"] == "val" for it in its)
        print(f"  {src['tag']:12s} {len(its):7d} samples  groups={n_groups}  val={n_val}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    N = len(items)
    print(f"  total {N}", flush=True)

    mm_path = out / "corpus_u8.npy"
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=(N, P, P, 3))

    # materialise: hdf5 sources read in index order (sequential), others via a pool
    print("[write]", flush=True)
    t0 = time.time()
    pos = 0
    by_src = {}
    for i, it in enumerate(items):
        by_src.setdefault((it["type"], it["path"]), []).append(i)
    from multiprocessing import Pool
    for (typ, path), idxs in by_src.items():
        if typ == "hdf5":
            with h5py.File(path, "r") as h:
                ds = h["patches"]
                order = sorted(idxs, key=lambda i: items[i]["index"])
                for s in range(0, len(order), 512):
                    chunk = order[s:s + 512]
                    sel = [items[i]["index"] for i in chunk]
                    arr = ds[sel]
                    for j, i in enumerate(chunk):
                        mm[i] = to224(arr[j])
        elif typ == "numpy":
            a = np.load(path, mmap_mode="r")
            for i in idxs:
                mm[i] = to224(np.asarray(a[items[i]["index"]]))
        else:
            continue  # folders handled below in one pool
        print(f"  {typ} {os.path.basename(path)[:40]}: {len(idxs)} ({time.time()-t0:.0f}s)", flush=True)
    folder_idx = [i for i, it in enumerate(items) if it["type"] == "folder"]
    if folder_idx:
        with Pool(args.workers) as pool:
            for k, (i, arr) in enumerate(pool.imap_unordered(_load_global, [(i, items[i]["path"]) for i in folder_idx], chunksize=64)):
                mm[i] = arr
                if k % 20000 == 0:
                    print(f"  folder {k}/{len(folder_idx)} ({time.time()-t0:.0f}s)", flush=True)
    mm.flush(); del mm

    print("[hash]", flush=True)
    hsh = hashlib.sha256()
    with open(mm_path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 26), b""):
            hsh.update(blk)
    mm_digest = hsh.hexdigest()
    body = {"config": cfg, "seed": args.seed, "val_frac": args.val_frac, "n": N,
            "memmap": str(mm_path), "memmap_sha256": mm_digest, "samples": items}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["manifest_sha256"] = digest
    with open(out / "manifest.json", "w") as f:
        json.dump(body, f)
    with open(out / "meta.json", "w") as f:
        json.dump({k: v for k, v in body.items() if k != "samples"}, f, indent=1)
    # compact arrays for fast loading in training
    np.savez(out / "index.npz",
             tag=np.array([it["tag"] for it in items]),
             group=np.array([it["group"] for it in items]),
             split=np.array([it["split"] for it in items]))
    print(f"[done] N={N} memmap sha256={mm_digest[:16]} manifest sha256={digest[:16]} "
          f"({time.time()-t0:.0f}s)", flush=True)


def _load_global(arg):
    i, path = arg
    return i, to224(np.asarray(Image.open(path).convert("RGB")))


if __name__ == "__main__":
    main()
