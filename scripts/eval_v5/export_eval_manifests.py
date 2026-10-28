#!/usr/bin/env python3
"""Export tile-level manifests of the evaluation subsets that have no manifest in data/manifests yet, so that every
subset used in the paper can be checked against the public cohorts:
  c16_annotated_270_tiles.csv.gz   CAMELYON16: tile index, slide file, label, SHA-256 of the 224x224 RGB tile
  kather_msi_tiles.csv.gz          Kather-MSI (CRC_DX, Zenodo 10.5281/zenodo.2530835): tile index, split, patient,
                                   label, source file (matched by pixel content), SHA-256
  crc_val_7k_tiles.csv.gz          CRC-VAL-HE-7K: file, class, SHA-256 split half used by the probe
  tcga_brca_center_tiles.csv.gz    TCGA-BRCA site task: tile index, slide file, patient, site code, level-0 coordinates
Writes to data/manifests/. SHA-256 is over the uint8 array bytes exactly as stored in the tile files."""
import csv, gzip, hashlib, os, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval_v5"))
from datasets import REGISTRY, TileSet  # noqa: E402

OUT = ROOT / "data/manifests"
C16_WSI = Path(os.environ.get("WSI_C16_DIR", "/mnt/nas1/wsi_datasets/camelyon16"))
KATHER = Path(os.environ.get("WSI_EXTERNAL_DIR", "/mnt/nas1/wsi_datasets/external")) / "kather_msi"
KATHER_DIRS = {("train", "MSIMUT"): ("CRC_DX_TRAIN_MSIMUT_unz/MSIMUT", 1), ("train", "MSS"): ("CRC_DX_TRAIN_MSS_unz/MSS", 0),
               ("test", "MSIMUT"): ("CRC_DX_TEST_MSIMUT_unz/MSIMUT", 1), ("test", "MSS"): ("CRC_DX_TEST_MSS_unz/MSS", 0)}
PAT = re.compile(r"(TCGA-[A-Z0-9]+-[A-Z0-9]+)")


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).hexdigest()


def h5_hashes(path, chunk=512):
    with h5py.File(path, "r") as h:
        P = h["patches"]; return [sha(x) for s in range(0, len(P), chunk) for x in P[s:s + chunk]]


def write(name, header, rows):
    with gzip.open(OUT / name, "wt", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", name, len(rows), "rows", flush=True)


def c16():
    path = REGISTRY["c16_annotated_270"]["path"]; slides = sorted(p.name for p in C16_WSI.glob("*.tif"))
    with h5py.File(path, "r") as h:
        sid, lab = h["slide_ids"][:], h["labels"][:]
    hs = h5_hashes(path)
    write("c16_annotated_270_tiles.csv.gz", ["tile", "slide", "label_tumour", "sha256"],
          [(i, slides[s], int(l), x) for i, (s, l, x) in enumerate(zip(sid, lab, hs))])


def _file_hash(p):
    try:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        if a.shape[:2] != (224, 224):
            a = np.asarray(Image.fromarray(a).resize((224, 224)), dtype=np.uint8)
        return p, sha(a)
    except Exception:
        return p, None


def kather():
    path = REGISTRY["kather_msi"]["path"]
    files, meta = [], {}
    for (split, cls), (sub, lab) in KATHER_DIRS.items():
        for fn in os.listdir(KATHER / sub):
            m = PAT.search(fn)
            if m:
                p = str(KATHER / sub / fn); files.append(p); meta[p] = (split, m.group(1), lab)
    by_hash = {}
    with ProcessPoolExecutor(48) as ex:
        for p, x in ex.map(_file_hash, files, chunksize=256):
            if x:
                by_hash.setdefault(x, []).append(p)
    with h5py.File(path, "r") as h:
        lab, test = h["labels"][:], h["is_test"][:]
    hs = h5_hashes(path); rows, miss = [], 0
    for i, (x, l, t) in enumerate(zip(hs, lab, test)):
        cand = [p for p in by_hash.get(x, []) if meta[p][0] == ("test" if t else "train") and meta[p][2] == l]
        if not cand:
            miss += 1
        p = cand[0] if cand else ""
        rows.append((i, "test" if t else "train", meta[p][1] if p else "", int(l), os.path.relpath(p, KATHER) if p else "", x))
    write("kather_msi_tiles.csv.gz", ["tile", "split", "patient", "label_msi", "source_file", "sha256"], rows)
    print("kather tiles without a matched source file:", miss)


def crc7k():
    import run_probes  # noqa: E402
    ts = TileSet("crc_val_7k")
    lab = dict(np.load(run_probes.FEAT_ROOT / "crc_val_7k" / "labels.npz", allow_pickle=True))
    assert np.array_equal(lab["labels"], ts.labels)
    te = set(run_probes.split_for("crc7k", lab)[0][1].tolist())
    write("crc_val_7k_tiles.csv.gz", ["tile", "file", "class", "half"],
          [(i, os.path.relpath(ts.files[i], REGISTRY["crc_val_7k"]["root"]), int(ts.labels[i]), "test" if i in te else "train")
           for i in range(ts.n)])


def center():
    path = REGISTRY["tcga_brca_center"]["path"]
    with h5py.File(path, "r") as h:
        sid, xy = h["slide_ids"][:], h["coords"][:]
        names = [x.decode() for x in h.attrs["slide_names"]]; pats = [x.decode() for x in h.attrs["slide_patients"]]
        sites = [x.decode() for x in h.attrs["slide_groups"]]
    write("tcga_brca_center_tiles.csv.gz", ["tile", "slide", "patient", "site", "x0", "y0"],
          [(i, names[s], pats[s], sites[s], int(c[0]), int(c[1])) for i, (s, c) in enumerate(zip(sid, xy))])


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["c16", "kather", "crc7k", "center"]):
        globals()[f]()
