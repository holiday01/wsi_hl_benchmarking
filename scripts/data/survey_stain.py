#!/usr/bin/env python3
"""Thumbnail-based stain survey: flag IHC (DAB brown) slides hiding among TCGA "DX"
diagnostic slides. For every slide: fraction of saturated tissue pixels whose hue is
brown (DAB, 15-60 deg) vs pink/purple (H&E, 250-360 deg).
  python scripts/data/survey_stain.py --meta data/manifests/tcga_svs_meta_2026-10-16.tsv \
      --out data/manifests/tcga_stain_survey_2026-10-16.tsv --workers 24
"""
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_patches import build_tissue_mask  # noqa: E402


def one(path):
    import openslide
    from PIL import Image
    try:
        s = openslide.OpenSlide(path)
        th = np.asarray(s.get_thumbnail((1024, 1024)).convert("RGB"))
        s.close()
        tissue = build_tissue_mask(th)
        hsv = np.asarray(Image.fromarray(th).convert("HSV")).astype(np.float32)
        hue = hsv[..., 0] * 360.0 / 255.0; sat = hsv[..., 1] / 255.0
        m = tissue & (sat > 0.15)
        n = int(m.sum())
        if n < 200:
            return (path, "ERR", "too little tissue", 0, 0, 0)
        h = hue[m]
        brown = float(((h >= 15) & (h <= 60)).mean())
        pink = float(((h >= 250) | (h <= 10)).mean())
        blue = float(((h >= 180) & (h < 250)).mean())
        return (path, "ok", "", round(brown, 4), round(pink, 4), round(blue, 4))
    except Exception as e:  # noqa: BLE001
        return (path, "ERR", str(e)[:60], 0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--root", default="/mnt/nas1/wsi_datasets/tcga/")
    args = ap.parse_args()
    paths = []
    with open(args.meta) as f:
        for row in csv.reader(f, delimiter="\t"):
            paths.append(args.root + row[0])
    with Pool(args.workers) as pool, open(args.out, "w") as f:
        f.write("path\tstatus\tnote\tbrown_frac\tpink_frac\tblue_frac\n")
        for i, r in enumerate(pool.imap_unordered(one, paths, chunksize=4)):
            f.write("\t".join(map(str, r)) + "\n")
            if i % 200 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
