#!/usr/bin/env python3
"""MPP-aware WSI patch extraction (v5 redo). Replaces the pyramid-level rule of
extract_patches.py ("level 1 = 20x"), which is wrong for Aperio SVS (level 1 is 1/4)
and for 20x-native slides.

Every tile is read at level 0 over a field of view of patch_size * target_mpp / mpp0
pixels and resized to patch_size, so all tiles are at target_mpp (default 0.5 um/px,
"20x") regardless of the scanner's native resolution.

Tissue is detected with Otsu on a thumbnail (same rule as extract_patches.py); tiles
are sampled from a non-overlapping grid with a per-slide RNG seeded by sha256 of the
slide name, so a slide's tiles do not depend on which other slides are processed.

Optional ASAP XML annotations (CAMELYON16/17) give tile labels: 1 if the tile centre
lies in a tumour polygon (minus exclusions), else 0. --tumor_cap/--normal_cap limit
each class per annotated slide; --patches_per_slide limits unannotated slides.

Optional label masks (PANDA Radboud, --mask_labels radboud): the mask tiff (R channel:
0 background, 1 stroma, 2 benign epithelium, 3/4/5 Gleason 3/4/5) is read over the same
field of view; the tile label is the majority non-background class, merged into
0 = benign (stroma + benign epithelium), 1 = Gleason 3, 2 = Gleason 4, 3 = Gleason 5,
and the tile is kept only if that majority covers >= --mask_majority of the tissue pixels
(and tissue >= tissue_ratio). --per_class_cap limits each class per slide.

Output HDF5 (same layout the evaluation scripts read):
  patches [N,P,P,3] uint8, labels [N], slide_ids [N], coords [N,2] (level-0 x,y)
  attrs: slide_names, slide_patients, slide_groups, slide_mpp, target_mpp, patch_size
"""
import argparse, hashlib, json, os, random, sys, time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from PIL import Image



def build_tissue_mask(thumb_rgb: np.ndarray, min_luminance: float = 0.05):
    """Thumbnail -> tissue mask. Otsu is computed on NON-black pixels only: CAMELYON17
    TIFFs carry large black scanned-out regions that otherwise capture the threshold
    (extract_patches.build_tissue_mask thresholds the whole image and finds ~0 tissue)."""
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu
    gray = rgb2gray(thumb_rgb)
    valid = gray > min_luminance
    try:
        thr = threshold_otsu(gray[valid]) if valid.sum() > 100 else 0.8
    except ValueError:
        thr = 0.8
    return valid & (gray < thr)


def parse_polygons(xml_path):
    from matplotlib.path import Path as MplPath
    root = ET.parse(str(xml_path)).getroot()
    tumor, exclude = [], []
    for ann in root.iter("Annotation"):
        grp = (ann.get("PartOfGroup") or "").strip().lower()
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in ann.iter("Coordinate")]
        if len(pts) < 3:
            continue
        poly = MplPath(np.asarray(pts))
        (exclude if grp in ("exclusion", "_2", "none") else tumor).append(poly)
    return tumor, exclude


def tile_is_tumor(cx, cy, tumor, exclude):
    if not any(p.contains_point((cx, cy)) for p in tumor):
        return False
    return not any(p.contains_point((cx, cy)) for p in exclude)


def slide_rng(name: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16], 16))


def extract_one(job):
    """job: dict(path, name, xml (or None), patches_per_slide, tumor_cap, normal_cap,
    patch_size, target_mpp, tissue_ratio, seed). Returns dict with arrays or error."""
    import openslide
    t0 = time.time()
    try:
        s = openslide.OpenSlide(job["path"])
        pr = s.properties
        mpp = pr.get("openslide.mpp-x")
        if mpp is None:
            return {"name": job["name"], "error": "no mpp"}
        mpp = float(mpp)
        ps, tmpp = job["patch_size"], job["target_mpp"]
        fov = int(round(ps * tmpp / mpp))            # level-0 pixels per tile
        W, H = s.dimensions
        thumb = s.get_thumbnail((4096, 4096)).convert("RGB")
        thumb_np = np.asarray(thumb)
        tissue = build_tissue_mask(thumb_np)
        sy, sx = thumb_np.shape[0] / H, thumb_np.shape[1] / W
        cands = []
        for y0 in range(0, H - fov + 1, fov):
            ty0, ty1 = int(y0 * sy), max(int((y0 + fov) * sy), int(y0 * sy) + 1)
            row = tissue[ty0:ty1]
            if row.size == 0:
                continue
            for x0 in range(0, W - fov + 1, fov):
                tx0, tx1 = int(x0 * sx), max(int((x0 + fov) * sx), int(x0 * sx) + 1)
                sub = row[:, tx0:tx1]
                if sub.size and sub.mean() >= job["tissue_ratio"]:
                    cands.append((x0, y0))
        if len(cands) < job.get("min_candidates", 50):
            return {"name": job["name"], "error": f"only {len(cands)} candidate tiles (tissue detection failed?)"}
        rng = slide_rng(job["name"], job["seed"])
        rng.shuffle(cands)
        tumor = exclude = None
        if job["xml"]:
            tumor, exclude = parse_polygons(job["xml"])
        mask = None
        if job.get("mask_labels") == "radboud" and job.get("mask"):
            mask = openslide.OpenSlide(job["mask"])
            if mask.dimensions != s.dimensions:
                return {"name": job["name"], "error": f"mask dims {mask.dimensions} != image {s.dimensions}"}
        chosen, n_t, n_n = [], 0, 0
        n_cls = {}
        for (x0, y0) in cands:
            if mask is not None:
                m = np.asarray(mask.read_region((x0, y0), 0, (fov, fov)))[..., 0]
                fg = m[m > 0]
                if fg.size < job["tissue_ratio"] * m.size:
                    continue
                vals, cnts = np.unique(fg, return_counts=True)
                merged = {}
                for v, c in zip(vals, cnts):
                    k = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3}.get(int(v))
                    if k is not None:
                        merged[k] = merged.get(k, 0) + int(c)
                if not merged:
                    continue
                lab, c = max(merged.items(), key=lambda kv: kv[1])
                if c < job["mask_majority"] * fg.size:
                    continue
                if n_cls.get(lab, 0) >= job["per_class_cap"]:
                    continue
                n_cls[lab] = n_cls.get(lab, 0) + 1
                chosen.append((x0, y0, lab))
                if len(chosen) >= job["patches_per_slide"]:
                    break
            elif tumor is not None:
                lab = 1 if tile_is_tumor(x0 + fov / 2, y0 + fov / 2, tumor, exclude) else 0
                if lab == 1:
                    if n_t >= job["tumor_cap"]:
                        continue
                    n_t += 1
                else:
                    if n_n >= job["normal_cap"]:
                        continue
                    n_n += 1
                chosen.append((x0, y0, lab))
                if n_t >= job["tumor_cap"] and n_n >= job["normal_cap"]:
                    break
            else:
                chosen.append((x0, y0, 0))
                if len(chosen) >= job["patches_per_slide"]:
                    break
        patches = np.empty((len(chosen), ps, ps, 3), dtype=np.uint8)
        for i, (x0, y0, _) in enumerate(chosen):
            im = s.read_region((x0, y0), 0, (fov, fov)).convert("RGB")
            if fov != ps:
                im = im.resize((ps, ps), Image.Resampling.LANCZOS)
            patches[i] = np.asarray(im)
        s.close()
        if mask is not None:
            mask.close()
        return {"name": job["name"], "mpp": mpp, "fov": fov, "n_cand": len(cands),
                "patches": patches, "labels": np.asarray([c[2] for c in chosen], dtype=np.int64),
                "coords": np.asarray([(c[0], c[1]) for c in chosen], dtype=np.int64).reshape(-1, 2),
                "secs": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"name": job["name"], "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True,
                    help="JSON list of {path, name, patient?, group?, xml?} or a manifest from "
                         "select_tcga_slides.py (use --cohort/--role to filter)")
    ap.add_argument("--cohort"); ap.add_argument("--role")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patches_per_slide", type=int, default=300)
    ap.add_argument("--tumor_cap", type=int, default=250)
    ap.add_argument("--normal_cap", type=int, default=250)
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--target_mpp", type=float, default=0.5)
    ap.add_argument("--tissue_ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mask_labels", choices=["radboud"], default=None)
    ap.add_argument("--mask_majority", type=float, default=0.6)
    ap.add_argument("--per_class_cap", type=int, default=100)
    args = ap.parse_args()

    with open(args.slides) as f:
        spec = json.load(f)
    slides = spec["slides"] if isinstance(spec, dict) else spec
    if args.cohort:
        slides = [r for r in slides if r.get("cohort") == args.cohort]
    if args.role:
        slides = [r for r in slides if r.get("role") == args.role]
    slides = sorted(slides, key=lambda r: r["name"])
    if not slides:
        raise SystemExit("no slides selected")
    print(f"[extract_v5] {len(slides)} slides -> {args.out}  target_mpp={args.target_mpp} "
          f"patch={args.patch_size} workers={args.workers}", flush=True)

    jobs = [dict(path=r["path"], name=r["name"], xml=r.get("xml"), mask=r.get("mask"),
                 mask_labels=args.mask_labels, mask_majority=args.mask_majority,
                 per_class_cap=args.per_class_cap,
                 patches_per_slide=args.patches_per_slide, tumor_cap=args.tumor_cap,
                 normal_cap=args.normal_cap, patch_size=args.patch_size,
                 target_mpp=args.target_mpp, tissue_ratio=args.tissue_ratio, seed=args.seed)
            for r in slides]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ps = args.patch_size
    names, patients, groups, mpps, errors = [], [], [], [], []
    n_w = 0
    t0 = time.time()
    with h5py.File(out, "w") as h5, Pool(args.workers) as pool:
        P = h5.create_dataset("patches", (0, ps, ps, 3), maxshape=(None, ps, ps, 3),
                              dtype="uint8", chunks=(16, ps, ps, 3))
        L = h5.create_dataset("labels", (0,), maxshape=(None,), dtype="int64")
        S = h5.create_dataset("slide_ids", (0,), maxshape=(None,), dtype="int64")
        C = h5.create_dataset("coords", (0, 2), maxshape=(None, 2), dtype="int64")
        for k, (r, res) in enumerate(zip(slides, pool.imap(extract_one, jobs))):
            if "error" in res:
                print(f"[{k+1}/{len(slides)}] {res['name']}: SKIP {res['error']}", flush=True)
                errors.append((res["name"], res["error"])); continue
            n = len(res["patches"])
            if n == 0:
                print(f"[{k+1}/{len(slides)}] {res['name']}: 0 tiles", flush=True); continue
            sidx = len(names)
            for ds, arr in ((P, res["patches"]), (L, res["labels"]),
                            (S, np.full(n, sidx, dtype=np.int64)), (C, res["coords"])):
                ds.resize(n_w + n, axis=0); ds[n_w:n_w + n] = arr
            n_w += n
            names.append(r["name"]); patients.append(r.get("patient", r["name"]))
            groups.append(str(r.get("group", r.get("tss", r.get("center", "")))))
            mpps.append(res["mpp"])
            print(f"[{k+1}/{len(slides)}] {res['name']}: {n} tiles (tumor={int(res['labels'].sum())}) "
                  f"mpp={res['mpp']:.3f} fov={res['fov']} cand={res['n_cand']} {res['secs']:.0f}s "
                  f"| total {n_w} elapsed {time.time()-t0:.0f}s", flush=True)
        h5.attrs["patch_size"] = ps
        h5.attrs["target_mpp"] = args.target_mpp
        h5.attrs["slide_names"] = np.array(names, dtype="S")
        h5.attrs["slide_patients"] = np.array(patients, dtype="S")
        h5.attrs["slide_groups"] = np.array(groups, dtype="S")
        h5.attrs["slide_mpp"] = np.array(mpps, dtype=np.float64)
        h5.attrs["source_manifest"] = str(args.slides)
        h5.attrs["args"] = json.dumps(vars(args))
    print(f"[done] {n_w} tiles from {len(names)} slides ({len(errors)} skipped) -> {out} "
          f"in {time.time()-t0:.0f}s", flush=True)
    if errors:
        print("skipped:", errors, flush=True)


if __name__ == "__main__":
    main()
