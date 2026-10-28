#!/usr/bin/env python3
"""Manifest for the 50 lesion-annotated CAMELYON17 slides: path, XML, centre, patient,
slide-level stage (from stages.csv). Feeds extract_patches_v5.py (--slides).
  python scripts/data/make_camelyon17_manifest.py --raw /data2/wsi_hl_raw/camelyon17 \
      --out data/manifests/camelyon17_annotated50_v5.json
"""
import argparse, csv, hashlib, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--list", default=str(Path(__file__).parent / "camelyon17_annotated50.txt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.raw)
    stages = {}
    with open(raw / "stages.csv") as f:
        for r in csv.DictReader(f):
            stages[r["patient"]] = (r["stage"], int(r["center"]))
    rows = []
    for s in open(args.list).read().split():
        img, xml, mask = raw / "images" / f"{s}.tif", raw / "annotations" / f"{s}.xml", raw / "masks" / f"{s}_mask.tif"
        st, ce = stages[f"{s}.tif"]
        rows.append({"name": f"{s}.tif", "path": str(img), "xml": str(xml), "mask": str(mask),
                     "patient": s.split("_node")[0], "center": ce, "group": f"center{ce}",
                     "stage": st, "present": img.exists() and xml.exists()})
    rows.sort(key=lambda r: r["name"])
    body = {"source": "s3://camelyon-dataset/CAMELYON17 (public)", "n": len(rows), "slides": rows}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    json.dump(body, open(args.out, "w"), indent=1)
    from collections import Counter
    print(Counter((r["center"], r["stage"]) for r in rows))
    print("present:", sum(r["present"] for r in rows), "/", len(rows), "->", args.out)


if __name__ == "__main__":
    main()
