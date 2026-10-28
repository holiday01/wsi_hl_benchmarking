#!/usr/bin/env python3
"""Deterministic Radboud subset of PANDA for the Gleason-pattern tile task.
Radboud masks carry pattern labels (0 bg, 1 stroma, 2 benign epithelium, 3/4/5 Gleason
3/4/5); Karolinska masks do not, so only Radboud biopsies are used. Biopsies are ranked
by sha256("panda:<image_id>") within each ISUP grade and the first n_per_grade are taken.
  python scripts/data/select_panda.py --train_csv /data2/wsi_hl_raw/panda/train.csv \
      --n_per_grade 270 --out data/manifests/panda_radboud_v5.json
"""
import argparse, csv, hashlib, json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--n_per_grade", type=int, default=270)
    ap.add_argument("--raw", default="/data2/wsi_hl_raw/panda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by_grade = defaultdict(list)
    with open(args.train_csv) as f:
        for r in csv.DictReader(f):
            if r["data_provider"] != "radboud":
                continue
            h = int(hashlib.sha256(f"panda:{r['image_id']}".encode()).hexdigest(), 16) / float(1 << 256)
            by_grade[int(r["isup_grade"])].append(dict(image_id=r["image_id"], isup=int(r["isup_grade"]),
                                                     gleason=r["gleason_score"], bucket=h))
    sel = []
    for g in sorted(by_grade):
        rows = sorted(by_grade[g], key=lambda r: (r["bucket"], r["image_id"]))[:args.n_per_grade]
        sel += rows
    for r in sel:
        r["path"] = f"{args.raw}/train_images/{r['image_id']}.tiff"
        r["mask"] = f"{args.raw}/train_label_masks/{r['image_id']}_mask.tiff"
        r["name"] = f"{r['image_id']}.tiff"; r["group"] = r["image_id"]
    sel.sort(key=lambda r: r["image_id"])
    body = {"params": vars(args), "provider": "radboud", "n": len(sel), "slides": sel}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    json.dump(body, open(args.out, "w"), indent=1)
    print(f"selected {len(sel)} radboud biopsies ({args.n_per_grade}/grade) -> {args.out}")


if __name__ == "__main__":
    main()
