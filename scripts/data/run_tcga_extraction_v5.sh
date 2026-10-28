#!/usr/bin/env bash
# Extract all v5 TCGA patch sets from the manifest, sequentially (NAS-bound), 20 workers.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/home/holiday/venvs/wsi_hl/bin/python}
M=data/manifests/tcga_slides_v5.json
OUT=${WSI_V5_ROOT:-/data/wsi_hl_v5}/patches
mkdir -p "$OUT"
run() {  # name cohort role patches_per_slide
  [ -f "$OUT/$1.done" ] && { echo "[skip] $1"; return; }
  $PY scripts/data/extract_patches_v5.py --slides $M --cohort $2 --role $3 \
     --out "$OUT/$1.h5" --patches_per_slide $4 --workers 20 > "$OUT/$1.log" 2>&1 \
     && date -Is > "$OUT/$1.done"
}
run tcga_brca_distill_v5 brca distill 300
run tcga_lung_distill_v5 lung distill 200
run tcga_lihc_distill_v5 lihc distill 200
run tcga_brca_eval_v5    brca eval_center_probe 200
echo "[all done] $(date -Is)"
