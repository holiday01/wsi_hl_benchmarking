#!/usr/bin/env bash
# Wait for download_camelyon17.sh to finish with all 150 files md5-verified, then build
# the manifest and extract tumour/normal tiles at 0.5 um/px (XML lesion polygons,
# <=250 tumour + <=250 normal tiles per slide, like the CAMELYON16 annotated set).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/home/holiday/venvs/wsi_hl/bin/python}
RAW=${C17_RAW:-/data2/wsi_hl_raw/camelyon17}
OUT=${WSI_V5_ROOT:-/data/wsi_hl_v5}/patches
until grep -q "^\[done\].*checked=150 mismatches=0" "$RAW/download.log" 2>/dev/null \
      && [ "$(grep -c '^\[md5 ok\]' "$RAW/download.log")" -eq 150 ]; do sleep 120; done
$PY scripts/data/make_camelyon17_manifest.py --raw $RAW --out data/manifests/camelyon17_annotated50_v5.json
$PY scripts/data/extract_patches_v5.py --slides data/manifests/camelyon17_annotated50_v5.json \
   --out $OUT/camelyon17_annotated50_v5.h5 --tumor_cap 250 --normal_cap 250 --workers 6 \
   && grep -q "\[done\] .* from 50 slides (0 skipped)" $OUT/camelyon17_annotated50_v5.log \
   && date -Is > $OUT/camelyon17_annotated50_v5.done
