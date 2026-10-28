#!/usr/bin/env bash
# Wait for download_panda.sh to finish, then extract Gleason-pattern tiles
# (Radboud masks, 0.5 um/px, <=50 tiles per class and <=120 per biopsy).
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-/home/holiday/venvs/wsi_hl/bin/python}
RAW=${PANDA_RAW:-/data2/wsi_hl_raw/panda}
OUT=${WSI_V5_ROOT:-/data/wsi_hl_v5}/patches
until grep -qE "^\[done\].*images=([0-9]+)/\1 " "$RAW/download.log" 2>/dev/null; do sleep 120; done
$PY scripts/data/extract_patches_v5.py --slides data/manifests/panda_radboud_v5.json \
   --out $OUT/panda_radboud_v5.h5 --mask_labels radboud --patches_per_slide 120 --per_class_cap 50 \
   --workers 8 && date -Is > $OUT/panda_radboud_v5.done
