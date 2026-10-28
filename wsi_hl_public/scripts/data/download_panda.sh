#!/usr/bin/env bash
# Download the Radboud biopsies + masks listed in data/manifests/panda_radboud_v5.json
# from the PANDA Kaggle competition (needs ~/.kaggle credentials and accepted rules).
# Kaggle rate-limits per-file downloads (HTTP 429 after ~400 files in 8 min at 6 workers),
# so this runs SERIALLY with a pause between files and a long back-off on 429.
# Each file arrives as <name>.zip and is unzipped in place. Resumable: existing .tiff skipped.
set -u
cd "$(dirname "$0")/../.."
K=/home/holiday/venvs/wsi_hl/bin/kaggle
PY=/home/holiday/venvs/wsi_hl/bin/python
RAW=/data2/wsi_hl_raw/panda
PAUSE="${1:-3}"
mkdir -p "$RAW/train_images" "$RAW/train_label_masks"
one() {  # one <subdir> <filename>
  local d="$1" f="$2" err
  [ -f "$RAW/$d/$f" ] && { echo "[have] $d/$f"; return 0; }
  for try in 1 2 3 4 5 6; do
    err=$($K competitions download -c prostate-cancer-grade-assessment -f "$d/$f" -p "$RAW/$d" 2>&1 | grep -v "outdated")
    if [ -f "$RAW/$d/$f.zip" ]; then $PY -m zipfile -e "$RAW/$d/$f.zip" "$RAW/$d" && rm -f "$RAW/$d/$f.zip"; fi
    [ -f "$RAW/$d/$f" ] && { echo "[ok] $d/$f"; return 0; }
    if echo "$err" | grep -q "429"; then echo "[429] $d/$f  backing off 600s (try $try)"; sleep 600
    elif echo "$err" | grep -q "404"; then echo "[404] $d/$f (not in competition data)"; return 1
    else echo "[err] $d/$f: $(echo "$err" | tail -1 | cut -c1-120) (try $try)"; sleep $((try * 20)); fi
  done
  echo "[FAIL] $d/$f"; return 1
}
$PY -c "
import json; m=json.load(open('data/manifests/panda_radboud_v5.json'))
for r in m['slides']:
    print('train_images', r['image_id']+'.tiff'); print('train_label_masks', r['image_id']+'_mask.tiff')
" | while read -r d f; do
  one "$d" "$f"
  sleep "$PAUSE"
done
NI=$(ls $RAW/train_images/*.tiff 2>/dev/null | wc -l); NM=$(ls $RAW/train_label_masks/*.tiff 2>/dev/null | wc -l)
NT=$($PY -c "import json; print(json.load(open('data/manifests/panda_radboud_v5.json'))['n'])")
echo "[done] $(date -Is) images=$NI/$NT masks=$NM/$NT (PANDA ships ~1% of biopsies without a mask; those are skipped at extraction)"
[ "$NI" -eq "$NT" ]
