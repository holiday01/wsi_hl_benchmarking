#!/usr/bin/env bash
# Whole-biopsy PANDA grading with label-independent tiles: up to 120 tissue tiles per biopsy
# (Otsu tissue mask, no Gleason mask), same 1,568 biopsies and biopsy split as panda_biopsy.
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
OUT=/data/wsi_hl_v5/patches/panda_radboud_tissue_v5b.h5
LOG=outputs/v5_sens/panda_tissue; mkdir -p $LOG
say() { echo "[$(date -Is)] $*" | tee -a $LOG/run.log; }
[ -f $OUT.done ] || { $PY scripts/data/extract_patches_v5.py --slides data/manifests/panda_radboud_tissue_v5b.json \
    --out $OUT --patches_per_slide 120 --workers 16 > $LOG/extract.log 2>&1 && date -Is > $OUT.done; } || { say "extract FAIL"; exit 1; }
say "extract done"
DS=panda_radboud_tissue
for t in $(ls outputs/v5); do
  [ -d outputs/v5/$t/vit-base ] || continue
  $PY scripts/eval_v5/extract_features.py --dataset $DS --model teacher:$t >> $LOG/feat.log 2>&1 || say "FAIL teacher $t"
done
$PY scripts/eval_v5/extract_features.py --dataset $DS --model teacher:h0-mini --batch_size 128 >> $LOG/feat.log 2>&1 || say "FAIL h0-mini"
for d in outputs/v5/*/*/seed*; do
  $PY scripts/eval_v5/extract_features.py --dataset $DS --model student:$d >> $LOG/feat.log 2>&1 || say "FAIL student $d"
done
say "features done"
for d in outputs/v5/*/*/seed*; do
  t=$(basename $(dirname $(dirname $d))); key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
  out=$d/probe_panda_biopsy_tissue.json; [ -f $out ] && continue
  echo "$t $key $out"
done | OMP_NUM_THREADS=1 xargs -P 40 -L 1 bash -c "$PY scripts/eval_v5/run_probes.py --task panda_biopsy_tissue --teacher \$0 --student_key \$1 --out \$2 >> $LOG/probes.log 2>&1"
EXT=outputs/v5_external/h-optimus-0/h0-mini/seed0
$PY scripts/eval_v5/run_probes.py --task panda_biopsy_tissue --teacher h-optimus-0 --student_key teacher__h0-mini --out $EXT/probe_panda_biopsy_tissue.json >> $LOG/probes.log 2>&1 || say "FAIL ext"
$PY scripts/eval_v5/aggregate.py --tasks panda_biopsy_tissue --out $LOG/summary >> $LOG/run.log 2>&1
say "all done"; date -Is > $LOG/done
