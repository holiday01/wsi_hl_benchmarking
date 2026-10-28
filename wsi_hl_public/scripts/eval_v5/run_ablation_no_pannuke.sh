#!/usr/bin/env bash
# Targeted ablation: ViT-B students of H-optimus-0 (14-px patch) and Phikon (16-px patch), seed 42,
# distilled without the PanNuke folds 1-2 (corpus tag "pannuke"), then evaluated on every task
# exactly as the main runs. Waits for the PANDA whole-biopsy feature stage to release the GPU.
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
ROOT=/data/wsi_hl_v5
A=outputs/v5_ablation/no_pannuke
LOG=$A/logs; mkdir -p $LOG
say() { echo "[$(date -Is)] $*" | tee -a $LOG/run.log; }
until grep -q "features done" outputs/v5_sens/panda_tissue/run.log 2>/dev/null; do sleep 120; done
say "start training"
for t in h-optimus-0 phikon; do
  d=$A/$t/vit-base/seed42; mkdir -p $d
  $PY scripts/distill_v5.py --corpus $ROOT/corpus --cache_root $ROOT/teacher_cache --teacher $t \
     --student vit_base_patch16_224 --seed 42 --output $d --num_workers 5 --exclude_tags pannuke > $d/train.log 2>&1 &
  sleep 150
done
wait
say "training done"
for d in $A/*/vit-base/seed42; do
  grep -q '"finished": true' $d/training_log.json || { say "FAIL training $d"; continue; }
  for ds in c16_annotated_270 kather_msi crc_val_7k tcga_brca_center camelyon17_annotated50 panda_radboud panda_radboud_tissue; do
    $PY scripts/eval_v5/extract_features.py --dataset $ds --model student:$d >> $LOG/feat.log 2>&1 || say "FAIL feat $ds $d"
  done
  $PY scripts/eval_v5/pannuke_seg.py cache --model student:$d >> $LOG/feat.log 2>&1 || say "FAIL pannuke cache $d"
done
say "features done"
for d in $A/*/vit-base/seed42; do
  t=$(basename $(dirname $(dirname $d))); key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
  for task in c16 c17_loco msi panda crc7k center panda_biopsy panda_biopsy_tissue; do
    $PY scripts/eval_v5/run_probes.py --task $task --teacher $t --student_key $key --out $d/probe_$task.json >> $LOG/probes.log 2>&1 || say "FAIL $task $d"
  done
  $PY scripts/eval_v5/pannuke_seg.py probe --teacher $t --student_key $key --out $d/probe_pannuke_seg.json >> $LOG/probes.log 2>&1 || say "FAIL pannuke $d"
done
say "all done"; date -Is > $A/done
