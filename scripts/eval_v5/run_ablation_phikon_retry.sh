#!/usr/bin/env bash
# Phikon ablation student, relaunched with the identical configuration after its first launch ran out of GPU memory
# while sharing the GPU with the H-optimus-0 ablation run (no epoch completed). Runs after the H-optimus-0 training,
# then evaluates exactly as run_ablation_no_pannuke.sh.
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
ROOT=/data/wsi_hl_v5
A=outputs/v5_ablation/no_pannuke
LOG=$A/logs
say() { echo "[$(date -Is)] $*" | tee -a $LOG/run.log; }
until grep -q "training done" $LOG/run.log 2>/dev/null; do sleep 60; done
t=phikon; d=$A/$t/vit-base/seed42
[ -f $d/train.log ] && mv $d/train.log $d/train_oom_first_launch.log
say "phikon relaunch"
$PY scripts/distill_v5.py --corpus $ROOT/corpus --cache_root $ROOT/teacher_cache --teacher $t \
   --student vit_base_patch16_224 --seed 42 --output $d --num_workers 5 --exclude_tags pannuke > $d/train.log 2>&1
grep -q '"finished": true' $d/training_log.json || { say "FAIL phikon retry training"; exit 1; }
say "phikon training done"
until [ -f $A/done ]; do sleep 60; done     # let the main runner finish its H-optimus-0 evaluation first
for ds in c16_annotated_270 kather_msi crc_val_7k tcga_brca_center camelyon17_annotated50 panda_radboud panda_radboud_tissue; do
  $PY scripts/eval_v5/extract_features.py --dataset $ds --model student:$d >> $LOG/feat.log 2>&1 || say "FAIL feat $ds $d"
done
$PY scripts/eval_v5/pannuke_seg.py cache --model student:$d >> $LOG/feat.log 2>&1 || say "FAIL pannuke cache $d"
key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
for task in c16 c17_loco msi panda crc7k center panda_biopsy panda_biopsy_tissue; do
  $PY scripts/eval_v5/run_probes.py --task $task --teacher $t --student_key $key --out $d/probe_$task.json >> $LOG/probes.log 2>&1 || say "FAIL $task $d"
done
$PY scripts/eval_v5/pannuke_seg.py probe --teacher $t --student_key $key --out $d/probe_pannuke_seg.json >> $LOG/probes.log 2>&1 || say "FAIL pannuke $d"
say "phikon all done"; date -Is > $A/done_phikon
