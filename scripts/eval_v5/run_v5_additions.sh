#!/usr/bin/env bash
# Additions registered in research/preregistration_v5.md §6 on 2026-10-17 (before S7):
#   A. external teacher-student pair H-optimus-0 -> H0-mini (public ViT-B student, Filiot 2025)
#      on every task incl. pannuke_seg and panda_biopsy; summarised separately in outputs/v5_external
#   B. secondary task panda_biopsy for all 108 runs
#   C. secondary descriptive analyses (scripts/eval_v5/secondary_analyses.py) after S8
# Runs beside scripts/run_v5_pipeline.sh (never edit that one while it runs). Waits on the
# pipeline's state files; every step is idempotent.
set -u
cd /home/holiday/wsi_hl
PY=/home/holiday/venvs/wsi_hl/bin/python
ROOT=/data/wsi_hl_v5; STATE=$ROOT/pipeline_state; LOG=$STATE/additions.log
EXT=outputs/v5_external/h-optimus-0/h0-mini/seed0
say() { echo "[$(date -Is)] $*" | tee -a $LOG; }
exec > >(tee -a $LOG) 2>&1
say "additions start"

until [ -f $STATE/S5_features.done ]; do sleep 600; done
say "S5 done -> A1 H0-mini features"
for ds in c16_annotated_270 kather_msi crc_val_7k tcga_brca_center camelyon17_annotated50 panda_radboud; do
  $PY scripts/eval_v5/extract_features.py --dataset $ds --model teacher:h0-mini --batch_size 128 || say "  [FAIL] $ds teacher:h0-mini"
done
until [ -f $STATE/S5b_pannuke_tokens.done ]; do sleep 300; done
$PY scripts/eval_v5/pannuke_seg.py cache --model teacher:h0-mini || say "  [FAIL] pannuke teacher:h0-mini"

say "A2 external pair probes"
mkdir -p $EXT
for task in c16 c17_loco msi panda panda_biopsy crc7k center; do
  out=$EXT/probe_$task.json; [ -f $out ] && continue
  $PY scripts/eval_v5/run_probes.py --task $task --teacher h-optimus-0 --student_key teacher__h0-mini --out $out || say "  [FAIL] ext $task"
done
[ -f $EXT/probe_pannuke_seg.json ] || $PY scripts/eval_v5/pannuke_seg.py probe --teacher h-optimus-0 --student_key teacher__h0-mini --out $EXT/probe_pannuke_seg.json || say "  [FAIL] ext pannuke"
$PY scripts/eval_v5/aggregate.py --root outputs/v5_external --out outputs/v5_external/summary || say "  [FAIL] ext aggregate"
say "A done"

say "B panda_biopsy probes for all runs"
for d in outputs/v5/*/*/seed*; do
  t=$(basename $(dirname $(dirname $d))); key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
  out=$d/probe_panda_biopsy.json; [ -f $out ] && continue
  $PY scripts/eval_v5/run_probes.py --task panda_biopsy --teacher $t --student_key $key --out $out || say "  [FAIL] panda_biopsy $d"
done
say "B done"

until grep -q 'pipeline complete' $STATE/pipeline.log; do sleep 600; done
say "S8 done -> re-aggregate with panda_biopsy, then C secondary analyses"
$PY scripts/eval_v5/aggregate.py --root outputs/v5 --out outputs/v5/summary || say "  [FAIL] aggregate"
$PY scripts/eval_v5/secondary_analyses.py --root outputs/v5 --summary outputs/v5/summary --out outputs/v5/secondary || say "  [FAIL] secondary"
say "additions DONE"
