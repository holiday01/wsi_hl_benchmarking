#!/usr/bin/env bash
# Teacher-side work that does not depend on any student run, started while S3/S4 train:
#   1. S5 teacher feature caches for the tile sets that already exist (S5 later skips them)
#   2. bug-1 (legacy ImageNet-normalised) teacher features for the five affected teachers
#   3. S5b PanNuke teacher token caches
#   4. bug-impact quantification (scripts/eval_v5/teacher_bug_impact.py)
#   5. teacher-only reference probes on the pre-registered splits (teacher_reference.py)
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
LOG=/data/wsi_hl_v5/tmp/teacher_side.log
say(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
source <(grep '^TEACHERS=' scripts/run_v5_pipeline.sh)
LEGACY=(h-optimus-0 conch midnight hibou-b hibou-l)
READY=(crc_val_7k kather_msi tcga_brca_center c16_annotated_270)
until grep -qE "SMOKE OK|FAILED" /data/wsi_hl_v5/tmp/smoke_v5.log 2>/dev/null; do sleep 60; done
# GPU launches share the pipeline's lock and wait for free memory (S4 runs 2-3 students at once).
LOCK=/data/wsi_hl_v5/pipeline_state/gpu_launch.lock
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_run() {  # need_mb cmd... : launch once need_mb is free, hold the lock 90 s, then wait for the command
  local need=$1; shift
  exec 9> $LOCK
  while true; do
    until [ "$(gpu_free_mb)" -ge "$need" ]; do sleep 60; done
    flock 9
    [ "$(gpu_free_mb)" -ge "$need" ] && break
    flock -u 9
  done
  "$@" & local pid=$!
  sleep 90; flock -u 9; exec 9>&-
  wait $pid
}
say "teacher-side start"
for ds in "${READY[@]}"; do
  for t in "${TEACHERS[@]}"; do
    [ -f /data/wsi_hl_v5/features/$ds/teacher__$t/meta.json ] && continue      # done earlier; no GPU slot needed
    gpu_run 9000 $PY scripts/eval_v5/extract_features.py --dataset $ds --model teacher:$t --batch_size 64 >> $LOG.feat 2>&1 || say "  [FAIL] $ds teacher:$t"
  done
  for t in "${LEGACY[@]}"; do
    [ -f /data/wsi_hl_v5/features/$ds/teacher_legacy__$t/meta.json ] && continue
    gpu_run 9000 $PY scripts/eval_v5/extract_features.py --dataset $ds --model teacher_legacy:$t --batch_size 64 >> $LOG.feat 2>&1 || say "  [FAIL] $ds teacher_legacy:$t"
  done
  say "features done: $ds"
done
for t in "${TEACHERS[@]}"; do
  [ -f /data/wsi_hl_v5/features/pannuke/teacher__$t/tokens_fold3.npy ] && continue
  gpu_run 9000 $PY scripts/eval_v5/pannuke_seg.py cache --model teacher:$t >> $LOG.pannuke 2>&1 || say "  [FAIL] pannuke teacher:$t"
done
say "pannuke teacher caches done"
gpu_run 9000 $PY scripts/eval_v5/teacher_bug_impact.py --out research/teacher_bug_impact_v5 >> $LOG.bug 2>&1 && say "bug impact done" || say "  [FAIL] bug impact"
printf '%s\n' crc7k msi center c16 | xargs -P 4 -I{} sh -c "$PY scripts/eval_v5/teacher_reference.py --task {} >> $LOG.ref.{} 2>&1 || echo '  [FAIL] reference {}' >> $LOG"
say "teacher-side DONE"
