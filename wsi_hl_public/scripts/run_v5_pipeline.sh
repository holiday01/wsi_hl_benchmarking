#!/usr/bin/env bash
# =====================================================================
# v5 redo pipeline, unattended. Every stage writes a .done marker in
# $STATE and is skipped on re-run, so it can be relaunched any time.
#
#   S1  wait for TCGA extractions (run_tcga_extraction_v5.sh)      CPU/NAS
#   S2  build the distillation corpus memmap + manifest            CPU
#   S3  cache the 12 teachers' targets (one at a time)             GPU
#   S4  108 student runs (12 teachers x 3 students x 3 seeds),     GPU
#       3 concurrent, teacher-major order so the cache stays hot
#   S5  evaluation feature caches: every teacher + every student    GPU
#       on every tile set that exists (C17/PANDA join when ready)
#   S6  GATE: research/preregistration_v5.md must be committed      -
#       (ANALYSIS_PLAN_GATE=0 skips it in a code-only checkout)
#   S7  probes + equivalence statistics                             CPU
# =====================================================================
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/holiday/venvs/wsi_hl/bin/python}
ROOT=${WSI_V5_ROOT:-/data/wsi_hl_v5}
STATE=$ROOT/pipeline_state; mkdir -p "$STATE" "$ROOT/corpus" "$ROOT/teacher_cache" outputs/v5
LOG=$STATE/pipeline.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date -Is)] $*"; }
stage_done() { [ -f "$STATE/$1.done" ]; }
mark_done() { date -Is > "$STATE/$1.done"; say "STAGE $1 COMPLETE"; }

TEACHERS=(phikon hibou-b conch phikon-v2 uni hibou-l virchow virchow2 uni2-h h-optimus-0 prov-gigapath midnight)
STUDENTS=(vit_tiny_patch16_224 vit_small_patch16_224 vit_base_patch16_224)
SEEDS=(42 2025 7)
short() { case "$1" in vit_tiny_patch16_224) echo vit-tiny;; vit_small_patch16_224) echo vit-small;; vit_base_patch16_224) echo vit-base;; esac; }

# ---------------- S1 ----------------
if ! stage_done S1_tcga; then
  say "S1 waiting for TCGA extraction"
  until grep -q "^\[all done\]" $ROOT/patches/run_all.log 2>/dev/null; do sleep 300; done
  for n in tcga_brca_distill_v5 tcga_lung_distill_v5 tcga_lihc_distill_v5 tcga_brca_eval_v5; do
    [ -f $ROOT/patches/$n.done ] || { say "S1 FAILED: $n not done"; exit 1; }
  done
  mark_done S1_tcga
fi

# ---------------- S2 ----------------
if ! stage_done S2_corpus; then
  say "S2 build corpus"
  $PY scripts/data/build_corpus.py --config configs/corpus_v5.json --out_dir $ROOT/corpus --workers 24 \
    && mark_done S2_corpus || { say "S2 FAILED"; exit 1; }
fi

# ---------------- S3 ----------------
if ! stage_done S3_cache; then
  say "S3 teacher caches"
  for t in "${TEACHERS[@]}"; do
    [ -f $ROOT/teacher_cache/$t/meta.json ] && grep -q '"verified": true' $ROOT/teacher_cache/$t/meta.json && { say "  cache $t exists"; continue; }
    say "  caching $t"
    $PY scripts/cache_teacher_targets.py --corpus $ROOT/corpus --teacher $t --out_root $ROOT/teacher_cache \
       --batch_size 128 --num_workers 8 > $ROOT/teacher_cache/$t.log 2>&1 || { say "S3 FAILED on $t (see $ROOT/teacher_cache/$t.log)"; exit 1; }
  done
  mark_done S3_cache
fi

# ---------------- S4 ----------------
# Measured peak GPU memory (fp32, batch 128, 1536-d targets): tiny 4.9 GB, small 8.6 GB,
# base 16.7 GB allocated, plus ~1 GB CUDA context each. tiny+small+base does not fit in
# 32 GB, so every launch waits until the GPU has the run's requirement free. The lock is
# shared with run_teacher_side_v5.sh so two launchers never judge the same free memory.
need_mb() { case "$1" in vit_tiny*) echo 6500;; vit_small*) echo 10500;; *) echo 19000;; esac; }
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
gpu_acquire() {  # need_mb -> returns holding fd 9 with at least need_mb free
  local need=$1
  exec 9> $STATE/gpu_launch.lock
  while true; do
    until [ "$(gpu_free_mb)" -ge "$need" ]; do sleep 60; done
    flock 9
    [ "$(gpu_free_mb)" -ge "$need" ] && return 0
    flock -u 9
  done
}
run_one() {  # teacher student seed
  local t=$1 s=$2 seed=$3 d=outputs/v5/$1/$(short $2)/seed$3
  if [ -f $d/training_log.json ] && grep -q '"finished": true' $d/training_log.json; then echo "[skip] $d"; return; fi
  mkdir -p $d
  local need; need=$(need_mb $s)
  gpu_acquire $need
  echo "[run] $d $(date -Is) free=$(gpu_free_mb)MB need=${need}MB"
  $PY scripts/distill_v5.py --corpus $ROOT/corpus --cache_root $ROOT/teacher_cache --teacher $t --student $s \
     --seed $seed --output $d --num_workers 5 > $d/train.log 2>&1 &
  local pid=$!
  sleep 150; flock -u 9; exec 9>&-        # the run reaches its peak before the next launch is judged
  wait $pid || echo "[FAIL] $d $(date -Is)"
}
export -f run_one short need_mb gpu_free_mb gpu_acquire; export PY ROOT STATE
if ! stage_done S4_distill; then
  say "S4 student runs (3 concurrent)"
  for t in "${TEACHERS[@]}"; do for seed in "${SEEDS[@]}"; do for s in "${STUDENTS[@]}"; do echo "$t $s $seed"; done; done; done \
    | xargs -P 3 -L 1 bash -c 'run_one $0 $1 $2'
  n_ok=$(grep -l '"finished": true' outputs/v5/*/*/seed*/training_log.json 2>/dev/null | wc -l)
  say "  finished runs: $n_ok / $(( ${#TEACHERS[@]} * ${#STUDENTS[@]} * ${#SEEDS[@]} ))"
  [ "$n_ok" -ge $(( ${#TEACHERS[@]} * ${#STUDENTS[@]} * ${#SEEDS[@]} )) ] && mark_done S4_distill || { say "S4 incomplete; rerun the pipeline"; exit 1; }
fi

# ---------------- S5 ----------------
DATASETS=(c16_annotated_270 kather_msi crc_val_7k tcga_brca_center camelyon17_annotated50 panda_radboud)
ds_ready() { case "$1" in camelyon17_annotated50) [ -f $ROOT/patches/camelyon17_annotated50_v5.done ];; panda_radboud) [ -f $ROOT/patches/panda_radboud_v5.done ];; tcga_brca_center) [ -f $ROOT/patches/tcga_brca_eval_v5.done ];; *) true;; esac; }
if ! stage_done S5_features; then
  say "S5 evaluation features"
  say "  waiting for CAMELYON17 and PANDA tile sets"
  until ds_ready camelyon17_annotated50 && ds_ready panda_radboud; do sleep 300; done
  for ds in "${DATASETS[@]}"; do
    for t in "${TEACHERS[@]}"; do
      $PY scripts/eval_v5/extract_features.py --dataset $ds --model teacher:$t >> $STATE/features.log 2>&1 || say "  [FAIL] $ds teacher:$t"
    done
    for d in outputs/v5/*/*/seed*; do
      $PY scripts/eval_v5/extract_features.py --dataset $ds --model student:$d >> $STATE/features.log 2>&1 || say "  [FAIL] $ds student:$d"
    done
    say "  features done: $ds"
  done
  mark_done S5_features
fi

# ---------------- S5b PanNuke patch-token caches ----------------
if ! stage_done S5b_pannuke_tokens; then
  say "S5b PanNuke token caches"
  for t in "${TEACHERS[@]}"; do
    $PY scripts/eval_v5/pannuke_seg.py cache --model teacher:$t >> $STATE/pannuke_cache.log 2>&1 || say "  [FAIL] pannuke teacher:$t"
  done
  for d in outputs/v5/*/*/seed*; do
    $PY scripts/eval_v5/pannuke_seg.py cache --model student:$d >> $STATE/pannuke_cache.log 2>&1 || say "  [FAIL] pannuke student:$d"
  done
  mark_done S5b_pannuke_tokens
fi

# ---------------- S6 gate ----------------
if [ "${ANALYSIS_PLAN_GATE:-1}" = 1 ] && ! git log --oneline -- research/preregistration_v5.md | grep -q .; then
  say "S6 GATE: research/preregistration_v5.md is not committed; stopping before any probe statistics."
  exit 0
fi

# ---------------- S7 ----------------
if ! stage_done S7_probes; then
  say "S7 probes"
  for task in c16 c17_loco msi panda crc7k center; do
    for d in outputs/v5/*/*/seed*; do
      t=$(basename $(dirname $(dirname $d))); key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
      out=$d/probe_$task.json; [ -f $out ] && continue
      $PY scripts/eval_v5/run_probes.py --task $task --teacher $t --student_key $key --out $out >> $STATE/probes.log 2>&1 || say "  [FAIL] $task $d"
    done
    say "  probes done: $task"
  done
  mark_done S7_probes
fi

# ---------------- S7b PanNuke segmentation probes ----------------
if ! stage_done S7b_pannuke_probes; then
  say "S7b PanNuke probes"
  for d in outputs/v5/*/*/seed*; do
    t=$(basename $(dirname $(dirname $d))); key="student__$(echo ${d#outputs/} | sed 's#/#_#g')"
    out=$d/probe_pannuke_seg.json; [ -f $out ] && continue
    $PY scripts/eval_v5/pannuke_seg.py probe --teacher $t --student_key $key --out $out >> $STATE/probes.log 2>&1 || say "  [FAIL] pannuke $d"
  done
  mark_done S7b_pannuke_probes
fi

# ---------------- S8 seed-pooled summary ----------------
say "S8 aggregate"
$PY scripts/eval_v5/aggregate.py --root outputs/v5 --out outputs/v5/summary >> $STATE/aggregate.log 2>&1 || say "  [FAIL] aggregate"
say "pipeline complete"
