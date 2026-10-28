#!/usr/bin/env bash
# Probe-C sensitivity (C=0.1, 10) for the whole-biopsy PANDA task, mirroring run_sensitivity_v5b.sh.
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for d in outputs/v5/*/*/seed*; do
  t=$(basename $(dirname $(dirname $d))); rel=${d#outputs/v5/}; key="student__v5_$(echo $rel | sed 's#/#_#g')"
  for c in 0.1 10; do echo "C$c panda_biopsy_tissue $t $key $rel --C $c"; done
done | nice -n 10 xargs -P 24 -L 1 bash -c 'out=outputs/v5_sens/$0/$4/probe_$1.json; [ -f $out ] || '"$PY"' scripts/eval_v5/run_probes.py --task $1 --teacher $2 --student_key $3 --out $out "${@:5}" > /dev/null 2>>outputs/v5_sens/errors_tissue.log || echo "FAIL $*" >> outputs/v5_sens/errors_tissue.log'
for v in C0.1 C10; do
  $PY scripts/eval_v5/aggregate.py --root outputs/v5_sens/$v --tasks panda_biopsy_tissue --out outputs/v5_sens/$v/summary_tissue
done
date -Is > outputs/v5_sens/done_tissue
