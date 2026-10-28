#!/usr/bin/env bash
# Probe sensitivity analyses on the cached features (no retraining):
#   C0.1, C10     every classification task with the probe's C changed from 1
#   site_patient  site task with folds grouped by patient instead of slide
# Outputs mirror outputs/v5 under outputs/v5_sens/<variant>/ so aggregate.py can pool seeds.
set -u
cd "$(dirname "$0")/../.."
PY=/home/holiday/venvs/wsi_hl/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
jobs() {
  for d in outputs/v5/*/*/seed*; do
    t=$(basename $(dirname $(dirname $d))); rel=${d#outputs/v5/}; key="student__v5_$(echo $rel | sed 's#/#_#g')"
    echo "site_patient center $t $key $rel --center_group patient"
    for c in 0.1 10; do for task in c16 c17_loco msi panda crc7k center panda_biopsy; do
      echo "C$c $task $t $key $rel --C $c"
    done; done
  done
}
jobs | xargs -P 40 -L 1 bash -c 'out=outputs/v5_sens/$0/$4/probe_$1.json; [ -f $out ] || '"$PY"' scripts/eval_v5/run_probes.py --task $1 --teacher $2 --student_key $3 --out $out "${@:5}" > /dev/null 2>>outputs/v5_sens/errors.log || echo "FAIL $*" >> outputs/v5_sens/errors.log'
for v in site_patient C0.1 C10; do
  $PY scripts/eval_v5/aggregate.py --root outputs/v5_sens/$v --out outputs/v5_sens/$v/summary
done
date -Is > outputs/v5_sens/done
