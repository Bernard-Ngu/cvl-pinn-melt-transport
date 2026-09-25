#!/usr/bin/env bash
#
# Reproduce the study, from the gridded input to the figures.
#
#   ./run_all.sh
#
# The stages must run in this order: each writes files a later stage reads,
# so running one out of sequence uses stale inputs rather than failing, which
# is why this script exists rather than a note in the README.
#
# The three training runs dominate the wall clock, of order ten minutes each.
# Everything after them takes seconds.
#
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
mkdir -p logs output figures runs

if [ ! -f data/grids.npz ]; then
    echo "error: data/grids.npz is missing - it holds the tomography and the" >&2
    echo "       lithosphere-asthenosphere boundary the model is conditioned on." >&2
    exit 1
fi

run () { echo; echo "=== $1 ==="; shift; python3 "$@"; }

# --- background state --------------------------------------------------
# Converts the shear-wave anomaly into temperature and melt fraction and
# writes data/background.npz, which every later stage reads.  This is why
# background.npz is not in the repository: it is derived, in about a minute,
# from grids.npz.
run "fields.py  (background temperature and melt)" code/fields.py

# --- the three lithospheric treatments ---------------------------------
# A is the model reported; L and S bound the effect of the lid on the flow.
# Each writes runs/params_<tag>.npz and runs/solution_<tag>.npz.  The trained
# weights are in the repository; the dense solutions are not, because they are
# 47 MB each and this step regenerates them.
for spec in "A none" "L locked" "S slip"; do
    set -- $spec
    echo; echo "=== train.py  (tag $1, lid $2) ==="
    python3 code/train.py --tag "$1" --lid "$2" --iters 5000 2>&1 \
        | tee "logs/train_$1.txt"
done

# --- verification and numbers ------------------------------------------
run "verify.py  (residuals and energy identity)" code/verify.py
run "stats.py   (every number the paper quotes)" code/stats.py

# --- figures ------------------------------------------------------------
run "make_figures.py  (Figs. 2-13 at the original type size)" code/make_figures.py

echo
echo "done.  For Fig. 1 run          code/run_map.sh"
echo "For publication type size run  code/rebuild_figures.sh"
echo "For the sensitivity analysis   code/run_experiments.sh"
