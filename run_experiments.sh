#!/usr/bin/env bash
#
# Reviewer experiments for the CVL manuscript.
#
#   ./run_experiments.sh              # 1500 iterations per configuration
#   ITERS=5000 ./run_experiments.sh   # full length, as the reported run
#
# Eleven configurations: a 3x3 architecture sweep, plus two ablations of the
# objective at the reported architecture.  Each one trains from the same seed
# with the same batch size, optimiser and curriculum, and each is evaluated
# afterwards on the same fixed set of collocation points, so the rows of the
# table are comparable.  Results land in results/<tag>.json; logs in logs/.
#
# Rough cost at 1500 iterations, from a 0.07 s/iteration measurement on the
# smallest network: about 45 minutes in total, most of it in 128x7.
# At ITERS=5000 reckon on two and a half to three hours.
#
# Safe to interrupt.  Every configuration checkpoints to results/ck_<tag>.npz
# and re-running the script picks up where it stopped.  (Adam's moment
# estimates restart on resume, so a run stitched together from interruptions
# is not bit-identical to one run straight through; for the numbers that go
# in the paper, let it run to the end.)
#
set -uo pipefail
cd "$(dirname "$0")"

ITERS=${ITERS:-1500}
mkdir -p ../experiments/results ../experiments/logs

while read -r tag mode w d; do
    [ -z "$tag" ] && continue
    case "$tag" in \#*) continue ;; esac
    if [ -f "../experiments/results/$tag.json" ]; then
        echo "=== $tag  already done, skipping ==="
        continue
    fi
    echo
    echo "=== $tag   mode=$mode   ${d} layers x ${w} neurons   iters=$ITERS ==="
    python3 exp_run.py --tag "$tag" --mode "$mode" --width "$w" --depth "$d" \
                       --iters "$ITERS" 2>&1 | tee "../experiments/logs/$tag.txt"
done < configs.txt

echo
echo "=== assembling the tables and figures ==="
python3 make_table.py
python3 make_figures_exp.py
