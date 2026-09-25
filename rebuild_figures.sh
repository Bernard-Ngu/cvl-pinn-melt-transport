#!/usr/bin/env bash
#
# Redraw every manuscript figure at publication type size.
#
#   ./rebuild_figures.sh            axis labels 24 pt, axis values 18 pt
#   ./rebuild_figures.sh --map-only just Fig. 1
#   ./rebuild_figures.sh --no-map   everything except Fig. 1
#   ./rebuild_figures.sh --only 2   just manuscript Fig. 2 (2 to 13)
#
# Sizes are set here and nowhere else.  Both the map in ../map and the
# figures in ../code read them from the environment, so changing a number
# below changes every figure at once.
#
#   CVL_TICK_ROOM    how much clear space each axis value demands; raise it
#                    to thin the numbers further, lower it to keep more
#   CVL_LABEL_SIZE   axis labels, colour-bar labels, panel titles
#   CVL_TICK_SIZE    axis values
#   CVL_LEGEND_SIZE  legends
#   CVL_FIG_SCALE    canvas multiplier for ../code figures
#   CVL_MAP_SCALE    canvas multiplier for the map
#   CVL_FIG2_SIZE    one size for everything in Fig. 2, the three map panels
#   CVL_FIG2_SCALE   canvas multiplier for Fig. 2, so its titles clear
#   CVL_FIG11_SIZE   one size for everything in Fig. 11, the 3 x 3 lid grid
#   CVL_CRATON_FS    craton names on Fig. 1
#   CVL_CENTRE_FS    volcanic centre names on Fig. 1
#   CVL_CONTEXT_FS   Benue Trough, Jos Plateau and fracture zones on Fig. 1
#
# Fig. 11 is the exception to the 24/18 split: a three-by-three grid of small
# panels reads better with a single size throughout, so it is drawn at 26.
#
# Output overwrites ../figures (and ../map/map1_recoloured.*), which is where
# build_revision.sh picks the images up, so the order is: redraw, then build.
#
set -euo pipefail
cd "$(dirname "$0")"

export CVL_LABEL_SIZE=${CVL_LABEL_SIZE:-32}
export CVL_TICK_SIZE=${CVL_TICK_SIZE:-28}
export CVL_LEGEND_SIZE=${CVL_LEGEND_SIZE:-26}
export CVL_FIG_SCALE=${CVL_FIG_SCALE:-3.4}
export CVL_MAP_SCALE=${CVL_MAP_SCALE:-1.15}
export CVL_FIG2_SIZE=${CVL_FIG2_SIZE:-28}
export CVL_FIG2_SCALE=${CVL_FIG2_SCALE:-3.6}
export CVL_FIG11_SIZE=${CVL_FIG11_SIZE:-30}
export CVL_FIG2_LABEL=${CVL_FIG2_LABEL:-32}
export CVL_TICK_ROOM=${CVL_TICK_ROOM:-1.8}
export CVL_CRATON_FS=${CVL_CRATON_FS:-15}
export CVL_CENTRE_FS=${CVL_CENTRE_FS:-12}
export CVL_CONTEXT_FS=${CVL_CONTEXT_FS:-11}
export PYTHONUNBUFFERED=1

PY=${PYTHON:-python3}
DO_MAP=1; DO_FIGS=1; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --map-only) DO_FIGS=0; shift ;;
        --no-map)   DO_MAP=0; shift ;;
        --only)     ONLY="$ONLY $2"; DO_MAP=0; shift 2 ;;
        -h|--help)  sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "type: labels ${CVL_LABEL_SIZE} pt, axis values ${CVL_TICK_SIZE} pt"
echo "      Fig. 2 (three map panels): ${CVL_FIG2_SIZE} pt throughout"
echo "      Fig. 11 (3 x 3 lid grid): ${CVL_FIG11_SIZE} pt throughout"
echo "      Fig. 1 place names: cratons ${CVL_CRATON_FS} pt, centres ${CVL_CENTRE_FS} pt, context ${CVL_CONTEXT_FS} pt"
echo

if [ "$DO_MAP" = 1 ]; then
    echo "=== Fig. 1: the tectonic map (../map) ==="
    # run_map.sh chooses PyGMT, GMT or the matplotlib fallback, and copies the
    # result to ../figures/fig1_tectonic_setting.* when it is done.
    ( cd ../map && ./run_map.sh )
    echo
fi

if [ "$DO_FIGS" = 1 ]; then
    if [ -n "$ONLY" ]; then
        echo "=== manuscript figure(s)$ONLY (../code) ==="
    else
        echo "=== Figs. 2-13: the model figures (../code) ==="
    fi
    if [ ! -f ../runs/solution_A.npz ]; then
        echo "error: ../runs/solution_A.npz is missing - train first" >&2
        exit 1
    fi
    # shellcheck disable=SC2086
    "$PY" make_figures_big.py $ONLY
    echo
fi

echo "done.  Next:  ./build_revision.sh"
