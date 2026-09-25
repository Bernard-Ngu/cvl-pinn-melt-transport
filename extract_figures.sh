#!/usr/bin/env bash
#
# Collect the manuscript figures, renumbered as the manuscript numbers them,
# into one folder for submission.
#
#   ./extract_figures.sh                   -> ../Figure_01.pdf, _01.png, ...
#   ./extract_figures.sh -o ../submission  -> into that folder instead
#   ./extract_figures.sh --pdf             -> PDF only
#   ./extract_figures.sh --png             -> PNG only
#
# Fig. 1 is the map from ../map; Figs. 2-13 are the model figures in
# ../figures, whose file names do not match the manuscript numbering.  This
# script is the mapping between the two, in one place.
#
set -euo pipefail
cd "$(dirname "$0")"

OUT="..";  WANT_PDF=1; WANT_PNG=1
while [ $# -gt 0 ]; do
    case "$1" in
        -o|--out) OUT="$2"; shift 2 ;;
        --pdf) WANT_PNG=0; shift ;;
        --png) WANT_PDF=0; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
mkdir -p "$OUT"

# manuscript number : stem, without extension
FIGS=(
 "01:../map/map1_recoloured"
 "02:../figures/fig1_setting"
 "03:../figures/fig2_vs_partition"
 "04:../figures/fig3_melt_porosity_slices"
 "05:../figures/fig4_flow_slices"
 "06:../figures/fig5_cross_sections"
 "07:../figures/fig6_oblique_sections"
 "08:../figures/fig7_melt_budget"
 "09:../figures/fig8_profiles"
 "10:../figures/fig9_oblique_melt"
 "11:../figures/fig10_lid_comparison"
 "12:../figures/fig11_lid_profiles"
 "13:../figures/fig12_convergence"
)

n=0; missing=0
for spec in "${FIGS[@]}"; do
    num=${spec%%:*}; stem=${spec#*:}
    # Fig. 1 falls back to the copy run_map.sh leaves in ../figures
    if [ "$num" = "01" ] && [ ! -f "$stem.png" ]; then
        stem="../figures/fig1_tectonic_setting"
    fi
    found=0
    for ext in pdf png; do
        [ "$ext" = pdf ] && [ "$WANT_PDF" = 0 ] && continue
        [ "$ext" = png ] && [ "$WANT_PNG" = 0 ] && continue
        if [ -f "$stem.$ext" ]; then
            cp -f "$stem.$ext" "$OUT/Figure_$num.$ext"
            found=1
        fi
    done
    if [ "$found" = 1 ]; then
        n=$((n+1))
        printf "  Figure_%s  <-  %s\n" "$num" "$(basename "$stem")"
    else
        missing=$((missing+1))
        printf "  Figure_%s  MISSING (%s)\n" "$num" "$stem"
    fi
done

echo
echo "$n figures written to $(cd "$OUT" && pwd)"
[ "$missing" -gt 0 ] && echo "$missing missing - run ./rebuild_figures.sh first"
exit 0
