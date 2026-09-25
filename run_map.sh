#!/usr/bin/env bash
#
# Draw the recoloured version of map1.png.
#
#   ./run_map.sh              or      sh run_map.sh
#
# The map itself is drawn by make_map1.py.  This script only works out how to
# give it a background topography, because that is the one piece that cannot
# come from the .gmt files in this folder.
#
# Three cases, tried in this order:
#
#   1. PyGMT installed  -> make_map1.py draws everything itself, including the
#      shaded relief, the bathymetry colour bar and the globe inset.  This is
#      the closest match to the original figure.
#
#   2. GMT installed but not PyGMT -> this script does what cvl.sh did:
#      downloads @earth_relief_01m for the map region and cuts it out, then
#      converts it to a plain binary grid that the matplotlib backend can
#      read.  You get the shaded relief without needing PyGMT.
#
#   3. Neither -> the map is drawn without topography and the script tells
#      you what to install.
#
# The relief tiles are downloaded by GMT once and cached in ~/.gmt, so only
# the first run needs a network connection.
#
# `sh` is not always bash, and some of the syntax below is not POSIX, so
# re-exec under bash when started by a plainer shell.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../figures

PY=${PYTHON:-python3}
RES=${RES:-01m}          # relief resolution: 30s, 01m, 02m, 05m ...
# The figure is written beside the other figures, and the topography grid
# beside the vector outlines it belongs with.
OUT=${OUT:-../figures/map1_recoloured}
TOPO=${TOPO:-../map/topo.bin}

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: $PY not found.  Set PYTHON=/path/to/python3 and retry." >&2
    exit 1
fi
if [ ! -f make_map1.py ]; then
    echo "error: make_map1.py is not in $(pwd)." >&2
    exit 1
fi

# The region is defined once, in make_map1.py, and read back here so the grid
# and the map can never be cut to different extents.
REGION=$("$PY" make_map1.py --print-region)
echo "region: $REGION"

# The paper uses this map as Fig. 1, so copy it next to the other figures
# whenever it is redrawn.
publish() {
    cp -f "$OUT.png" ../figures/fig1_tectonic_setting.png
    [ -f "$OUT.pdf" ] && cp -f "$OUT.pdf" ../figures/fig1_tectonic_setting.pdf
    echo "published -> figures/fig1_tectonic_setting.png (Fig. 1)"
}

# ---------------------------------------------------------------- 1. PyGMT
if "$PY" -c "import pygmt" >/dev/null 2>&1; then
    echo "PyGMT found - drawing the full figure (relief, bathymetry, inset)."
    "$PY" make_map1.py --backend pygmt -o "$OUT"
    publish
    echo
    echo "done -> $OUT.png and $OUT.pdf"
    exit 0
fi

echo "PyGMT not installed."

# ------------------------------------------------- 2. GMT command line only
if command -v gmt >/dev/null 2>&1; then
    echo "GMT found - extracting the relief grid for the matplotlib backend."
    echo "  (first run downloads @earth_relief_$RES and caches it in ~/.gmt)"

    # Same source cvl.sh used:  GRD=@earth_relief_01m ; gmt grdcut -R
    gmt grdcut "@earth_relief_$RES" -G../map/topo.nc -R"$REGION"

    # -C gives a single line: name x_min x_max y_min y_max z_min z_max \
    #                         dx dy n_columns n_rows ...
    gmt grdinfo ../map/topo.nc -C > ../map/topo.hdr

    # -ZTLf : 4-byte floats, row major, from the top-left corner.
    gmt grd2xyz ../map/topo.nc -ZTLf > "$TOPO"

    echo "  grid: $(awk '{print $10" x "$11" cells, "$6" to "$7" m"}' ../map/topo.hdr)"

    "$PY" make_map1.py --backend mpl --topo "$TOPO" -o "$OUT"

    rm -f ../map/topo.nc gmt.history
    publish
    echo
    echo "done -> $OUT.png and $OUT.pdf"
    echo "note: for the coastline, the proper CPTs and the globe inset,"
    echo "      install PyGMT:  conda install -c conda-forge pygmt"
    exit 0
fi

# ------------------------------------------------------------- 3. neither
# The repository ships ../map/topo.bin, the grid GMT produced for this
# region, so the fallback still gets its relief without GMT installed.
if [ -f "$TOPO" ]; then
    echo "GMT not found, but $TOPO is present - using it for the relief."
    "$PY" make_map1.py --backend mpl --topo "$TOPO" -o "$OUT"
else
    echo "GMT not found and no $TOPO - drawing the map without topography." >&2
    "$PY" make_map1.py --backend mpl --topo /nonexistent -o "$OUT"
fi
publish
echo
echo "done -> $OUT.png and $OUT.pdf"
echo
echo "For the coastline, the proper CPTs and the globe inset, install:"
echo "    conda install -c conda-forge pygmt     # best, full figure"
echo "    conda install -c conda-forge gmt       # relief only"
