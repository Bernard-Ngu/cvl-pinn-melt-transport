#!/usr/bin/env python3
"""
Reproduce map1.png from the vector data in this folder, with a distinct
colour for every tectonic feature.

Every colour lives in the COLORS dictionary below.  Change a value there and
the legend, the outlines and the fills all follow -- nothing else needs
editing.

The script prefers PyGMT, which gives the shaded relief, the bathymetry and
the globe inset of the original figure.  If PyGMT is not installed it falls
back to matplotlib, which needs nothing beyond numpy and draws the same
features in the same colours but without relief or bathymetry.

Background topography
---------------------
Taken from cvl.sh, which used GMT's remote dataset:

    GRD=@earth_relief_01m
    gmt grdcut $GRD -G$CUT -R$REGION

Nothing needs downloading by hand.  GMT fetches the tiles for the requested
region the first time the script runs and caches them under ~/.gmt, so the
first run needs a network connection and later runs do not.  The colour scale
is the same two-part CPT cvl.sh built: 'oslo' from -6000 m to 0 for the
bathymetry, 'ragray' from 0 to 5000 m for the land (see build_topo_cpt).

    python3 make_map1.py                 # -> map1_recoloured.png / .pdf
    python3 make_map1.py --backend mpl   # force the matplotlib fallback

Data files used (all already in this folder):

    African_Cratons.gmt  10 Archaean craton polygons
    SC.gmt               Sahara Meta-Craton polygon
    Benue.gmt            Benue (Mesozoic) Trough polygon
    Chad_bassin.gmt      Chad Basin polygon
    CVL.gmt              26 Cameroon Volcanic Line outcrop polygons
    Atlantic_FZ.gmt      7 Atlantic fracture zones
    rt.gmt               West and Central African Rift System
    sz.gmt               Central African Shear Zone system
    bt.gmt, bt1.gmt      Benue Trough bounding structures
    j.gmt, jos.gmt       Jos Plateau younger granites
    boundary.gmt         continent-ocean boundary

SHmax orientations and the stress-regime symbols of the original are NOT in
this folder.  Drop them in as shmax.csv and stress.csv (formats documented at
STRESS_FILES below) and they will be drawn automatically.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The vector outlines and the topography live in ../map; every data path
# below is built from HERE, so pointing HERE at them is the whole change
# needed to keep this script with the rest of the code.
if not os.path.exists(os.path.join(HERE, "CVL.gmt")):
    _data = os.path.abspath(os.path.join(HERE, "..", "map"))
    if os.path.exists(os.path.join(_data, "CVL.gmt")):
        HERE = _data

# --------------------------------------------------------------------------
# COLOURS -- the only place you need to edit
# --------------------------------------------------------------------------
COLORS = {
    # areas
    "craton":        "#BFB9B2",   # Archaean cratons
    "craton_edge":   "#6E6862",
    "sahara":        "#D8C7A9",   # Sahara Meta-Craton
    "sahara_edge":   "#8A7757",
    "benue":         "#CC79A7",   # Benue / Mesozoic Trough
    "benue_edge":    "#8E4B76",
    "chad":          "#F0E442",   # Chad Basin
    "chad_edge":     "#9C9130",
    "cvl":           "#D55E00",   # CVL volcanic outcrops
    "cvl_edge":      "#7F3800",
    "jos":           "#56B4E9",   # Jos Plateau younger granites
    "jos_edge":      "#1F6E9C",

    # lines
    "rift":          "#0072B2",   # West and Central African Rift System
    "shear":         "#000000",   # Central African Shear Zone
    "fz":            "#009E73",   # Atlantic fracture zones
    "boundary":      "#444444",   # continent-ocean boundary

    # stress symbols (only used if stress.csv is present)
    "compression":   "#B15928",
    "strikeslip":    "#E31A1C",
    "extension":     "#1F78B4",
    "transpression": "#FFD92F",
    "shmax":         "#000000",

    # base map
    "land":          "#F2F0EC",
    "water":         "#CFE3F2",
    "coast":         "#555555",
    "neutral":       "#FAFAF8",   # matplotlib backend background (no coastline)
    "leader":        "#333333",   # label leader lines
}

# --------------------------------------------------------------------------
# TEXT SIZES -- the second place to edit
# --------------------------------------------------------------------------
# `place_scale` multiplies the per-label sizes given in LABELS below, so the
# relative hierarchy of the place names is kept while all of them grow or
# shrink together.  The rest are absolute point sizes.
# Sizes may be overridden from the environment so that one shell script can
# set the type for this map and for the figures in ../code together.  With
# nothing set, the values are exactly what they have always been.
FONTS = {
    "place_scale": float(os.environ.get("CVL_PLACE_SCALE", 1.30)),
    "tick":        float(os.environ.get("CVL_TICK_SIZE", 16)),
    "axis":        float(os.environ.get("CVL_LABEL_SIZE", 18)),
    "legend":      float(os.environ.get("CVL_LEGEND_SIZE", 14)),
    "colorbar":    float(os.environ.get("CVL_LABEL_SIZE", 16)),
}

# Canvas size in inches (matplotlib backend).  Enlarge this together with the
# fonts: bigger type on the same canvas just makes the place names collide.
FIGSIZE = tuple(float(os.environ.get("CVL_MAP_SCALE", 1.0)) * v
                for v in (15.0, 11.0))

# --------------------------------------------------------------------------
# LINE WIDTHS -- the third place to edit
# --------------------------------------------------------------------------
# Multipliers on the per-feature widths given in LINES and POLYGONS below, so
# the relative hierarchy is kept (the rift system stays heavier than the
# continent-ocean boundary) while everything thickens together.
WIDTHS = {
    "line_scale": 2.0,     # tectonic lines: rift, shear zones, fracture zones
    "edge_scale": 1.8,     # outlines of the filled areas
    "leader":     1.0,     # label leader lines
}

# --------------------------------------------------------------------------
# Feature table: file, geometry, legend label, colour keys, line width
# --------------------------------------------------------------------------
POLYGONS = [
    # file,             label,                        fill,      edge,           lw,  z
    ("African_Cratons.gmt", "Archaean cratons",       "craton",  "craton_edge",  0.6, 2),
    ("SC.gmt",              "Sahara Meta-Craton",     "sahara",  "sahara_edge",  0.6, 2),
    ("Chad_bassin.gmt",     "Chad Basin",             "chad",    "chad_edge",    0.6, 3),
    ("Benue.gmt",           "Benue (Mesozoic) Trough","benue",   "benue_edge",   0.6, 4),
    ("jos.gmt",             "Jos Plateau granites",   "jos",     "jos_edge",     0.5, 5),
    ("j.gmt",               None,                     "jos",     "jos_edge",     0.5, 5),
    # CVL.gmt is drawn by draw_cvl_by_age(), one colour per age class.
]

LINES = [
    # file,           label,                                  colour,        lw,  z,  style
    ("rt.gmt",  "West and Central African Rift System", "rift",        1.6, 7, "-"),
    ("sz.gmt",  "Central African Shear Zone",           "shear",       1.4, 8, "-"),
    ("Atlantic_FZ.gmt", "Atlantic fracture zones",      "fz",          1.6, 7, "-"),
    ("boundary.gmt", "Continent-ocean boundary",        "boundary",    1.0, 6, "--"),

    # bt.gmt and bt1.gmt were drawn here in purple as "Benue Trough bounding
    # structures".  Removed: the trough itself is already shown as a filled
    # polygon from Benue.gmt, and the two line files carry no attribute
    # saying which structure they represent, so the extra outline added no
    # information.  Re-add a line here if their meaning is established.
]

# Optional extra data, drawn only if the file exists.
#   shmax.csv   lon,lat,azimuth_deg[,source]
#   stress.csv  lon,lat,regime      regime in {compression,strikeslip,
#                                              extension,transpression}
STRESS_FILES = {"shmax": "shmax.csv", "stress": "stress.csv"}

REGION = [-2.5, 21.0, -6.0, 12.0]        # lon_min, lon_max, lat_min, lat_max

# Place names.  (lon, lat, text, fontsize, rotation, anchor)
# `anchor` is the feature the label belongs to.  When it is given, the label
# is moved clear of the crowded volcanic chain and joined to its feature by a
# thin leader line, which is the only way the southern centres fit legibly.
# Two sizes carry the hierarchy of the map and are set here rather than
# repeated down the table: the cratons name the basement provinces the chain
# sits on, and the volcanic centres are what the figure is about, so they are
# read at the size the cratons used to be.  Everything else -- the Benue
# Trough, the Jos Plateau and the fracture zones -- is context and keeps its
# original size.
CRATON_FS = float(os.environ.get("CVL_CRATON_FS", 15))
CENTRE_FS = float(os.environ.get("CVL_CENTRE_FS", 12))
# The Benue Trough, the Jos Plateau granites and the Atlantic fracture zones:
# context rather than subject, but they still have to be readable at print
# size, so they sit just below the volcanic centres.
CONTEXT_FS = float(os.environ.get("CVL_CONTEXT_FS", 11))

LABELS = [
    (-1.0,  7.6, "West Africa\nCraton", CRATON_FS,   0, None),
    (16.5,  1.0, "Congo Craton", CRATON_FS + 1,   0, None),
    (17.0, 10.6, "Sahara\nMeta-Craton", CRATON_FS,   0, None),
    ( 9.3,  7.9, "Benue Trough",    CONTEXT_FS,  46, None),   # along the SW-NE limb
    (14.6,  5.9, "Adamawa Plateau", CENTRE_FS, -25, None),
    ( 7.7, 10.6, "Jos Plateau",            CONTEXT_FS,   0, (8.90,  9.90)),
    (13.0, 11.2, "Biu Plateau", CENTRE_FS,   0, (12.10, 10.55)),
    ( 7.1,  3.8, "Mt. Cameroon", CENTRE_FS,   0, (9.17,  4.20)),
    ( 6.5,  4.7, "Mt. Manengouba", CENTRE_FS,   0, (9.83,  5.03)),
    ( 6.2,  5.6, "Mt. Bamboutos", CENTRE_FS,   0, (10.05, 5.63)),
    ( 5.9,  6.5, "Oku Volcanic Group", CENTRE_FS,   0, (10.50, 6.20)),
    ( 8.0,  2.7, "Bioko", CENTRE_FS,   0, (8.72,  3.50)),
    ( 6.2,  1.7, "Príncipe", CENTRE_FS,   0, (7.40,  1.60)),
    ( 5.2,  0.3, "São Tomé", CENTRE_FS,   0, (6.60,  0.25)),
    ( 4.2, -1.4, "Annobón", CENTRE_FS,   0, (5.63, -1.43)),
    (-0.7,  4.3, "Romanche FZ",            CONTEXT_FS,  22, None),
    (-0.7,  2.5, "Chain FZ",               CONTEXT_FS,  22, None),
    (-0.7,  1.0, "Charcot FZ",             CONTEXT_FS,  20, None),
    ( 1.2, -4.4, "Ascension FZ",           CONTEXT_FS,  24, None),
]


# --------------------------------------------------------------------------
# VOLCANIC AGES -- the third place to edit
# --------------------------------------------------------------------------
# Age of volcanic activity at each centre, in Ma, as (onset, youngest).
#
# DRAFTED from the compilations already cited in the manuscript: Fitton and
# Dunlop (1985), Fitton (1987), Meyers et al. (1998), Nkouathio et al. (2008)
# and Adams (2022).  CHECK EVERY VALUE against those papers before the figure
# is submitted -- the map prints them.  Changing a number here updates the
# outcrop colour, the legend and the inset together; nothing else to edit.
AGES = {
    "Mandara":     (35.0, 20.0),
    "Principe":    (31.0,  3.0),
    "Oku":         (25.0,  0.0),
    "Bambouto":    (21.0,  4.5),
    "Sao Tome":    (13.0,  0.1),
    "Adamawa":     (11.0,  0.0),
    "Biu":         ( 5.0,  0.0),
    "Annobon":     ( 4.8,  0.0),
    "Manengouba":  ( 1.5,  0.0),
    "Bioko":       ( 1.0,  0.0),
    "Mt Cameroon": ( 1.0,  0.0),
}

# Each CVL.gmt polygon is attached to the nearest of these, and takes its age.
CENTRES = [
    ("Annobon",     5.63, -1.44), ("Sao Tome",    6.60,  0.25),
    ("Principe",    7.40,  1.60), ("Bioko",       8.72,  3.50),
    ("Mt Cameroon", 9.17,  4.20), ("Manengouba",  9.83,  5.03),
    ("Bambouto",   10.05,  5.63), ("Oku",        10.50,  6.20),
    ("Adamawa",    13.67,  7.25), ("Biu",        12.10, 10.55),
    ("Mandara",    13.70, 10.80),
]

# Colour by age of onset.  One hue, light to dark with increasing age, so the
# ramp reads as a sequence and keeps the orange the outcrops already had.
AGE_BINS = [
    ( 1.0, "#FEE391", "Onset < 1 Ma"),
    (10.0, "#FE9929", "Onset 1-10 Ma"),
    (30.0, "#CC4C02", "Onset 10-30 Ma"),
    (1e9,  "#662506", "Onset > 30 Ma"),
]


# --------------------------------------------------------------------------
# GMT multi-segment reader
# --------------------------------------------------------------------------
def read_gmt(path):
    """Return a list of segments; each segment is an (N, 2) lon/lat array.

    Handles the two flavours in this folder: '>' separated multi-segment
    files, and files whose first line is a bare label (SC.gmt).  '#' header
    lines written by ogr2ogr are skipped.
    """
    segs, cur = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith(">"):
                if cur:
                    segs.append(np.array(cur))
                cur = []
                continue
            parts = s.split()
            try:
                cur.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                # a bare text label such as the first line of SC.gmt
                if cur:
                    segs.append(np.array(cur))
                cur = []
    if cur:
        segs.append(np.array(cur))
    return [s for s in segs if len(s) >= 2]


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print("  ! %s not found, skipping" % name)
        return []
    return read_gmt(path)


# --------------------------------------------------------------------------
# Background topography for the matplotlib backend
# --------------------------------------------------------------------------
# matplotlib cannot read GMT's remote grids, so run_map.sh extracts them once
# with the GMT command line:
#
#   gmt grdcut @earth_relief_01m -Gtopo.nc -R<region>
#   gmt grdinfo topo.nc -C            > topo.hdr
#   gmt grd2xyz topo.nc -ZTLf         > topo.bin
#
# -ZTLf writes plain 4-byte floats, row major, starting at the top-left
# corner, which numpy reads directly.  The .hdr line gives the bounds and the
# grid size needed to reshape it.
def load_topo(binpath):
    hdr = os.path.splitext(binpath)[0] + ".hdr"
    if not (os.path.exists(binpath) and os.path.exists(hdr)):
        return None
    f = open(hdr).read().split()
    # grdinfo -C: name x_min x_max y_min y_max z_min z_max dx dy n_col n_row
    x0, x1, y0, y1 = (float(v) for v in f[1:5])
    ncol, nrow = int(f[9]), int(f[10])
    z = np.fromfile(binpath, dtype="<f4")
    if z.size != ncol * nrow:
        print("  ! %s has %d values, expected %d x %d - ignoring topography"
              % (os.path.basename(binpath), z.size, ncol, nrow))
        return None
    z = z.reshape(nrow, ncol).astype(float)
    z[~np.isfinite(z)] = 0.0
    return z, (x0, x1, y0, y1)


def topo_cmap():
    """Approximate the two-part CPT of cvl.sh: oslo below sea level, ragray
    above.  Used only by the matplotlib backend; PyGMT uses the real CPTs."""
    from matplotlib.colors import LinearSegmentedColormap
    oslo = ["#01111e", "#0a2f4f", "#13537d", "#3079a4", "#72a6c6",
            "#b3d0e2", "#dcebf4"]
    ragray = ["#8f9488", "#a5a99b", "#bbbfaf", "#d0d3c4", "#e1e3d8",
              "#eff0e9", "#fbfbf7"]
    return LinearSegmentedColormap.from_list("oslo_ragray", oslo + ragray)


def hillshade(z, dx, dy, azimuth=315.0, altitude=45.0):
    gy, gx = np.gradient(z, dy, dx)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    s = (np.sin(alt) * np.sin(slope)
         + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(s, 0.0, 1.0)


def load_csv(key):
    path = os.path.join(HERE, STRESS_FILES[key])
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.lower().startswith("lon"):
                continue
            rows.append(s.split(","))
    return rows


# --------------------------------------------------------------------------
# matplotlib backend -- no geospatial dependencies
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Volcanic ages and model context
# --------------------------------------------------------------------------
def age_of(seg):
    """The age record of the centre nearest this outcrop polygon."""
    c = seg[:, :2].mean(axis=0)
    name = min(CENTRES,
               key=lambda t: (c[0] - t[1]) ** 2 + (c[1] - t[2]) ** 2)[0]
    return name, AGES.get(name)


def age_colour(onset):
    for hi, col, _ in AGE_BINS:
        if onset <= hi:
            return col
    return AGE_BINS[-1][1]


def draw_cvl_by_age(ax, alpha, handles):
    """CVL outcrops, filled by the age at which each centre started."""
    from matplotlib.patches import Patch
    segs = load("CVL.gmt")
    if not segs:
        return
    elw = 0.4 * WIDTHS["edge_scale"]
    used, unknown = set(), 0
    for seg in segs:
        name, rec = age_of(seg)
        if rec is None:
            col, unknown = COLORS["cvl"], unknown + 1
        else:
            col = age_colour(rec[0])
            used.add(col)
        ax.fill(seg[:, 0], seg[:, 1], facecolor=col,
                edgecolor=COLORS["cvl_edge"], linewidth=elw, zorder=6,
                alpha=alpha)
    handles.append(Patch(facecolor="none", edgecolor="none",
                         label="CVL outcrops, by age of onset:"))
    for hi, col, text in AGE_BINS:
        if col in used:
            handles.append(Patch(facecolor=col, edgecolor=COLORS["cvl_edge"],
                                 linewidth=elw, label="   " + text))
    if unknown:
        print("  ! %d CVL polygons had no age and kept the default colour"
              % unknown)


def draw_matplotlib(out, topo=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor(COLORS["neutral"])

    handles = []

    # ---- shaded relief and bathymetry ------------------------------------
    topo_drawn = False
    if topo:
        loaded = load_topo(topo)
        if loaded:
            z, (x0, x1, y0, y1) = loaded
            norm = TwoSlopeNorm(vmin=-6000, vcenter=0, vmax=5000)
            cmap = topo_cmap()
            rgb = cmap(norm(z))
            dx = (x1 - x0) / z.shape[1] * 111.32
            dy = (y1 - y0) / z.shape[0] * 110.57
            sh = hillshade(z, dx, dy)[..., None]
            rgb[..., :3] = np.clip(rgb[..., :3] * (0.78 + 0.40 * sh), 0, 1)
            ax.imshow(rgb, extent=(x0, x1, y0, y1), origin="upper",
                      interpolation="bilinear", zorder=0)
            topo_drawn = True
            print("  topography: %d x %d grid, %.0f to %.0f m"
                  % (z.shape[1], z.shape[0], z.min(), z.max()))
    if not topo_drawn:
        print("  ! no topography grid - run through run_map.sh to fetch it")

    # features are drawn semi-transparent over the relief so it shows through
    alpha = 0.72 if topo_drawn else 1.0

    for fname, label, fill, edge, lw, z in POLYGONS:
        segs = load(fname)
        if not segs:
            continue
        elw = lw * WIDTHS["edge_scale"]
        for s in segs:
            ax.fill(s[:, 0], s[:, 1], facecolor=COLORS[fill],
                    edgecolor=COLORS[edge], linewidth=elw, zorder=z,
                    alpha=alpha)
        if label:
            handles.append(Patch(facecolor=COLORS[fill],
                                 edgecolor=COLORS[edge], linewidth=elw,
                                 label=label))

    for fname, label, col, lw, z, style in LINES:
        segs = load(fname)
        if not segs:
            continue
        w = lw * WIDTHS["line_scale"]
        for s in segs:
            ax.plot(s[:, 0], s[:, 1], color=COLORS[col], linewidth=w,
                    linestyle=style, zorder=z, solid_capstyle="round",
                    solid_joinstyle="round")
        if label:
            handles.append(Line2D([], [], color=COLORS[col], linewidth=w,
                                  linestyle=style, label=label))

    draw_cvl_by_age(ax, alpha, handles)

    # optional stress data
    stress = load_csv("stress")
    if stress:
        seen = set()
        for r in stress:
            lon, lat, regime = float(r[0]), float(r[1]), r[2].strip().lower()
            c = COLORS.get(regime, "#888888")
            ax.plot(lon, lat, "o", ms=8, mfc=c, mec="k", mew=0.7, zorder=10)
            if regime not in seen:
                seen.add(regime)
                handles.append(Line2D([], [], marker="o", ls="", ms=8,
                                      mfc=c, mec="k", label=regime.title()))
    shmax = load_csv("shmax")
    if shmax:
        for r in shmax:
            lon, lat, az = float(r[0]), float(r[1]), float(r[2])
            dx = 0.45 * np.sin(np.radians(az))
            dy = 0.45 * np.cos(np.radians(az))
            ax.plot([lon - dx, lon + dx], [lat - dy, lat + dy],
                    color=COLORS["shmax"], lw=1.6, zorder=11)
        handles.append(Line2D([], [], color=COLORS["shmax"], lw=1.6,
                              label="SHmax"))

    scale = FONTS["place_scale"]
    halo = [pe.withStroke(linewidth=1.9 + 0.9 * scale, foreground="white")]
    for lon, lat, text, fs, rot, anchor in LABELS:
        if anchor:
            ax.annotate("", xy=anchor, xytext=(lon, lat), zorder=11,
                        arrowprops=dict(arrowstyle="-",
                                        lw=WIDTHS["leader"],
                                        color=COLORS["leader"],
                                        shrinkA=3, shrinkB=1))
        ax.text(lon, lat, text, fontsize=fs * scale, rotation=rot, zorder=12,
                ha="center", va="center", fontweight="bold",
                rotation_mode="anchor", path_effects=halo)

    ax.set_xlim(REGION[0], REGION[1])
    ax.set_ylim(REGION[2], REGION[3])
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(REGION[2:]))))
    ax.set_xlabel("Longitude (°E)", fontsize=FONTS["axis"])
    ax.set_ylabel("Latitude (°N)", fontsize=FONTS["axis"])
    ax.tick_params(axis="both", which="major", labelsize=FONTS["tick"],
                   length=6, width=1.1)
    ax.grid(alpha=0.25, ls=":", zorder=1)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=FONTS["legend"], frameon=True)

    if topo_drawn:
        import matplotlib.cm as cm
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          fraction=0.035, pad=0.08, extend="both")
        cb.set_label("Topography (km)", fontsize=FONTS["colorbar"])
        ticks = [-6000, -4000, -2000, 0, 2000, 4000]
        cb.set_ticks(ticks)
        cb.set_ticklabels(["%g" % (t / 1000.0) for t in ticks])
        cb.ax.tick_params(labelsize=FONTS["tick"])

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out, ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s.png and %s.pdf  (matplotlib backend)" % (out, out))


# --------------------------------------------------------------------------
# PyGMT backend -- relief, bathymetry and globe inset, as in the original
# --------------------------------------------------------------------------
def build_topo_cpt(path):
    """Two-part topography CPT, exactly as cvl.sh builds it.

        gmt makecpt -Coslo   -T-6000/0 -N -H >  color.cpt
        gmt makecpt -Cragray -T0/5000     -H >> color.cpt

    'oslo' colours the bathymetry, 'ragray' the land, and -N on the first
    call suppresses the background/foreground/NaN lines so the second CPT can
    simply be appended.
    """
    import pygmt

    sea = path + ".sea"
    land = path + ".land"
    pygmt.makecpt(cmap="oslo", series=[-6000, 0], no_bg=True, output=sea)
    pygmt.makecpt(cmap="ragray", series=[0, 5000], output=land)
    with open(path, "w") as out_fh:
        for part in (sea, land):
            with open(part) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    out_fh.write(line)
    for part in (sea, land):
        os.remove(part)
    return path


def draw_pygmt(out):
    import pygmt

    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd:mm:ssF",
                 MAP_FRAME_PEN="dimgray",
                 MAP_FRAME_WIDTH="0.1c",
                 MAP_ANNOT_OFFSET="0.1c",
                 MAP_TICK_PEN_PRIMARY="thinner,dimgray",
                 MAP_GRID_PEN_PRIMARY="thinner,dimgray",
                 FONT_ANNOT_PRIMARY="%gp,Helvetica,dimgray" % FONTS["tick"],
                 FONT_LABEL="%gp,Helvetica,dimgray" % FONTS["axis"])

    proj = "M15c"

    # Background topography and bathymetry.
    #
    # cvl.sh uses GMT's remote dataset  @earth_relief_01m  and cuts it to the
    # region with grdcut.  load_earth_relief does both in one call: GMT
    # downloads the tiles on first use and caches them under ~/.gmt, so this
    # needs a network connection once and is offline thereafter.  Drop to
    # resolution="30s" for a sharper map, or "02m"/"05m" for a faster one.
    grid = pygmt.datasets.load_earth_relief(resolution="01m", region=REGION)

    cpt = build_topo_cpt(os.path.join(HERE, "temp_topo.cpt"))
    fig.basemap(region=REGION, projection=proj, frame="WsNe")
    fig.grdimage(grid=grid, cmap=cpt, shading="+d")

    fig.coast(shorelines="0.5p,%s" % COLORS["coast"],
              borders="1/thin,black", resolution="f")

    for fname, label, fill, edge, lw, _z in POLYGONS:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            continue
        fig.plot(data=path, fill=COLORS[fill],
                 pen="%.2gp,%s" % (lw * WIDTHS["edge_scale"], COLORS[edge]),
                 label=label if label else None)

    for fname, label, col, lw, _z, style in LINES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            continue
        pen = "%.2gp,%s" % (lw * WIDTHS["line_scale"], COLORS[col])
        if style == "--":
            pen += ",-"
        fig.plot(data=path, pen=pen, label=label if label else None)

    stress = load_csv("stress")
    if stress:
        for regime in ("compression", "strikeslip", "extension",
                       "transpression"):
            pts = [(float(r[0]), float(r[1])) for r in stress
                   if r[2].strip().lower() == regime]
            if not pts:
                continue
            fig.plot(x=[p[0] for p in pts], y=[p[1] for p in pts],
                     style="c0.22c", fill=COLORS[regime], pen="0.4p,black",
                     label=regime.title())
    shmax = load_csv("shmax")
    if shmax:
        fig.plot(x=[float(r[0]) for r in shmax],
                 y=[float(r[1]) for r in shmax],
                 direction=[[float(r[2]) for r in shmax],
                            [0.6] * len(shmax)],
                 style="v0.0c", pen="1.4p,%s" % COLORS["shmax"],
                 label="SHmax")

    for lon, lat, text, fs, rot, anchor in LABELS:
        if anchor:
            fig.plot(x=[lon, anchor[0]], y=[lat, anchor[1]],
                     pen="%.2gp,%s" % (WIDTHS["leader"], COLORS["leader"]))
        fig.text(x=lon, y=lat, text=text.replace("\n", " "),
                 font="%gp,Helvetica-Bold,black" % (fs * FONTS["place_scale"]),
                 angle=rot, fill="white@30", clearance="0.05c/0.05c")

    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p0.8p,black")

    # north arrow and scale bar, as in cvl.sh
    fig.basemap(rose="jTR+w0.45i+l+f3+o0.3c/1.6c", box="+gwhite@50")
    fig.basemap(map_scale="jRB+c%g+w200k+f+o0.6c/0.6c+u" % np.mean(REGION[2:]),
                box="+gwhite@50")

    # -W0.001 rescales the CPT from metres to kilometres for the annotation
    fig.colorbar(cmap=cpt, position="JBC+w7.5c/0.418c+o0c/0.9c+e+h",
                 frame=['xa1+l"Topography"', 'y+l"(km)"'],
                 shading=True, scale=0.001)

    fig.basemap(frame=["xa2f1", "ya4f2"])

    # globe inset, orthographic on the map centre, cratons outlined in red
    with fig.inset(position="jTL+w1.95i+o-0.75i/-0.75i"):
        lon0 = 0.5 * (REGION[0] + REGION[1])
        lat0 = 0.5 * (REGION[2] + REGION[3]) + 12
        fig.coast(region="g", projection="G%g/%g/?" % (lon0, lat0),
                  land="gray", water="steelblue1", area_thresh=5000,
                  shorelines="faint", borders="1", frame="g")
        fig.plot(data=os.path.join(HERE, "African_Cratons.gmt"),
                 projection="G%g/%g/?" % (lon0, lat0), region="g",
                 pen="1.25p,red", straight_line=True)

    fig.savefig("%s.png" % out, dpi=300)
    fig.savefig("%s.pdf" % out)
    for tmp in (cpt,):
        if os.path.exists(tmp):
            os.remove(tmp)
    print("wrote %s.png and %s.pdf  (PyGMT backend)" % (out, out))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["auto", "pygmt", "mpl"],
                    default="auto")
    ap.add_argument("-o", "--out",
                    default=os.path.join(HERE, "map1_recoloured"))
    ap.add_argument("--topo", default=os.path.join(HERE, "topo.bin"),
                    help="binary relief grid written by run_map.sh "
                         "(matplotlib backend only)")
    ap.add_argument("--print-region", action="store_true",
                    help="print the region as lon0/lon1/lat0/lat1 and exit, "
                         "so run_map.sh and this script cannot disagree")
    a = ap.parse_args()

    if a.print_region:
        print("%g/%g/%g/%g" % tuple(REGION))
        return

    backend = a.backend
    if backend == "auto":
        try:
            import pygmt          # noqa: F401
            backend = "pygmt"
        except Exception:
            backend = "mpl"
            print("note: PyGMT not available, using the matplotlib fallback.")
            print("      Install it with:  conda install -c conda-forge pygmt")

    if backend == "pygmt":
        draw_pygmt(a.out)
    else:
        draw_matplotlib(a.out, topo=a.topo)


if __name__ == "__main__":
    main()
