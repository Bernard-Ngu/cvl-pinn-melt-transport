"""Larger type for the manuscript figures.

The figure code in ../code was laid out for 8 pt type. This module raises the
type to publication size without editing any of it:

  * rcParams set the axis labels, tick labels, titles and legends;
  * every explicit `fontsize=` inside the figure code is scaled by the same
    factor, so annotations grow with everything else;
  * figure canvases are enlarged by the same factor, because larger type on
    the same canvas only makes labels collide.

Sizes come from the environment so that one shell script drives every figure:

  CVL_LABEL_SIZE   axis labels, colour-bar labels, panel titles   (default 32)
  CVL_TICK_SIZE    axis values                                    (default 28)
  CVL_LEGEND_SIZE  legends                                        (default 26)
  CVL_TICK_ROOM    spacing demanded between axis values           (default 1.8)
  CVL_FIG_SCALE    canvas multiplier                (default: label/8.5)
  CVL_DPI          raster resolution for the PNGs                 (default 200)
  CVL_FIG2_SIZE    one size for everything in Fig. 2                (default 26)
  CVL_FIG2_LABEL   axis labels on Fig. 2                            (default 28)
  CVL_FIG11_SIZE   one size for everything in the 3 x 3 lid figure (default 26)

override() raises every size for a single figure; Fig. 11 uses it, because a
three-by-three grid of small panels needs the same size throughout rather
than a label/value split.

Import it and call apply() AFTER importing the figure module, because that
module sets its own rcParams when it is imported.
"""
import os

BASE_FONT = 8.0          # what ../code/newfigs.py was written against
BASE_LABEL = 8.5

LABEL = float(os.environ.get("CVL_LABEL_SIZE", 32))
TICK = float(os.environ.get("CVL_TICK_SIZE", 28))
LEGEND = float(os.environ.get("CVL_LEGEND_SIZE", 26))
SCALE = TICK / BASE_FONT                      # for explicit fontsize= calls
FIG_SCALE = float(os.environ.get("CVL_FIG_SCALE", LABEL / BASE_LABEL))
# The canvases grow by FIG_SCALE, so the same effective resolution is reached
# at a lower dpi; 200 keeps the files a sensible size for a Word document.
DPI = float(os.environ.get("CVL_DPI", 200))
FIG11 = float(os.environ.get("CVL_FIG11_SIZE", 30))
FIG2 = float(os.environ.get("CVL_FIG2_SIZE", 28))
# Axis labels a little above the rest, so the panels keep a hierarchy.
FIG2_LABEL = float(os.environ.get("CVL_FIG2_LABEL", 32))
# Fig. 2 carries three panels with long titles side by side, so it needs a
# wider canvas than the default multiplier gives, or the titles collide.
FIG2_SCALE = float(os.environ.get("CVL_FIG2_SCALE", 3.6))
# Fig. 7 hard-codes its annotations at 4.8 and 6 pt.  The delaminated body is
# the point of the panel, so it is set outright, above everything else.
FIG7_DELAM = float(os.environ.get("CVL_FIG7_DELAM", 30))
FIG7_ANN = float(os.environ.get("CVL_FIG7_ANN", 1.7))
# Fig. 4 shares one colour bar across each row of panels; see override().
FIG4_SPACE = float(os.environ.get("CVL_FIG4_SPACE", 0.30))
# At this type size the default number of ticks no longer fits, so the values
# are thinned until they have room: a multiple of the widest label.
TICK_ROOM = float(os.environ.get("CVL_TICK_ROOM", 1.8))

_patched = False
# When an override asks for one size throughout, legends take that size
# outright rather than being scaled from whatever the figure code asked for.
# Legends are the "key" to a figure and have to be read at print size.  The
# figure code asks for 5.6 or 6 pt, which stays small however much everything
# else is scaled, so legends take CVL_LEGEND_SIZE outright instead.
FORCE_LEGEND = True
# Extra multiplier for text drawn inside the axes (ax.text, ax.annotate), and
# absolute sizes for named strings.  Some figures hard-code annotations at 5
# or 6 pt, which stay small even after everything else is scaled up.
ANN_SCALE = 1.0
TEXT_SIZES = {}


def apply():
    """Raise the type and enlarge the canvases. Safe to call more than once."""
    global _patched
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": TICK,
        "axes.labelsize": LABEL,
        "axes.titlesize": LABEL,
        "figure.titlesize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        # the furniture has to grow with the type or the figure looks spindly
        "axes.linewidth": 0.6 * SCALE,
        "lines.linewidth": 1.0 * SCALE,
        "patch.linewidth": 0.8 * SCALE,
        "grid.linewidth": 0.5 * SCALE,
        "xtick.major.width": 0.6 * SCALE,
        "ytick.major.width": 0.6 * SCALE,
        "xtick.major.size": 3.0 * SCALE,
        "ytick.major.size": 3.0 * SCALE,
        "xtick.minor.width": 0.4 * SCALE,
        "ytick.minor.width": 0.4 * SCALE,
        "lines.markersize": 4.0 * SCALE,
        # The panels were positioned for 8 pt type; at publication size the
        # labels no longer fit the space reserved for them, so let matplotlib
        # re-lay-out each figure at draw time.
        "figure.autolayout": os.environ.get("CVL_AUTOLAYOUT", "1") != "0",
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if _patched:
        return
    _patched = True

    # --- explicit fontsize= inside the figure code -----------------------
    # Only the call sites the figure code uses are wrapped.  Patching
    # Text.set_fontsize instead would also catch the sizes matplotlib itself
    # passes from rcParams when it builds axis labels, and scale them twice.
    def _scale_kw(kw, key="fontsize"):
        v = kw.get(key)
        if isinstance(v, (int, float)):
            kw[key] = v * SCALE
        fd = kw.get("fontdict")
        if isinstance(fd, dict) and isinstance(fd.get("fontsize"), (int, float)):
            kw["fontdict"] = dict(fd, fontsize=fd["fontsize"] * SCALE)
        return kw

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    # Figure.suptitle is deliberately absent: it creates its text by calling
    # Figure.text, so wrapping both would scale a suptitle twice.
    # titles and figure-level text: the ordinary scale
    for cls, name in ((Axes, "set_title"), (Figure, "text")):
        orig = getattr(cls, name)
        setattr(cls, name,
                (lambda o: lambda self, *a, **k: o(self, *a, **_scale_kw(k)))(orig))

    # text inside the axes: the ordinary scale times ANN_SCALE, unless the
    # string is named in TEXT_SIZES, which sets its size outright
    def _ann_kw(kw, s=None):
        if isinstance(s, str) and s in TEXT_SIZES:
            kw["fontsize"] = TEXT_SIZES[s]
            return kw
        v = kw.get("fontsize")
        if isinstance(v, (int, float)):
            kw["fontsize"] = v * SCALE * ANN_SCALE
        return kw

    _text, _annotate = Axes.text, Axes.annotate

    def ax_text(self, *a, **k):
        s = a[2] if len(a) > 2 else k.get("s")
        return _text(self, *a, **_ann_kw(k, s))

    def ax_annotate(self, *a, **k):
        s = a[0] if a else k.get("text")
        return _annotate(self, *a, **_ann_kw(k, s))

    Axes.text, Axes.annotate = ax_text, ax_annotate

    # --- legends ----------------------------------------------------------
    # Legend sizes go through FontProperties rather than the keyword above,
    # so they need their own wrapper.
    def _legend_kw(kw):
        fs = kw.get("fontsize")
        if isinstance(fs, (int, float)):
            kw["fontsize"] = LEGEND if FORCE_LEGEND else fs * SCALE
        prop = kw.get("prop")
        if isinstance(prop, dict) and isinstance(prop.get("size"), (int, float)):
            kw["prop"] = dict(prop, size=(LEGEND if FORCE_LEGEND
                                          else prop["size"] * SCALE))
        return kw

    _ax_legend, _fig_legend = Axes.legend, Figure.legend
    Axes.legend = lambda self, *a, **k: _ax_legend(self, *a, **_legend_kw(k))
    Figure.legend = lambda self, *a, **k: _fig_legend(self, *a, **_legend_kw(k))

    # --- axis values: thin them until they have room ----------------------
    # Large type means the default tick count collides.  Rather than shrink
    # the numbers, drop some of them: the figure is drawn once, every label
    # measured, and the locator replaced where they are too close.  Log axes
    # keep their decade ticks.
    from matplotlib.ticker import MaxNLocator
    from matplotlib.figure import Figure as _Fig

    def _thin(fig):
        try:
            r = fig.canvas.get_renderer()
        except Exception:
            return
        for ax in fig.axes:
            for axis, horiz in ((ax.xaxis, True), (ax.yaxis, False)):
                if axis.get_scale() != "linear":
                    continue
                labs = [t for t in axis.get_ticklabels() if t.get_text()]
                if len(labs) < 4:
                    continue
                try:
                    ext = max((l.get_window_extent(r).width if horiz
                               else l.get_window_extent(r).height)
                              for l in labs)
                except Exception:
                    continue
                span = ax.bbox.width if horiz else ax.bbox.height
                if ext <= 0:
                    continue
                room = int(span / (ext * TICK_ROOM))
                if 2 <= room < len(labs):
                    axis.set_major_locator(
                        MaxNLocator(nbins=room, steps=[1, 2, 2.5, 5, 10]))

    _savefig = _Fig.savefig

    def savefig(self, *a, **k):
        try:
            self.canvas.draw()      # lay the labels out so they can be measured
            _thin(self)
        except Exception:
            pass
        return _savefig(self, *a, **k)

    _Fig.savefig = savefig

    # --- canvas size ------------------------------------------------------
    def _grow(kw):
        fs = kw.get("figsize")
        if fs:
            # read the module global, so override(fig_scale=...) is honoured
            kw["figsize"] = tuple(FIG_SCALE * float(v) for v in fs)
        return kw

    # Only plt.figure is wrapped: plt.subplots creates its figure by calling
    # it, so wrapping both would apply the multiplier twice.
    _figure = plt.figure
    plt.figure = lambda *a, **k: _figure(*a, **_grow(k))



    print("figstyle: labels %g pt, axis values %g pt, canvas x%.2f, %g dpi"
          % (LABEL, TICK, FIG_SCALE, DPI))


import contextlib   # noqa: E402


@contextlib.contextmanager
def override(size=None, label=None, tick=None, legend=None,
             fig_scale=None, ann_scale=None, text_sizes=None,
             autolayout=None):
    """Temporarily change the type size for one figure.

        with figstyle.override(size=26):
            draw_the_figure()

    `size` sets labels, axis values and legends together; the individual
    arguments override it.  `fig_scale` widens the canvas for this figure
    only, which is what long side-by-side panel titles need.  Left out, the
    canvas multiplier is unchanged and only the type differs.
    """
    global LABEL, TICK, LEGEND, SCALE, FIG_SCALE, FORCE_LEGEND
    global ANN_SCALE, TEXT_SIZES
    import matplotlib.pyplot as plt

    new_label = label if label is not None else (size if size else LABEL)
    new_tick = tick if tick is not None else (size if size else TICK)
    new_legend = legend if legend is not None else (size if size else LEGEND)

    keep = (LABEL, TICK, LEGEND, SCALE, FIG_SCALE, FORCE_LEGEND,
            ANN_SCALE, TEXT_SIZES)
    saved = {k: plt.rcParams[k] for k in
             ("font.size", "axes.labelsize", "axes.titlesize",
              "figure.titlesize", "xtick.labelsize", "ytick.labelsize",
              "legend.fontsize", "figure.autolayout")}
    LABEL, TICK, LEGEND = new_label, new_tick, new_legend
    SCALE = TICK / BASE_FONT          # the wrappers read this at call time
    if fig_scale is not None:
        FIG_SCALE = fig_scale
    # `size` means one size for everything, legends included
    if size is not None:
        FORCE_LEGEND = True
    if ann_scale is not None:
        ANN_SCALE = ann_scale
    if text_sizes is not None:
        TEXT_SIZES = dict(text_sizes)
    plt.rcParams.update({
        "font.size": TICK,
        "axes.labelsize": LABEL,
        "axes.titlesize": LABEL,
        "figure.titlesize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
    })
    if autolayout is not None:
        # A colour bar that steals space from a whole row of axes is undone
        # by tight_layout, which puts the axes back over it.  Figures built
        # that way have to opt out.
        plt.rcParams["figure.autolayout"] = autolayout
    try:
        yield
    finally:
        (LABEL, TICK, LEGEND, SCALE, FIG_SCALE, FORCE_LEGEND,
         ANN_SCALE, TEXT_SIZES) = keep
        plt.rcParams.update(saved)
