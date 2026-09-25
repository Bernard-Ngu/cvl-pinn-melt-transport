#!/usr/bin/env python3
"""Redraw every manuscript figure at publication type size.

Identical to ../code/make_figures.py except that figstyle.apply() runs after
newfigs has been imported, so the style in figstyle.py wins over the 8 pt
defaults newfigs sets for itself. Output goes to ../figures, the same place
the manuscript reads from.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
# every module lives beside this file
OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
sys.path.insert(0, HERE)

import newfigs as NF        # noqa: E402  (sets its own rcParams on import)
import figstyle             # noqa: E402

MAIN = "A"                  # free lithosphere: the model reported


# manuscript figure number -> what draws it
def _draw(n):
    if n == 2:
        with figstyle.override(size=figstyle.FIG2, label=figstyle.FIG2_LABEL,
                               fig_scale=figstyle.FIG2_SCALE):
            NF.fig_setting()
    elif n == 3:
        NF.fig_partition()
    elif n == 4:
        # the colour bars take their space from a whole row of panels,
        # and tight_layout puts the panels back on top of them
        with figstyle.override(autolayout=False):
            NF.fig_melt_porosity_slices()
    elif n == 5:
        NF.fig_flow_slices()
    elif n == 6:
        NF.fig_cross_sections()
    elif n == 7:
        # the delaminated body is what this panel is for, so its label is set
        # outright rather than scaled from the 6 pt the figure code asks for
        with figstyle.override(ann_scale=figstyle.FIG7_ANN,
                               text_sizes={"delaminated body":
                                           figstyle.FIG7_DELAM}):
            NF.fig_oblique_sections()
    elif n == 8:
        NF.fig_budget()
    elif n == 9:
        NF.fig_profiles()
    elif n == 10:
        NF.fig_oblique_melt()
    elif n == 11:
        with figstyle.override(size=figstyle.FIG11):
            NF.F.fig_lid_comparison()
        NF._ren("fig9_lid_comparison", "fig10_lid_comparison")
    elif n == 12:
        NF.F.fig_lid_profiles()
        NF._ren("fig10_lid_profiles", "fig11_lid_profiles")
    elif n == 13:
        NF.F.fig_convergence(tags=("A", "L", "S"))
        NF._ren("fig8_convergence", "fig12_convergence")
    else:
        raise SystemExit("no manuscript figure %r is drawn here (2-13)" % n)


def only(numbers):
    """Redraw just these manuscript figures, leaving the rest alone."""
    figstyle.apply()
    NF.configure(main=MAIN, out=OUT)
    for n in numbers:
        _draw(int(n))
    print("figures written to", OUT)


def main():
    figstyle.apply()        # must come after newfigs
    NF.configure(main=MAIN, out=OUT)
    # Fig. 2: three map panels side by side with long titles.  Everything at
    # one size, on a wider canvas so the titles do not run into each other.
    with figstyle.override(size=figstyle.FIG2, label=figstyle.FIG2_LABEL,
                           fig_scale=figstyle.FIG2_SCALE):
        NF.fig_setting()            # manuscript Fig. 2
    NF.fig_partition()              # Fig. 3
    # Fig. 4's colour bars take their space from a whole row of panels,
    # and tight_layout puts the panels back on top of them
    with figstyle.override(autolayout=False):
        NF.fig_melt_porosity_slices()   # Fig. 4
    NF.fig_flow_slices()            # Fig. 5
    NF.fig_cross_sections()         # Fig. 6
    with figstyle.override(ann_scale=figstyle.FIG7_ANN,
                           text_sizes={"delaminated body":
                                       figstyle.FIG7_DELAM}):
        NF.fig_oblique_sections()   # Fig. 7
    NF.fig_budget()                 # Fig. 8
    NF.fig_profiles()               # Fig. 9
    NF.fig_oblique_melt()           # Fig. 10

    # Figs. 11, 12 and 13 come from three separate calls inside NF.fig_lids();
    # they are made here one at a time so that Fig. 11, the three-by-three
    # grid of lid treatments against depth, can be drawn at a single larger
    # size.  Small panels read better with one size throughout than with a
    # label/value split.
    with figstyle.override(size=figstyle.FIG11):
        NF.F.fig_lid_comparison()                 # Fig. 11
    NF.F.fig_lid_profiles()                       # Fig. 12
    NF.F.fig_convergence(tags=("A", "L", "S"))    # Fig. 13
    NF._ren("fig9_lid_comparison", "fig10_lid_comparison")
    NF._ren("fig10_lid_profiles", "fig11_lid_profiles")
    NF._ren("fig8_convergence", "fig12_convergence")
    print("figures written to", OUT)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a.strip()]
    if args:
        only(a.lower().lstrip("fig") for a in args)
    else:
        main()
