#!/usr/bin/env python3
"""Draw every figure in the manuscript, in manuscript order.

newfigs.py holds the figure definitions; this driver fixes the model whose
solution they are drawn from and the directory they are written to, so that
the figure set is reproducible from one command.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import newfigs as NF

MAIN = "A"                                   # free lithosphere: the model reported
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def main():
    NF.configure(main=MAIN, out=os.path.abspath(OUT))
    NF.fig_setting()                # 1  setting
    NF.fig_partition()              # 2  Vs partition
    NF.fig_melt_porosity_slices()   # 3  melt and porosity slices
    NF.fig_flow_slices()            # 4  flow slices
    NF.fig_cross_sections()         # 5  orthogonal sections
    NF.fig_oblique_sections()       # 6  oblique sections
    NF.fig_budget()                 # 7  melt budget
    NF.fig_profiles()               # 8  profiles
    NF.fig_oblique_melt()           # 9  oblique melt
    NF.fig_lids()                   # 10, 11, 12  lid variants and convergence
    print("figures written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
