"""Figures for Section 2.11, from the results written by exp_run.py.

  figA1_architecture_sweep   (a) momentum residual over the 3 x 3 grid
                             (b) all five residuals, configuration by
                                 configuration, with the reported network
                                 marked
                             (c) training histories, all nine runs

  figA2_physics_ablation     (a) the five residuals under the three
                                 objectives
                             (b) the buoyancy-work integral, whose sign is
                                 the whole point
                             (c) porosity misfit against the tomographic
                                 estimate

Run it after run_experiments.sh, or at any point to see what has finished:
missing configurations are skipped with a warning rather than crashing.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "experiments"))
RES = os.path.join(EXP, "results")
FIGS = os.path.join(EXP, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "figure.dpi": 160, "savefig.dpi": 300,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

DEPTHS = [3, 5, 7]
WIDTHS = [32, 64, 128]
GROUPS = [("mom", "momentum"), ("rheo", "rheology"), ("solid", "solid mass"),
          ("melt", "melt mass"), ("darcy", "Darcy")]
GCOL = ["#1b4965", "#5fa8d3", "#c08552", "#7d4f50", "#4c956c"]
REPORTED = (5, 64)


def load():
    out = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        r = json.load(open(f))
        out[r["tag"]] = r
    return out


def tag_of(d, w):
    return "d%dw%d" % (d, w)


def history(tag):
    p = os.path.join(RES, "ck_%s.npz" % tag)
    if not os.path.exists(p):
        return None
    h = np.load(p)["hist"]
    return h if len(h) else None


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, "%s.%s" % (name, ext)),
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("figures", name + ".pdf/.png"))


# ----------------------------------------------------------------- helpers
def _spread(ys, lo, hi, gap=0.055):
    """Nudge label positions apart in log space so they do not collide."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    span = np.log10(hi) - np.log10(lo)
    g = gap * span
    prev = None
    for i in order:
        l = np.log10(out[i])
        if prev is not None and l - prev < g:
            l = prev + g
        out[i] = 10.0 ** l
        prev = l
    return out


# ------------------------------------------------------------------ fig A1
def fig_sweep(R):
    have = [(d, w) for d in DEPTHS for w in WIDTHS if tag_of(d, w) in R]
    if not have:
        print("architecture sweep: nothing to plot yet")
        return
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.5), layout="constrained",
                            width_ratios=[1.0, 1.3, 1.05])

    # (a) the momentum residual over the grid ----------------------------
    # Every cell is annotated, so a colour bar would only take space from
    # the neighbouring panel; the shading is there to be read at a glance.
    ax = axs[0]
    M = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    for i, d in enumerate(DEPTHS):
        for j, w in enumerate(WIDTHS):
            r = R.get(tag_of(d, w))
            if r:
                M[i, j] = r["mom"]
    ax.imshow(M, cmap="YlGnBu", origin="lower", aspect="auto")
    fin = M[np.isfinite(M)]
    cut = (fin.min() + 0.55 * np.ptp(fin)) if len(fin) else np.inf
    for i in range(len(DEPTHS)):
        for j in range(len(WIDTHS)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "not run", ha="center", va="center",
                        fontsize=7, color="0.45")
                continue
            ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center",
                    fontsize=8.5,
                    color="white" if M[i, j] > cut else "0.12")
    ax.set_xticks(range(len(WIDTHS)), [str(w) for w in WIDTHS])
    ax.set_yticks(range(len(DEPTHS)), [str(d) for d in DEPTHS])
    ax.set_xlabel("neurons per layer")
    ax.set_ylabel("hidden layers")
    ax.set_title("(a) momentum residual")
    i0, j0 = DEPTHS.index(REPORTED[0]), WIDTHS.index(REPORTED[1])
    ax.add_patch(plt.Rectangle((j0 - .5, i0 - .5), 1, 1, fill=False,
                               ec="#b23a48", lw=1.8, zorder=5))
    ax.text(j0, i0 + 0.40, "reported", ha="center", va="center",
            fontsize=6.5, color="#b23a48", zorder=6)

    # (b) every residual, configuration by configuration ------------------
    # Labelled at the right-hand end rather than in a legend, which would
    # sit on top of the lines.
    ax = axs[1]
    xs = np.arange(len(have))
    vals = {}
    for (k, name), c in zip(GROUPS, GCOL):
        v = np.array([R[tag_of(d, w)][k] for (d, w) in have])
        vals[k] = v
        ax.plot(xs, v, "o-", ms=3.4, lw=1.1, color=c, clip_on=False)
    allv = np.concatenate(list(vals.values()))
    lo, hi = allv.min() / 2.2, allv.max() * 2.2
    ends = _spread([vals[k][-1] for k, _ in GROUPS], lo, hi)
    for (k, name), c, ye in zip(GROUPS, GCOL, ends):
        ax.annotate(name, xy=(xs[-1], vals[k][-1]),
                    xytext=(xs[-1] + 0.45, ye), fontsize=7, color=c,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5,
                                    shrinkA=1, shrinkB=1))
    if REPORTED in have:
        ax.axvline(have.index(REPORTED), color="#b23a48", lw=1.0, ls="--",
                   zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(-0.4, len(have) - 1 + 2.4)
    ax.set_ylim(lo, hi)
    ax.set_xticks(xs, ["%d x %d" % (d, w) for (d, w) in have],
                  rotation=45, ha="right")
    ax.set_ylabel("relative r.m.s. residual")
    ax.set_title("(b) residuals across the sweep")
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    # (c) training histories ----------------------------------------------
    # Nine curves with nine legend entries is unreadable; the spread is the
    # point, so it is drawn as an envelope with the reported run on top.
    ax = axs[2]
    H = {t: history(t) for t in [tag_of(d, w) for (d, w) in have]}
    H = {t: h for t, h in H.items() if h is not None}
    if H:
        # The runs are not all logged on the same iteration grid (the logging
        # interval changed part-way through the sweep), so each history is
        # interpolated onto a common axis before the envelope is taken.
        lo = max(h[0, 0] for h in H.values())
        hi = min(h[-1, 0] for h in H.values())
        it = np.linspace(lo, hi, 60)
        Y = np.array([np.interp(it, h[:, 0], h[:, 1]) for h in H.values()])
        ax.fill_between(it, Y.min(0), Y.max(0), color="#5fa8d3", alpha=0.28,
                        lw=0, label="range over the sweep")
        ax.plot(it, np.median(Y, 0), color="#1b4965", lw=1.1, label="median")
        rt = tag_of(*REPORTED)
        if rt in H:
            ax.plot(H[rt][:, 0], H[rt][:, 1], color="#b23a48", lw=1.2,
                    label="reported, %d x %d" % REPORTED)
        ax.set_xlim(lo, hi)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective")
    ax.set_title("(c) training history")
    ax.grid(lw=0.4, alpha=0.5)

    save(fig, "figA1_architecture_sweep")


# ------------------------------------------------------------------ fig A2
ABL = [("d5w64", "full\nobjective", "#1b4965"),
       ("nowork", "energy constraint\nremoved", "#c08552")]


def fig_ablation(R):
    have = [(t, lab, c) for t, lab, c in ABL if t in R]
    if not have:
        print("energy-constraint run: nothing to plot yet")
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), layout="constrained",
                            width_ratios=[1.8, 1.0])

    # (a) the five residuals ----------------------------------------------
    ax = axs[0]
    xs = np.arange(len(GROUPS))
    bw = 0.8 / len(have)
    top = 0.0
    for n, (t, lab, c) in enumerate(have):
        v = [R[t][k] for k, _ in GROUPS]
        top = max(top, max(v))
        ax.bar(xs + (n - (len(have) - 1) / 2) * bw, v, bw * 0.92, color=c,
               label=lab.replace("\n", " "))
    ax.set_yscale("log")
    ax.set_ylim(top=top * 12)                    # headroom for the legend
    ax.set_xticks(xs, [n for _, n in GROUPS], rotation=20, ha="right")
    ax.set_ylabel("relative r.m.s. residual")
    ax.set_title("(a) conservation residuals")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=len(have), loc="upper center")

    # (b) the buoyancy-work integral ---------------------------------------
    ax = axs[1]
    vals = [R[t]["work"] for t, _, _ in have]
    ax.bar(range(len(have)), vals,
           color=["#4c956c" if v > 0 else "#b23a48" for v in vals], width=0.6)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(range(len(have)), [lab for _, lab, _ in have], fontsize=7)
    ax.set_ylabel("buoyancy work")
    ax.set_title("(b) energy admissibility")
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = 0.30 * max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.5, 0.965, "an admissible flow dissipates: W > 0",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color="0.35")

    save(fig, "figA2_energy_constraint")


if __name__ == "__main__":
    R = load()
    if not R:
        print("no results yet - run ./run_experiments.sh first")
    else:
        fig_sweep(R)
        fig_ablation(R)
        missing = [t for t in [tag_of(d, w) for d in DEPTHS for w in WIDTHS]
                   + ["nowork"] if t not in R]
        if missing:
            print("still to run:", " ".join(missing))
