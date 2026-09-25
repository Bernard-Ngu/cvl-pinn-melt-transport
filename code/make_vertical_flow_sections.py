"""
Cross sections showing the VERTICAL flow field only.

Same sections as Figure 5, but the left-hand panels now show vertical velocity
(colour, red = upwelling) with purely vertical arrows, instead of temperature
with in-plane vectors.  Temperature is retained only as white contours for
context.  The horizontal component of the flow is deliberately not drawn.

Two section families are produced, because "Figure 5" can mean either:

  fig5v_*   orthogonal sections: W-E at 4.2 N through Mount Cameroon, and
            S-N at 10 E
  figPv_*   oblique sections: P1 along the CVL axis (SW oceanic to NE
            continental) and P2 from the Congo craton through Mount Cameroon
            to the NW

Output: ../figures_vertical_flow/ as both .png and .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

import fields as fl
import physics as ph
import figures as F
import make_profile_sections as P

OUT = os.path.abspath(os.path.join(F.HERE, "..", "figures_vertical_flow"))
os.makedirs(OUT, exist_ok=True)

YR = F.YR
PLIM_PCT = 1.0e-1                 # porosity colour ceiling, per cent
ZMAX = 400.0
MODELS = P.MODELS


def _panels(fig, axs, row, dist, zk, w, T, phi, Feq, lab, xlabel, title,
            wlim, plim, marks=None, vline=None):
    """Left: vertical velocity + vertical arrows.  Right: porosity."""
    ax = axs[row, 0]
    im = ax.pcolormesh(dist, zk, w.T, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(0.0, -wlim, wlim))
    ax.contour(dist, zk, T.T, levels=[1400, 1600, 1700], colors="0.25",
               linewidths=0.5, alpha=0.8)
    st = max(1, len(dist) // 30)
    zst = max(1, len(zk) // 22)
    u0 = np.zeros_like(w[::st, ::zst].T)
    # NOTE: quiver uses angles='uv', i.e. SCREEN space, so V>0 draws upward
    # even though the depth axis is inverted.  Pass +w (upward-positive) so
    # upwelling arrows point up.
    ax.quiver(dist[::st], zk[::zst], u0, w[::st, ::zst].T,
              scale=max(np.abs(w).max(), 1e-9) * 22, width=0.0035,
              color="k", alpha=0.75)
    plt.colorbar(im, ax=ax, fraction=0.03, label="vertical velocity (mm/yr)")
    ax.set_title("(%s) %s\nvertical flow only (red = upwelling)"
                 % ("ac"[row], title), pad=8)

    ax2 = axs[row, 1]
    pm = np.maximum(phi * 100.0, 1e-6)
    im = ax2.pcolormesh(dist, zk, pm.T, cmap="magma", shading="auto",
                        norm=LogNorm(1e-5, plim))
    ax2.contour(dist, zk, Feq.T, levels=[1e-4], colors="cyan", linewidths=0.8)
    plt.colorbar(im, ax=ax2, fraction=0.03, label="porosity (%)")
    ax2.set_title("(%s) %s\nporosity" % ("bd"[row], title), pad=8)

    for k, ax_ in enumerate((axs[row, 0], axs[row, 1])):
        ax_.plot(dist, lab, "-", color="k" if k == 0 else "w", lw=1.2)
        ax_.set_ylim(ZMAX, 0)
        ax_.set_xlim(dist.min(), dist.max())
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel("depth (km)")
        if vline is not None:
            ax_.axvline(vline, color="k" if k == 0 else "w", ls="--",
                        lw=0.7, alpha=0.7)
        for i, (name, sd) in enumerate(marks or []):
            ax_.plot(sd, 0, "v", ms=5, mfc="w", mec="k", mew=0.7,
                     clip_on=False, zorder=9)
            ax_.text(sd, 0.03 * ZMAX + (i % 2) * 0.16 * ZMAX, name,
                     fontsize=4.8, rotation=-90,
                     color="k" if k == 0 else "w",
                     ha="center", va="top", zorder=9)


# ------------------------------------------------------- orthogonal sections
def figure_orthogonal(tag, slug, label, wlim, plim):
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    bgd = np.load(os.path.join(F.DATA, "background.npz"))
    zk = s["z"] / 1e3
    jc = int(np.argmin(abs(lat - 4.2)))
    ic = int(np.argmin(abs(lon - 10.0)))

    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6))
    for row, (cut, axis, xlabel, title) in enumerate((
            (jc, lon, "longitude (E)", "W-E section at 4.2 N"),
            (ic, lat, "latitude (N)", "S-N section at 10.0 E"))):
        sel = (slice(None), cut) if row == 0 else (cut, slice(None))
        w = -s["uz"][sel] * YR * 1e3
        T = s["T"][sel]
        phi = s["phi"][sel]
        Feq = s["F"][sel]
        labg = bgd["lab"][:, :, 0]
        if row == 0:
            lab = labg[:, int(labg.shape[1] * cut / s["phi"].shape[1])]
        else:
            lab = labg[int(labg.shape[0] * cut / s["phi"].shape[0]), :]
        lab = np.interp(np.linspace(0, 1, len(axis)),
                        np.linspace(0, 1, len(lab)), lab) / 1e3
        _panels(fig, axs, row, axis, zk, w, T, phi, Feq, lab, xlabel, title,
                wlim, plim)
    fig.suptitle("Vertical flow field - %s" % label, y=1.02, fontsize=9.5)
    fig.tight_layout()
    F.save(fig, "fig5v_vertical_%s" % slug, OUT)


# ---------------------------------------------------------- oblique sections
def figure_oblique(tag, slug, label, tracks, wlim, plim):
    s = F.load(tag)
    zk = s["z"] / 1e3
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6))
    for row, (ptag, title, a, b) in enumerate(tracks):
        d = P.sample_track(s, a, b)
        lab = P.sample_lab(a, b)
        vol = P.project_volcanoes(a, b, max_off=60.0)
        marks = [(n, sd) for n, sd, _ in vol]
        vline = marks[0][1] if ptag == "P2" and marks else None
        _panels(fig, axs, row, d["dist"], zk, d["w"] * YR * 1e3, d["T"],
                d["phi"], d["F"], lab, "distance along %s (km)" % ptag,
                title, wlim, plim, marks=marks, vline=vline)
    fig.suptitle("Vertical flow field - %s" % label, y=1.02, fontsize=9.5)
    fig.tight_layout()
    F.save(fig, "figPv_vertical_%s" % slug, OUT)


def main():
    tracks = P.build_tracks()
    wlim = 0.0
    for tag, _, _ in MODELS:
        s = F.load(tag)
        wlim = max(wlim, float(np.percentile(np.abs(s["uz"]), 99.5)) * YR * 1e3)
    print("common vertical-velocity scale: +/- %.3f mm/yr" % wlim)
    print("common porosity scale:          1e-5 .. %.0e %%\n" % PLIM_PCT)

    for tag, slug, label in MODELS:
        figure_orthogonal(tag, slug, label, wlim, PLIM_PCT)
        figure_oblique(tag, slug, label, tracks, wlim, PLIM_PCT)

    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write(
            "Cross sections with the VERTICAL flow field only.\n\n"
            "Left panels  vertical velocity as colour (red = upwelling,\n"
            "             blue = downwelling) with purely vertical arrows.\n"
            "             Grey contours are the 1400, 1600 and 1700 K\n"
            "             isotherms; the horizontal flow is NOT drawn.\n"
            "Right panels porosity, cyan contour = equilibrium melting region.\n"
            "Black/white line: LAB.\n\n"
            "fig5v_vertical_<model>   W-E section at 4.2 N through Mount\n"
            "                         Cameroon, and S-N section at 10 E\n"
            "figPv_vertical_<model>   P1 along the CVL axis (SW oceanic to NE\n"
            "                         continental) and P2 from the Congo\n"
            "                         craton through Mount Cameroon to the NW\n\n"
            "The vertical-velocity scale (+/- %.3f mm/yr) and the porosity\n"
            "scale (1e-5 to %.0e %%) are common to all four models, so panels\n"
            "may be compared directly across files.\n\n"
            "Note: vertical velocity is a true 3-D field component, so unlike\n"
            "the in-plane vectors of Figure 5 nothing is lost by the choice of\n"
            "section orientation.\n" % (wlim, PLIM_PCT))
    print("\nwrote %d figures (png + pdf) to %s" % (2 * len(MODELS), OUT))


if __name__ == "__main__":
    main()
