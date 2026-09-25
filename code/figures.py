"""
Figures for the rigid-lid revision of the CVL PINN manuscript.

Differences from ../pinn/figures.py:
  * every figure is written as both .png and .pdf
  * four-panel figures are laid out 2x2 and five/six-panel figures 2x3
  * solutions are searched for in ./runs first, then ../pinn/runs, so the
    lid runs (L, S) and the original runs (A, B) can be mixed freely
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

import fields as fl
import physics as ph

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [os.path.join(HERE, "..", "runs")]
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

YR = 3.15576e7
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "figure.dpi": 160, "savefig.dpi": 300,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Cameroon Volcanic Line volcanic centres (approximate summit coordinates).
CVL = [
    ("Annobon",     -1.437,  5.633),
    ("Sao Tome",     0.320,  6.600),
    ("Principe",     1.610,  7.400),
    ("Bioko",        3.350,  8.630),
    ("Mt Cameroon",  4.203,  9.170),
    ("Manengouba",   5.030,  9.830),
    ("Bambouto",     5.650, 10.050),
    ("Mt Oku",       6.200, 10.519),
    ("Ngaoundere",   7.250, 13.670),
    ("Biu",         10.750, 12.200),
    ("Mandara",     10.800, 13.700),
]

LABELS = {"A": "free lithosphere", "L": "lid locked", "S": "lid slip"}


def save(fig, name, outdir=None):
    """Write both raster and vector versions."""
    d = outdir or FIGS
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s/%s.{png,pdf}" % (os.path.basename(d), name))


def load(tag):
    for d in RUNS:
        p = os.path.join(d, "solution_%s.npz" % tag)
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError("solution_%s.npz not found in %s" % (tag, RUNS))


def load_hist(tag):
    for d in RUNS:
        p = os.path.join(d, "params_%s.npz" % tag)
        if os.path.exists(p):
            return np.load(p)["hist"]
    raise FileNotFoundError("params_%s.npz" % tag)


def lonlat_axes(s):
    lon, _ = fl.xy_to_lonlat(s["x"] - fl.LX / 2, np.zeros_like(s["x"]))
    _, lat = fl.xy_to_lonlat(np.zeros_like(s["y"]), s["y"] - fl.LY / 2)
    return lon, lat


def kidx(s, d_km):
    return int(np.argmin(abs(s["z"] - d_km * 1e3)))


def draw_cvl(ax, labels=False, ms=3.2):
    for name, la, lo in CVL:
        if not (ph.LON_MIN < lo < ph.LON_MAX and ph.LAT_MIN < la < ph.LAT_MAX):
            continue
        ax.plot(lo, la, "^", ms=ms, mfc="none", mec="k", mew=0.7, zorder=6)
        if labels:
            ax.annotate(name, (lo, la), textcoords="offset points",
                        xytext=(3, 2), fontsize=5.2, zorder=6)


def basemap(ax, lon, lat):
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(fl.LAT0)))
    ax.set_xlabel("longitude (E)")
    ax.set_ylabel("latitude (N)")
    ax.tick_params(length=2)


# ============================================================ Fig 1  setting
def fig_setting():
    bgd = np.load(os.path.join(DATA, "background.npz"))
    s = load("L")
    lon, lat = lonlat_axes(s)
    lonb = np.linspace(ph.LON_MIN, ph.LON_MAX, bgd["dlnvs"].shape[0])
    latb = np.linspace(ph.LAT_MIN, ph.LAT_MAX, bgd["dlnvs"].shape[1])

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3))

    ax = axs[0]
    im = ax.pcolormesh(lonb, latb, (bgd["lab"][:, :, 0] / 1e3).T,
                       cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=ax, label="LAB depth (km)", fraction=0.046)
    ax.set_title("(a) lithosphere-asthenosphere boundary")
    draw_cvl(ax, labels=True)
    basemap(ax, lon, lat)

    ax = axs[1]
    k = int(np.argmin(abs(np.linspace(0, fl.LZ, bgd["dlnvs"].shape[2]) - 100e3)))
    im = ax.pcolormesh(lonb, latb, 100 * bgd["dlnvs"][:, :, k].T,
                       cmap="RdBu_r", vmin=-6, vmax=6, shading="auto")
    plt.colorbar(im, ax=ax, label="dlnVs (%)", fraction=0.046)
    ax.set_title("(b) shear-velocity anomaly, 100 km")
    draw_cvl(ax)
    basemap(ax, lon, lat)

    ax = axs[2]
    zb = np.linspace(0, fl.LZ, bgd["dlnvs"].shape[2]) / 1e3
    P = ph.lithostatic_pressure(zb * 1e3)
    ax.plot(bgd["T"].mean(axis=(0, 1)), zb, "k-", lw=1.3, label="mean T (this study)")
    ax.plot(bgd["T_plugin"].mean(axis=(0, 1)), zb, "k--", lw=1.0,
            label="mean T (plugin conversion)")
    ax.plot(ph.adiabat(zb * 1e3), zb, color="0.55", lw=0.9, label="adiabat")
    ax.plot(ph.solidus(P), zb, "r-", lw=1.1, label="dry solidus")
    ax.fill_betweenx(zb, bgd["T"].min(axis=(0, 1)), bgd["T"].max(axis=(0, 1)),
                     color="0.85", zorder=0)
    ax.set_ylim(400, 0)
    ax.set_xlim(600, 2200)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("depth (km)")
    ax.set_title("(c) thermal structure and solidi")
    ax.legend(fontsize=5.6, loc="lower left", frameon=False)

    fig.tight_layout()
    save(fig, "fig1_setting")


# ================================================== Fig 2  Vs partition
def fig_partition():
    dep = np.array([80e3, 120e3, 200e3, 300e3])
    dl = np.linspace(-0.07, 0.01, 200)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(dep)))
    for d, c in zip(dep, cols):
        dT, F = ph.invert_temperature_melt(dl, np.full_like(dl, d))
        axs[0].plot(100 * dl, dT, color=c, lw=1.2, label="%d km" % (d / 1e3))
        axs[0].plot(100 * dl, -dl / ph.ALPHA * ph.VS_TO_DENSITY, ":",
                    color=c, lw=0.9)
        axs[1].plot(100 * dl, 100 * F, color=c, lw=1.2)
    axs[0].set_xlabel("dlnVs (%)")
    axs[0].set_ylabel("temperature excess (K)")
    axs[0].set_title("(a) thermal interpretation\nsolid: self-consistent, dotted: plugin")
    axs[0].legend(fontsize=6, frameon=False)
    axs[1].set_xlabel("dlnVs (%)")
    axs[1].set_ylabel("equilibrium melt fraction (%)")
    axs[1].set_title("(b) melt fraction")
    axs[1].set_ylim(0, 2.0)
    fig.tight_layout()
    save(fig, "fig2_vs_partition")


# ============================================ Fig 3  depth slices of melt
def fig_melt_slices(tags=("A",), depths=(100, 150, 250, 300)):
    fig, axs = plt.subplots(len(tags), len(depths),
                            figsize=(2.55 * len(depths), 2.75 * len(tags)))
    axs = np.atleast_2d(axs)
    names = dict(LABELS)
    for r, tag in enumerate(tags):
        s = load(tag)
        lon, lat = lonlat_axes(s)
        F = s["F"] * 100.0
        vmax = max(F.max(), 1e-3)
        for c, d in enumerate(depths):
            ax = axs[r, c]
            k = kidx(s, d)
            f = np.maximum(F[:, :, k], 1e-4)
            im = ax.pcolormesh(lon, lat, f.T, cmap="magma",
                               norm=LogNorm(vmin=1e-4, vmax=vmax),
                               shading="auto")
            p = s["phi"][:, :, k] * 100
            if p.max() > 2e-4:
                ax.contour(lon, lat, p.T, levels=[5e-4, 1e-3, 1.5e-2],
                           colors="cyan", linewidths=0.6)
            draw_cvl(ax)
            basemap(ax, lon, lat)
            ax.set_title("%s\n%d km" % (names.get(tag, tag) if c == 0 else "", d))
            if c:
                ax.set_ylabel("")
            plt.colorbar(im, ax=ax, fraction=0.046, label="melt fraction (%)")
    fig.suptitle("Equilibrium melt fraction (colour) and retained porosity "
                 "from the PINN (cyan contours)", y=1.02, fontsize=9)
    fig.tight_layout()
    save(fig, "fig3_melt_slices")


# ================================================= Fig 4  flow field, 2 x 2
def fig_flow_slices(tag="L", depths=(100, 150, 250, 300), name=None,
                    outdir=None, vlim=None):
    s = load(tag)
    lon, lat = lonlat_axes(s)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4))
    axs = axs.ravel()
    w = -s["uz"] * YR * 1e3
    lim = vlim or np.percentile(abs(w), 99.5)
    step = max(1, len(lon) // 16)
    uh_max = np.hypot(s["ux"], s["uy"]).max() * YR * 1e3
    for c, d in enumerate(depths):
        ax = axs[c]
        k = kidx(s, d)
        im = ax.pcolormesh(lon, lat, w[:, :, k].T, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
        ax.quiver(lon[::step], lat[::step],
                  s["ux"][::step, ::step, k].T * YR * 1e3,
                  s["uy"][::step, ::step, k].T * YR * 1e3,
                  scale=max(uh_max, 1e-9) * 12, width=0.005, color="k", alpha=0.8)
        ax.contour(lon, lat, s["phi"][:, :, k].T, levels=[1e-6],
                   colors="lime", linewidths=0.7)
        draw_cvl(ax)
        basemap(ax, lon, lat)
        ax.set_title("(%s) %d km" % ("abcd"[c], d))
        plt.colorbar(im, ax=ax, fraction=0.046, label="upwelling (mm/yr)")
    fig.suptitle("Solid flow field, %s  (arrows: horizontal velocity, "
                 "max %.2f mm/yr)" % (LABELS.get(tag, tag), uh_max),
                 y=1.00, fontsize=9)
    fig.tight_layout()
    save(fig, name or "fig4_flow_slices", outdir)


# ================================================ Fig 5  cross sections
def fig_cross_sections(tag="L", name=None, outdir=None, plim=None):
    s = load(tag)
    lon, lat = lonlat_axes(s)
    bgd = np.load(os.path.join(DATA, "background.npz"))
    zk = s["z"] / 1e3

    jc = int(np.argmin(abs(lat - 4.2)))
    ic = int(np.argmin(abs(lon - 10.0)))

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 5.4))
    for row, (cut, axis, coord, label) in enumerate((
            (jc, lon, "longitude (E)", "W-E section at 4.2 N"),
            (ic, lat, "latitude (N)", "S-N section at 10.0 E"))):
        T = s["T"][:, cut, :] if row == 0 else s["T"][cut, :, :]
        phi = s["phi"][:, cut, :] if row == 0 else s["phi"][cut, :, :]
        w = -(s["uz"][:, cut, :] if row == 0 else s["uz"][cut, :, :]) * YR * 1e3
        uh = (s["ux"][:, cut, :] if row == 0 else s["uy"][cut, :, :]) * YR * 1e3

        ax = axs[row, 0]
        im = ax.pcolormesh(axis, zk, T.T, cmap="inferno", shading="auto",
                           vmin=600, vmax=1950)
        ax.contour(axis, zk, T.T, levels=[1400, 1600, 1700], colors="w",
                   linewidths=0.5)
        st = max(1, len(axis) // 22)
        # NOTE: quiver uses angles='uv', i.e. SCREEN space, so V>0 draws
        # upward even though the depth axis is inverted.  Pass +w
        # (upward-positive) so upwelling arrows point up.
        ax.quiver(axis[::st], zk[::4], uh[::st, ::4].T, w[::st, ::4].T,
                  scale=max(np.abs(uh).max(), 1e-9) * 25, width=0.003,
                  color="w", alpha=0.85)
        ax.set_title("%s - temperature and solid flow\n%s"
                     % (label, LABELS.get(tag, tag)))
        plt.colorbar(im, ax=ax, fraction=0.03, label="T (K)")

        ax = axs[row, 1]
        pm = np.maximum(phi * 100, 1e-6)
        im = ax.pcolormesh(axis, zk, pm.T, cmap="magma", shading="auto",
                           norm=LogNorm(1e-5, plim or max(pm.max(), 1e-4)))
        ax.contour(axis, zk, (s["F"][:, cut, :] if row == 0
                              else s["F"][cut, :, :]).T, levels=[1e-4],
                   colors="cyan", linewidths=0.8)
        ax.set_title("%s - porosity" % label)
        plt.colorbar(im, ax=ax, fraction=0.03, label="porosity (%)")

        for ax in axs[row]:
            lab = (bgd["lab"][:, int(bgd["lab"].shape[1] * cut / s["phi"].shape[1]), 0]
                   if row == 0 else
                   bgd["lab"][int(bgd["lab"].shape[0] * cut / s["phi"].shape[0]), :, 0])
            la = np.linspace(axis.min(), axis.max(), len(lab))
            ax.plot(la, lab / 1e3, "w-", lw=1.1)
            ax.set_ylim(400, 0)
            ax.set_xlabel(coord)
            ax.set_ylabel("depth (km)")
    fig.tight_layout()
    save(fig, name or "fig5_cross_sections", outdir)


# ============================================ Fig 6  melt production / flux
def fig_melt_budget(tags=("A",)):
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.1))
    s = load(tags[0])
    lon, lat = lonlat_axes(s)
    dz = np.gradient(s["z"])

    prod = (np.maximum(s["Gamma"], 0) * dz[None, None, :]).sum(axis=2)
    prod_mMyr = prod / ph.RHO_S * YR * 1e6
    im = axs[0].pcolormesh(lon, lat, prod_mMyr.T, cmap="hot_r", shading="auto")
    plt.colorbar(im, ax=axs[0], fraction=0.046,
                 label="melt production (m / Myr)")
    axs[0].set_title("(a) integrated melt production")
    draw_cvl(axs[0], labels=True)
    basemap(axs[0], lon, lat)

    bgd = np.load(os.path.join(DATA, "background.npz"))
    labg = bgd["lab"][:, :, 0]
    labi = np.array([[labg[int(i * labg.shape[0] / len(lon)),
                           int(j * labg.shape[1] / len(lat))]
                      for j in range(len(lat))] for i in range(len(lon))])
    kk = np.clip(np.searchsorted(s["z"], labi.ravel()) - 1, 0,
                 len(s["z"]) - 1).reshape(labi.shape)
    ii, jj = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)), indexing="ij")
    qlab = -s["qz"][ii, jj, kk] * YR * 1e3
    im = axs[1].pcolormesh(lon, lat, np.maximum(qlab, 0).T, cmap="magma",
                           shading="auto")
    plt.colorbar(im, ax=axs[1], fraction=0.046, label="mm/yr")
    axs[1].set_title("(b) upward melt flux at the LAB")
    draw_cvl(axs[1])
    basemap(axs[1], lon, lat)

    for tag, c in zip(tags, ("C0", "C1")):
        t = load(tag)
        pr = np.maximum(t["Gamma"], 0).mean(axis=(0, 1)) / ph.RHO_S * YR
        axs[2].plot(pr, t["z"] / 1e3, color=c, lw=1.3,
                    label="hydrous")
    axs[2].set_ylim(400, 0)
    axs[2].set_xlabel("mean melting rate (1/yr)")
    axs[2].set_ylabel("depth (km)")
    axs[2].set_title("(c) where melt is produced")
    if len(tags) > 1:
        axs[2].legend(fontsize=6.5, frameon=False)
    axs[2].set_xscale("log")
    fig.tight_layout()
    save(fig, "fig6_melt_budget")


# ================================================ Fig 7  profiles, 2 x 3
def fig_profiles(tags=("A",)):
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.6), sharey=True)
    axs = axs.ravel()
    for tag, c in zip(tags, ("C0", "C1")):
        s = load(tag)
        z = s["z"] / 1e3
        nm = "hydrous"
        axs[0].plot(s["T"].mean(axis=(0, 1)), z, color=c, lw=1.3, label=nm)
        axs[1].plot(s["F"].mean(axis=(0, 1)) * 100, z, color=c, lw=1.3)
        axs[2].plot(s["phi"].mean(axis=(0, 1)) * 100, z, color=c, lw=1.3)
        axs[2].plot(s["phi"].max(axis=(0, 1)) * 100, z, color=c, lw=0.8, ls="--")
        sp = np.hypot(np.hypot(s["ux"], s["uy"]), s["uz"]) * YR * 1e3
        axs[3].plot(sp.mean(axis=(0, 1)), z, color=c, lw=1.3)
        axs[3].plot(sp.max(axis=(0, 1)), z, color=c, lw=0.8, ls="--")
        axs[4].plot(np.exp(np.log(s["eta"]).mean(axis=(0, 1))), z, color=c, lw=1.3)
        axs[5].plot(np.maximum(s["Gamma"], 0).mean(axis=(0, 1)) / ph.RHO_S * YR,
                    z, color=c, lw=1.3)
    P = ph.lithostatic_pressure(s["z"])
    axs[0].plot(ph.solidus(P), z, "r-", lw=0.9)
    titles = ("(a) temperature (K)", "(b) F_eq (%)", "(c) porosity (%)",
              "(d) |u| (mm/yr)", "(e) viscosity (Pa s)",
              "(f) melting rate (1/yr)")
    for ax, t in zip(axs, titles):
        ax.set_xlabel(t)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylim(400, 0)
    axs[0].set_ylabel("depth (km)")
    axs[3].set_ylabel("depth (km)")
    axs[0].legend(fontsize=6.5, frameon=False)
    for i in (2, 4, 5):
        axs[i].set_xscale("log")
    fig.suptitle("Horizontally averaged structure (dashed: maxima)",
                 y=1.01, fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_profiles")


# ============================================== Fig 8  training / residuals
def fig_convergence(tags=("A", "L", "S")):
    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.0))
    axs = axs.ravel()
    names = dict(mom="momentum", rheo="rheology", solid="solid mass",
                 melt="melt mass", darcy="Darcy")
    cols = dict(mom="C3", rheo="C0", solid="C2", melt="C1", darcy="C4")
    for _i, (ax, tag) in enumerate(zip(axs[:3], tags)):
        h = load_hist(tag)
        for i, k in enumerate(("mom", "rheo", "solid", "melt", "darcy")):
            ax.semilogy(h[:, 0], np.maximum(h[:, 2 + i], 1e-12),
                        color=cols[k], lw=1.1, label=names[k])
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean squared residual")
        ax.set_title("(%s) %s" % ("abc"[_i], LABELS.get(tag, tag)))
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].legend(fontsize=6, frameon=False, ncol=2)

    ax = axs[3]
    for tag, col in zip(("A", "L", "S"), ("C0", "C3", "C2")):
        s = load(tag)
        z = s["z"] / 1e3
        ax.semilogx(np.exp(np.log(np.maximum(s["eta"], 1e18)).mean(axis=(0, 1))),
                    z, color=col, lw=1.3, label=LABELS[tag])
    ax.set_ylim(660, 0)
    ax.set_xlabel("viscosity (Pa s)")
    ax.set_ylabel("depth (km)")
    ax.set_title("(d) mean effective viscosity")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    save(fig, "fig8_convergence")


# ====================================== Fig 9  rigid-lid comparison (maps)
def fig_lid_comparison(depths=(150, 250, 300)):
    runs = [("A", "free lithosphere"), ("L", "lid locked"), ("S", "lid slip")]
    sols = [(n, load(t)) for t, n in runs]
    fig, axs = plt.subplots(len(sols), len(depths),
                            figsize=(3.0 * len(depths), 2.9 * len(sols)))
    lim = max(np.percentile(np.abs(s["uz"]), 99.7) for _, s in sols) * YR * 1e3
    for r, (name, s) in enumerate(sols):
        lon, lat = lonlat_axes(s)
        w = -s["uz"] * YR * 1e3
        step = max(1, len(lon) // 14)
        uh_max = np.hypot(s["ux"], s["uy"]).max() * YR * 1e3
        for c, d in enumerate(depths):
            ax = axs[r, c]
            k = kidx(s, d)
            im = ax.pcolormesh(lon, lat, w[:, :, k].T, cmap="RdBu_r",
                               norm=TwoSlopeNorm(0, -lim, lim), shading="auto")
            ax.quiver(lon[::step], lat[::step],
                      s["ux"][::step, ::step, k].T * YR * 1e3,
                      s["uy"][::step, ::step, k].T * YR * 1e3,
                      scale=max(uh_max, 1e-9) * 12, width=0.005,
                      color="k", alpha=0.8)
            draw_cvl(ax)
            basemap(ax, lon, lat)
            ax.set_title(("%s\n" % name if c == 0 else "") + "%d km" % d)
            if c:
                ax.set_ylabel("")
            plt.colorbar(im, ax=ax, fraction=0.046,
                         label="upwelling (mm/yr)" if c == len(depths) - 1 else "")
    fig.suptitle("Effect of a rigid lithospheric lid on the recovered flow",
                 y=1.01, fontsize=9)
    fig.tight_layout()
    save(fig, "fig9_lid_comparison")


# ================================== Fig 10  rigid-lid profiles, 2 x 2
def fig_lid_profiles():
    runs = [("A", "free lithosphere", "C0"), ("L", "lid locked", "C3"),
            ("S", "lid slip", "C2")]
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.8), sharey=True)
    axs = axs.ravel()
    for tag, name, col in runs:
        s = load(tag)
        z = s["z"] / 1e3
        sp = np.hypot(np.hypot(s["ux"], s["uy"]), s["uz"]) * YR * 1e3
        w = -s["uz"] * YR * 1e3
        axs[0].plot(sp.mean(axis=(0, 1)), z, color=col, lw=1.3, label=name)
        axs[0].plot(sp.max(axis=(0, 1)), z, color=col, lw=0.8, ls="--")
        axs[1].plot(w.max(axis=(0, 1)), z, color=col, lw=1.3)
        axs[1].plot(w.min(axis=(0, 1)), z, color=col, lw=0.8, ls="--")
        axs[2].plot(np.hypot(s["ux"], s["uy"]).mean(axis=(0, 1)) * YR * 1e3, z,
                    color=col, lw=1.3)
        axs[3].plot(np.maximum(s["Gamma"], 0).mean(axis=(0, 1)) / ph.RHO_S * YR,
                    z, color=col, lw=1.3)
    lab = np.load(os.path.join(DATA, "background.npz"))["lab"][:, :, 0] / 1e3
    for ax in axs:
        ax.axhspan(lab.min(), lab.max(), color="0.88", zorder=0)
        ax.grid(alpha=0.25, lw=0.4)
    for ax, t in zip(axs, ("(a) |u| (mm/yr)", "(b) vertical velocity (mm/yr)",
                           "(c) horizontal speed (mm/yr)",
                           "(d) melting rate (1/yr)")):
        ax.set_xlabel(t)
    axs[0].set_ylim(400, 0)
    axs[0].set_ylabel("depth (km)")
    axs[2].set_ylabel("depth (km)")
    axs[0].legend(fontsize=6.5, frameon=False)
    axs[3].set_xscale("log")
    fig.suptitle("Grey band: range of LAB depth.  Dashed: maxima / minima",
                 y=1.01, fontsize=9)
    fig.tight_layout()
    save(fig, "fig10_lid_profiles")


ALL = {"1": fig_setting, "2": fig_partition, "3": fig_melt_slices,
       "4": fig_flow_slices, "5": fig_cross_sections, "6": fig_melt_budget,
       "7": fig_profiles, "8": fig_convergence, "9": fig_lid_comparison,
       "10": fig_lid_profiles}

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or list(ALL)
    for w in which:
        ALL[w]()
