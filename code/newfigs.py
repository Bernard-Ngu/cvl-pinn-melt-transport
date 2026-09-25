"""
Figures for the hydrous CVL PINN manuscript.

The main model is the FREE LITHOSPHERE run (tag A).  The locked and slip lid
runs are used only for the sensitivity comparison.

New relative to the earlier figure set:
  * vertical sections show temperature and VERTICAL VELOCITY, not porosity
  * oblique (diagonal) sections along the chain axis and across it
  * depth slices of melt fraction and porosity for the main model
  * the fast body beneath the thin continental lithosphere is outlined and
    labelled on the setting figure
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import (gaussian_filter, binary_opening, binary_closing,
                           binary_fill_holes, label)

import fields as fl
import physics as ph
import figures as F

OUT = os.path.abspath(os.path.join(F.HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)
YR = F.YR
MAIN = "A"                      # hydrous, free lithosphere


def configure(main=None, out=None):
    """Choose which run the figures show and where they are written."""
    global MAIN, OUT
    if main is not None:
        MAIN = main
    if out is not None:
        OUT = os.path.abspath(out)
        os.makedirs(OUT, exist_ok=True)
        F.FIGS = OUT
PLIM_PCT = 1.0e-1               # top of the porosity colour scale, per cent

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "figure.dpi": 160, "savefig.dpi": 300,
                     "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


# --------------------------------------------------------------- background
_BG = {}


def bg():
    if "bg" not in _BG:
        _BG["bg"] = np.load(os.path.join(F.DATA, "background.npz"))
    return _BG["bg"]


def latlon_grid(a):
    lon = np.linspace(ph.LON_MIN, ph.LON_MAX, a.shape[0])
    lat = np.linspace(ph.LAT_MIN, ph.LAT_MAX, a.shape[1])
    return lon, lat


# ---------------------------------------------------- the diagonal profiles
_ANN = (-1.437, 5.633)
_OKU = (6.200, 10.519)
_SLOPE = (_OKU[0] - _ANN[0]) / (_OKU[1] - _ANN[1])
MT_CAMEROON = (4.203, 9.170)


def _xy(lat, lon):
    x, y = fl.lonlat_to_xy(lon, lat)
    return x + fl.LX / 2.0, y + fl.LY / 2.0


def _clip(p0, t, pad=1.0e3):
    lo, hi = -1e9, 1e9
    for c, L in ((0, fl.LX), (1, fl.LY)):
        if abs(t[c]) < 1e-12:
            continue
        a = (pad - p0[c]) / t[c]
        b = (L - pad - p0[c]) / t[c]
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return p0 + lo * t, p0 + hi * t


def tracks():
    a = np.array(_xy(ph.LAT_MIN, _ANN[1] + (ph.LAT_MIN - _ANN[0]) / _SLOPE))
    b = np.array(_xy(ph.LAT_MAX, _ANN[1] + (ph.LAT_MAX - _ANN[0]) / _SLOPE))
    t1 = (b - a) / np.linalg.norm(b - a)
    a1, b1 = _clip(a, t1)
    t2 = np.array([t1[1], -t1[0]])
    if t2[0] > 0 or t2[1] < 0:
        t2 = -t2
    a2, b2 = _clip(np.array(_xy(*MT_CAMEROON)), t2)
    if np.dot(b2 - a2, t2) < 0:
        a2, b2 = b2, a2
    return [("P1", "CVL axis, SW (oceanic) to NE (continental)", a1, b1),
            ("P2", "Congo craton (SE) to Mount Cameroon and NW", a2, b2)]


def sample_track(s, a, b, n=340):
    tt = np.linspace(0.0, 1.0, n)
    px = a[0] + (b[0] - a[0]) * tt
    py = a[1] + (b[1] - a[1]) * tt
    dist = np.hypot(px - a[0], py - a[1]) / 1e3
    z = s["z"]
    X = np.repeat(px[:, None], len(z), axis=1).ravel()
    Y = np.repeat(py[:, None], len(z), axis=1).ravel()
    Z = np.tile(z, len(px))
    out = {}
    for k in ("T", "F", "phi", "ux", "uy", "uz"):
        out[k] = fl.trilinear(np.asarray(s[k], dtype=float), s["x"], s["y"], z,
                              X, Y, Z).reshape(len(px), len(z))
    tv = np.array([b[0] - a[0], b[1] - a[1]])
    tv /= np.linalg.norm(tv)
    out["u_par"] = out["ux"] * tv[0] + out["uy"] * tv[1]
    out["w"] = -out["uz"]
    out["dist"] = dist
    lab = np.asarray(bg()["lab"][:, :, 0], dtype=float)
    gx = np.linspace(0, fl.LX, lab.shape[0])
    gy = np.linspace(0, fl.LY, lab.shape[1])
    out["lab"] = fl.bilinear(lab, gx, gy, px, py) / 1e3
    return out


def project_volcanoes(a, b, max_off=60.0):
    t = np.array([b[0] - a[0], b[1] - a[1]])
    t /= np.linalg.norm(t)
    hits = []
    for name, la, lo in F.CVL:
        p = np.array(_xy(la, lo))
        d = p - np.array(a)
        s_along = float(np.dot(d, t)) / 1e3
        off = float(np.linalg.norm(d - np.dot(d, t) * t)) / 1e3
        if off <= max_off and 0 <= s_along <= np.hypot(*(np.array(b) - a)) / 1e3:
            hits.append((name, s_along, off))
    return sorted(hits, key=lambda h: h[1])


# ------------------------------------------------- the fast (detached) body
def fast_body_mask(vs_cut=2.0, lab_cut=140.0, depth_km=100.0):
    """dlnVs above vs_cut where the LAB is shallower than lab_cut, largest
    connected component, restricted to the continental corridor."""
    d = bg()
    k = int(np.argmin(abs(d["z"] - depth_km * 1e3)))
    vs = 100.0 * np.asarray(d["dlnvs"][:, :, k], dtype=float)
    lab = np.asarray(d["lab"][:, :, 0], dtype=float) / 1e3
    lon, lat = latlon_grid(vs)
    LO, LA = np.meshgrid(lon, lat, indexing="ij")
    m = ((vs > vs_cut) & (lab < lab_cut) & (LA > 0.0) & (LA < 7.0)
         & (LO > 8.0) & (LO < 15.5))
    m = binary_closing(binary_opening(m, np.ones((3, 3))), np.ones((7, 7)))
    lb, nl = label(m)
    if nl:
        sizes = [(lb == i).sum() for i in range(1, nl + 1)]
        m = lb == (1 + int(np.argmax(sizes)))
    m = binary_fill_holes(m)
    return lon, lat, m, vs, lab


def body_stats():
    lon, lat, m, vs, lab = fast_body_mask()
    LO, LA = np.meshgrid(lon, lat, indexing="ij")
    keel = lab > 200.0
    axis = lab < 100.0
    return dict(area_pct=round(100.0 * m.mean(), 1),
                dlnvs_mean=round(float(vs[m].mean()), 1),
                dlnvs_max=round(float(vs[m].max()), 1),
                lab_mean=int(round(float(lab[m].mean()))),
                lab_lo=int(round(float(lab[m].min()))),
                lab_hi=int(round(float(lab[m].max()))),
                lon_lo=round(float(LO[m].min()), 1),
                lon_hi=round(float(LO[m].max()), 1),
                lat_lo=round(float(LA[m].min()), 1),
                lat_hi=round(float(LA[m].max()), 1),
                lon_c=round(float(LO[m].mean()), 1),
                lat_c=round(float(LA[m].mean()), 1),
                keel_dlnvs=round(float(vs[keel].mean()), 1),
                keel_lab=int(round(float(lab[keel].mean()))),
                axis_dlnvs=round(float(vs[axis].mean()), 1))


# ============================================================ Fig 1 setting
def fig_setting():
    d = bg()
    lab = np.asarray(d["lab"][:, :, 0], dtype=float) / 1e3
    lonb, latb = latlon_grid(lab)
    k100 = int(np.argmin(abs(d["z"] - 1.0e5)))
    vs = 100.0 * np.asarray(d["dlnvs"][:, :, k100], dtype=float)

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.4))

    ax = axs[0]
    im = ax.pcolormesh(lonb, latb, lab.T, cmap="viridis_r", shading="auto")
    plt.colorbar(im, ax=ax, label="LAB depth (km)", fraction=0.046)
    for tag, _, a, b in tracks():
        lo0, la0 = fl.xy_to_lonlat(a[0] - fl.LX / 2, a[1] - fl.LY / 2)
        lo1, la1 = fl.xy_to_lonlat(b[0] - fl.LX / 2, b[1] - fl.LY / 2)
        ax.plot([lo0, lo1], [la0, la1], "-", color="crimson", lw=1.6, zorder=7)
        ax.text(lo0 + 0.08 * (lo1 - lo0), la0 + 0.08 * (la1 - la0), tag,
                fontsize=7, color="crimson", ha="center", va="center", zorder=8,
                bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.0))
    F.draw_cvl(ax, labels=True, ms=3.6)
    F.basemap(ax, lonb, latb)
    ax.set_title("(a) lithosphere-asthenosphere boundary")

    ax = axs[1]
    im = ax.pcolormesh(lonb, latb, vs.T, cmap="RdBu_r", shading="auto",
                       vmin=-6, vmax=6)
    plt.colorbar(im, ax=ax, label="dlnVs (%)", fraction=0.046)
    F.draw_cvl(ax, ms=3.6)
    F.basemap(ax, lonb, latb)
    ax.set_title("(b) shear-velocity anomaly, 100 km")

    ax = axs[2]
    z = np.asarray(d["z"], dtype=float)
    zb = z / 1e3
    P = ph.lithostatic_pressure(z)
    T = np.asarray(d["T"], dtype=float)
    ax.fill_betweenx(zb, T.min(axis=(0, 1)), T.max(axis=(0, 1)),
                     color="0.85", lw=0)
    ax.plot(T.mean(axis=(0, 1)), zb, "k-", lw=1.5, label="mean T (this study)")
    ax.plot(np.asarray(d["T_plugin"], dtype=float).mean(axis=(0, 1)), zb,
            "k--", lw=1.1, label="mean T (linear conversion)")
    ax.plot(ph.adiabat(z), zb, color="0.5", lw=0.9, label="adiabat")
    ax.plot(ph.solidus(P), zb, "r-", lw=1.1, label="dry solidus")
    ax.plot(ph.hydrous_solidus(P), zb, color="royalblue", lw=1.3,
            label="wet solidus (%d ppm)" % round(ph.X_BULK_H2O * 1e4))
    ax.set_ylim(400, 0)
    ax.set_xlim(600, 2300)
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("depth (km)")
    ax.legend(fontsize=5.6, loc="lower left", frameon=False)
    ax.set_title("(c) thermal structure and solidi")

    fig.tight_layout()
    F.save(fig, "fig1_setting", OUT)


# ======================================================== Fig 2 the partition
def fig_partition():
    depths = [80, 120, 180, 250]
    dl = np.linspace(-5.5, 2.0, 260) / 100.0
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(depths)))
    for c, dkm in zip(cols, depths):
        dep = np.full_like(dl, dkm * 1e3)
        dT, Fq = ph.invert_temperature_melt(dl, dep)
        axs[0].plot(100 * dl, dT, "-", color=c, lw=1.3, label="%d km" % dkm)
        axs[0].plot(100 * dl, -ph.VS_TO_DENSITY / ph.ALPHA * dl, ":",
                    color=c, lw=1.0)
        axs[1].plot(100 * dl, 100 * Fq, "-", color=c, lw=1.3,
                    label="%d km" % dkm)
    axs[0].set_xlabel("dlnVs (%)")
    axs[0].set_ylabel("temperature excess (K)")
    axs[0].set_title("(a) temperature excess\nsolid: partition, dotted: linear")
    axs[0].legend(fontsize=6, frameon=False)
    axs[1].set_xlabel("dlnVs (%)")
    axs[1].set_ylabel("equilibrium melt fraction (%)")
    axs[1].set_title("(b) equilibrium melt fraction")
    axs[1].legend(fontsize=6, frameon=False)
    for ax in axs:
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    F.save(fig, "fig2_vs_partition", OUT)


# ======================================================== Fig 4 flow slices
def fig_flow_slices(tag=None):
    tag = tag or MAIN
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    depths = [100, 150, 250, 300]
    w = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    ux = np.asarray(s["ux"], dtype=float) * YR * 1e3
    uy = np.asarray(s["uy"], dtype=float) * YR * 1e3
    vmax = float(np.abs(w).max())
    umax = float(np.hypot(ux, uy).max())
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax, dep in zip(axs.ravel(), depths):
        k = F.kidx(s, dep)
        im = ax.pcolormesh(lon, lat, w[:, :, k].T, cmap="RdBu_r",
                           shading="auto", vmin=-vmax, vmax=vmax)
        st = max(1, len(lon) // 22)
        ax.quiver(lon[::st], lat[::st], ux[::st, ::st, k].T,
                  uy[::st, ::st, k].T, scale=umax * 22, width=0.004,
                  color="0.15")
        ax.contour(lon, lat, np.asarray(s["F"])[:, :, k].T, levels=[1e-5],
                   colors="limegreen", linewidths=0.9)
        plt.colorbar(im, ax=ax, fraction=0.046, label="vertical velocity (mm/yr)")
        F.draw_cvl(ax, ms=3.4)
        F.basemap(ax, lon, lat)
        ax.set_title("(%s) %d km" % ("abcd"[depths.index(dep)], dep))
    fig.suptitle("Solid flow, %s   (arrows: horizontal velocity, max %.2f mm/yr)"
                 % (F.LABELS[tag], umax), y=0.995, fontsize=9)
    fig.tight_layout()
    F.save(fig, "fig4_flow_slices", OUT)


# ============================ Fig 5 orthogonal sections: T and vertical flow
def _section_indices(s, lat0=4.2, lon0=10.0):
    lon, lat = F.lonlat_axes(s)
    return int(np.argmin(abs(lat - lat0))), int(np.argmin(abs(lon - lon0))), lon, lat


def fig_cross_sections(tag=None, zmax=400.0):
    tag = tag or MAIN
    s = F.load(tag)
    j, i, lon, lat = _section_indices(s)
    zk = np.asarray(s["z"]) / 1e3
    T = np.asarray(s["T"], dtype=float)
    w = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    labg = np.asarray(bg()["lab"][:, :, 0], dtype=float) / 1e3
    lonb, latb = latlon_grid(labg)
    wmax = float(np.abs(w).max())

    fig, axs = plt.subplots(2, 2, figsize=(10.2, 6.4))
    rows = [("W-E section at %.1f N through Mount Cameroon" % lat[j],
             lon, T[:, j, :], w[:, j, :],
             np.interp(lon, lonb, labg[:, int(np.argmin(abs(latb - lat[j])))]),
             "longitude (E)"),
            ("S-N section at %.1f E along the continental chain" % lon[i],
             lat, T[i, :, :], w[i, :, :],
             np.interp(lat, latb, labg[int(np.argmin(abs(lonb - lon[i]))), :]),
             "latitude (N)")]
    for r, (title, ax_h, Ts, ws, labline, xlab) in enumerate(rows):
        ax = axs[r, 0]
        im = ax.pcolormesh(ax_h, zk, Ts.T, cmap="inferno", shading="auto",
                           vmin=600, vmax=1950)
        ax.contour(ax_h, zk, Ts.T, levels=[1400, 1600, 1700], colors="w",
                   linewidths=0.5)
        plt.colorbar(im, ax=ax, fraction=0.03, label="T (K)")
        ax.set_title("(%s) %s\ntemperature" % ("ac"[r], title), pad=6)

        ax2 = axs[r, 1]
        im2 = ax2.pcolormesh(ax_h, zk, ws.T, cmap="RdBu_r", shading="auto",
                             vmin=-wmax, vmax=wmax)
        st = max(1, len(ax_h) // 30)
        ax2.quiver(ax_h[::st], zk[::4], np.zeros_like(ws[::st, ::4].T),
                   ws[::st, ::4].T, scale=wmax * 26, width=0.003,
                   color="0.15", alpha=0.9)
        ax2.contour(ax_h, zk, ws.T, levels=[0.0], colors="0.4",
                    linewidths=0.4)
        plt.colorbar(im2, ax=ax2, fraction=0.03, label="vertical velocity (mm/yr)")
        ax2.set_title("(%s) %s\nvertical velocity" % ("bd"[r], title), pad=6)

        for a_ in (ax, ax2):
            a_.plot(ax_h, labline, "-", color="w" if a_ is ax else "k", lw=1.2)
            a_.set_ylim(zmax, 0)
            a_.set_xlim(ax_h.min(), ax_h.max())
            a_.set_xlabel(xlab)
            a_.set_ylabel("depth (km)")
    fig.suptitle("Orthogonal vertical sections, %s" % F.LABELS[tag],
                 y=1.01, fontsize=9)
    fig.tight_layout()
    F.save(fig, "fig5_cross_sections", OUT)


# ================== Fig 6 oblique sections: temperature and vertical flow
def _delam_patch(d, zk, dmin=1580.0, dmax=1850.0, zcap=200.0):
    """Melt-free block hanging beneath the lid on the continental sector of P1.

    The block is defined by the absence of melt between the LAB and 200 km
    inside the distance window in which the melting envelope pinches up to
    the lid; it is the outline drawn on the vertical-velocity panel.
    """
    D, Z = np.meshgrid(d["dist"], zk, indexing="ij")
    LAB = np.repeat(np.asarray(d["lab"])[:, None], len(zk), axis=1)
    m = ((np.asarray(d["F"]) < 1e-5) & (Z > LAB + 5.0) & (Z < zcap)
         & (D > dmin) & (D < dmax))
    return m.astype(float)


def fig_oblique_sections(tag=None, zmax=400.0):
    tag = tag or MAIN
    s = F.load(tag)
    tks = tracks()
    zk = np.asarray(s["z"]) / 1e3
    wall = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    wmax = float(np.abs(wall).max())
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6))
    for r, (ptag, title, a, b) in enumerate(tks):
        d = sample_track(s, a, b)
        vol = project_volcanoes(a, b)
        dist = d["dist"]

        ax = axs[r, 0]
        im = ax.pcolormesh(dist, zk, d["T"].T, cmap="inferno", shading="auto",
                           vmin=600, vmax=1950)
        ax.contour(dist, zk, d["T"].T, levels=[1400, 1600, 1700], colors="w",
                   linewidths=0.5)
        st = max(1, len(dist) // 34)
        up = d["u_par"] * YR * 1e3
        wp = d["w"] * YR * 1e3
        sc = max(np.abs(up).max(), np.abs(wp).max(), 1e-9)
        ax.quiver(dist[::st], zk[::4], up[::st, ::4].T, wp[::st, ::4].T,
                  scale=sc * 26, width=0.003, color="w", alpha=0.9)
        plt.colorbar(im, ax=ax, fraction=0.03, label="T (K)")
        ax.set_title("(%s) %s\ntemperature and in-plane flow" % ("ac"[r], title),
                     pad=6)

        ax2 = axs[r, 1]
        im2 = ax2.pcolormesh(dist, zk, wp.T, cmap="RdBu_r", shading="auto",
                             vmin=-wmax, vmax=wmax)
        # arrows follow the direction of the velocity in the plane of the
        # section; the colour behind them remains the vertical component
        ax2.quiver(dist[::st], zk[::4], up[::st, ::4].T, wp[::st, ::4].T,
                   scale=sc * 26, width=0.003, color="0.15", alpha=0.9)
        ax2.contour(dist, zk, d["F"].T, levels=[1e-5], colors="limegreen",
                    linewidths=0.9)
        if r == 0:
            dm = _delam_patch(d, zk)
            ax2.contour(dist, zk, dm.T, levels=[0.5], colors="k",
                        linestyles="--", linewidths=1.3, zorder=8)
            ax2.text(1715.0, 207.0, "delaminated body", fontsize=6.0,
                     color="k", ha="center", va="top", zorder=10,
                     bbox=dict(fc="w", ec="none", alpha=0.75,
                               pad=1.0, boxstyle="round,pad=0.18"))
        plt.colorbar(im2, ax=ax2, fraction=0.03, label="vertical velocity (mm/yr)")
        ax2.set_title("(%s) %s\nvertical velocity" % ("bd"[r], title), pad=6)

        for a_, lc in ((ax, "w"), (ax2, "k")):
            a_.plot(dist, d["lab"], "-", color=lc, lw=1.2)
            a_.set_ylim(zmax, 0)
            a_.set_xlim(dist.min(), dist.max())
            a_.set_xlabel("distance along %s (km)" % ptag)
            a_.set_ylabel("depth (km)")
            for n_, (name, sd, off) in enumerate(vol):
                a_.plot(sd, 0, "v", ms=4.5, mfc="w", mec="k", mew=0.7,
                        clip_on=False, zorder=9)
                a_.text(sd, 0.03 * zmax + (n_ % 2) * 0.15 * zmax, name,
                        fontsize=4.8, rotation=-90, color=lc, ha="center",
                        va="top", zorder=9)
    fig.suptitle("Oblique vertical sections, %s" % F.LABELS[tag], y=1.01,
                 fontsize=9)
    fig.tight_layout()
    F.save(fig, "fig6_oblique_sections", OUT)


# ======================== Fig 9 depth slices of melt fraction and porosity
def fig_melt_porosity_slices(tag=None, depths=(100, 150, 200)):
    tag = tag or MAIN
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    fig, axs = plt.subplots(2, len(depths), figsize=(3.5 * len(depths), 6.0))
    for c, dep in enumerate(depths):
        k = F.kidx(s, dep)
        ax = axs[0, c]
        f = np.maximum(np.asarray(s["F"], dtype=float)[:, :, k] * 100, 1e-6)
        im0 = ax.pcolormesh(lon, lat, f.T, cmap="magma", shading="auto",
                            norm=LogNorm(3e-2, 2.0))
        F.draw_cvl(ax, ms=3.2)
        F.basemap(ax, lon, lat)
        ax.set_title("(%s) melt fraction, %d km" % ("abc"[c], dep))
        ax.set_xlabel("")
        ax2 = axs[1, c]
        p = np.maximum(np.asarray(s["phi"], dtype=float)[:, :, k] * 100, 1e-6)
        im1 = ax2.pcolormesh(lon, lat, p.T, cmap="viridis", shading="auto",
                             norm=LogNorm(1e-5, PLIM_PCT))
        ax2.contour(lon, lat, np.asarray(s["F"])[:, :, k].T, levels=[1e-5],
                    colors="w", linewidths=0.7)
        F.draw_cvl(ax2, ms=3.2)
        F.basemap(ax2, lon, lat)
        ax2.set_title("(%s) porosity, %d km" % ("def"[c], dep))
        if c:
            ax.set_ylabel("")
            ax2.set_ylabel("")
    fig.colorbar(im0, ax=axs[0, :], fraction=0.022, pad=0.01,
                 label="melt fraction (%)")
    fig.colorbar(im1, ax=axs[1, :], fraction=0.022, pad=0.01,
                 label="porosity (%)")
    F.save(fig, "fig3_melt_porosity_slices", OUT)


# ============ Fig 10 oblique sections of melt fraction and porosity
def fig_oblique_melt(tag=None, zmax=400.0):
    tag = tag or MAIN
    s = F.load(tag)
    zk = np.asarray(s["z"]) / 1e3
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6))
    for r, (ptag, title, a, b) in enumerate(tracks()):
        d = sample_track(s, a, b)
        vol = project_volcanoes(a, b)
        dist = d["dist"]
        ax = axs[r, 0]
        im0 = ax.pcolormesh(dist, zk, np.maximum(d["F"] * 100, 1e-6).T,
                            cmap="magma", shading="auto",
                            norm=LogNorm(3e-2, 2.0))
        plt.colorbar(im0, ax=ax, fraction=0.03, label="melt fraction (%)")
        ax.set_title("(%s) %s\nequilibrium melt fraction" % ("ac"[r], title),
                     pad=6)
        ax2 = axs[r, 1]
        im1 = ax2.pcolormesh(dist, zk, np.maximum(d["phi"] * 100, 1e-6).T,
                             cmap="viridis", shading="auto",
                             norm=LogNorm(1e-5, PLIM_PCT))
        ax2.contour(dist, zk, d["F"].T, levels=[1e-5], colors="w",
                    linewidths=0.8)
        plt.colorbar(im1, ax=ax2, fraction=0.03, label="porosity (%)")
        ax2.set_title("(%s) %s\nretained porosity" % ("bd"[r], title), pad=6)
        for a_ in (ax, ax2):
            a_.plot(dist, d["lab"], "w-", lw=1.2)
            a_.set_ylim(zmax, 0)
            a_.set_xlim(dist.min(), dist.max())
            a_.set_xlabel("distance along %s (km)" % ptag)
            a_.set_ylabel("depth (km)")
            for n_, (name, sd, off) in enumerate(vol):
                a_.plot(sd, 0, "v", ms=4.5, mfc="w", mec="k", mew=0.7,
                        clip_on=False, zorder=9)
                a_.text(sd, 0.03 * zmax + (n_ % 2) * 0.15 * zmax, name,
                        fontsize=4.8, rotation=-90, color="w", ha="center",
                        va="top", zorder=9)
    fig.suptitle("Melt along the oblique sections, %s" % F.LABELS[tag],
                 y=1.01, fontsize=9)
    fig.tight_layout()
    F.save(fig, "fig9_oblique_melt", OUT)


# ============================== Figs 7, 8, 11, 12, 13 via the shared module
def _ren(a, b):
    for e in ("png", "pdf"):
        src = os.path.join(OUT, "%s.%s" % (a, e))
        if os.path.exists(src):
            os.replace(src, os.path.join(OUT, "%s.%s" % (b, e)))


def fig_budget():
    F.fig_melt_budget(tags=(MAIN,))
    _ren("fig6_melt_budget", "fig7_melt_budget")


def fig_profiles():
    F.fig_profiles(tags=(MAIN,))
    _ren("fig7_profiles", "fig8_profiles")


def fig_lids():
    F.fig_lid_comparison()
    F.fig_lid_profiles()
    F.fig_convergence(tags=("A", "L", "S"))
    _ren("fig9_lid_comparison", "fig10_lid_comparison")
    _ren("fig10_lid_profiles", "fig11_lid_profiles")
    _ren("fig8_convergence", "fig12_convergence")
