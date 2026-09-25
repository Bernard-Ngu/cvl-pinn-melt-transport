"""
Compute every number the manuscript quotes, from the hydrous solutions, and
write them to ../manuscript/cvlnumbers.py.

Nothing in the text is allowed to be typed by hand: if a value is not produced
here it does not go in the paper.
"""
import os
import json
import numpy as np

import fields as fl
import physics as ph
import figures as F
import newfigs as NF
import verify as V

YR = F.YR
OUT = os.path.abspath(os.path.join(F.HERE, "..", "manuscript"))
os.makedirs(OUT, exist_ok=True)

MAIN = "A"          # hydrous, free lithosphere
LIDS = [("A", "free", "none"), ("L", "locked", "locked"), ("S", "slip", "slip")]


def volcano_ratio(tag, depth_km=300.0):
    """Mean upwelling under the volcanic centres divided by the domain mean."""
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    k = F.kidx(s, depth_km)
    w = -np.asarray(s["uz"], dtype=float)[:, :, k] * YR * 1e3
    vals = []
    for _, la, lo in F.CVL:
        if not (lon.min() < lo < lon.max() and lat.min() < la < lat.max()):
            continue
        i = int(np.argmin(abs(lon - lo)))
        j = int(np.argmin(abs(lat - la)))
        vals.append(w[i, j])
    return float(np.mean(vals)) / float(np.abs(w).mean())


def budget(tag):
    """Column-integrated and domain-integrated melt production."""
    s = F.load(tag)
    x, y, z = s["x"], s["y"], s["z"]
    G = np.maximum(np.asarray(s["Gamma"], dtype=float), 0.0)
    mdot = np.trapezoid(G, z, axis=2) / ph.RHO_S            # m/s of melt
    mdot_m_per_Myr = mdot * YR * 1e6
    Mdot = np.trapezoid(np.trapezoid(mdot, y, axis=1), x)   # m3/s
    Mdot_km3_Myr = Mdot * YR * 1e6 / 1e9
    return dict(mdot_peak=float(mdot_m_per_Myr.max()),
                Mdot=float(Mdot_km3_Myr))


def lab_crossing(tag):
    """Fraction of production below the LAB and of the LAB crossed by melt."""
    s = F.load(tag)
    x, y, z = s["x"], s["y"], s["z"]
    d = NF.bg()
    labg = np.asarray(d["lab"][:, :, 0], dtype=float)
    gx = np.linspace(0, fl.LX, labg.shape[0])
    gy = np.linspace(0, fl.LY, labg.shape[1])
    XX, YY = np.meshgrid(x, y, indexing="ij")
    lab = fl.bilinear(labg, gx, gy, XX.ravel(), YY.ravel()).reshape(XX.shape)

    G = np.maximum(np.asarray(s["Gamma"], dtype=float), 0.0)
    Z = np.broadcast_to(z, G.shape)
    below = Z >= lab[:, :, None]
    tot = np.trapezoid(G, z, axis=2)
    bel = np.trapezoid(np.where(below, G, 0.0), z, axis=2)
    frac_below = 100.0 * bel.sum() / max(tot.sum(), 1e-300)

    qz = -np.asarray(s["qz"], dtype=float)          # upward positive
    ql = np.empty(lab.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            ql[i, j] = np.interp(lab[i, j], z, qz[i, j, :])
    up = ql > 0
    return dict(frac_below_lab=round(float(frac_below), 1),
                frac_lab_crossed=round(100.0 * float(up.mean()), 1),
                flux_lab_mean=float(ql[up].mean() * YR),
                flux_lab_max=float(ql.max() * YR))


def melt_geometry(key="F", tkey="T"):
    d = NF.bg()
    z = np.asarray(d["z"], dtype=float)
    Fq = np.asarray(d[key], dtype=float)
    zk = z / 1e3
    hot = [k for k in range(len(zk)) if Fq[:, :, k].max() > 1e-6]
    out = dict(melt_top_km=int(round(zk[hot[0]])),
               melt_base_km=int(round(zk[hot[-1]])),
               F_partition_max_pct=round(100 * float(Fq.max()), 1),
               F_plugin_max_pct=round(100 * float(np.asarray(d["F_plugin"]).max())),
               dT_plugin_max=int(round(float((np.asarray(d["T_plugin"], dtype=float)
                                              - np.asarray(d["T"], dtype=float)).max()))))
    for dep in (100, 150, 200):
        k = int(np.argmin(abs(zk - dep)))
        out["F%d_pct" % dep] = round(100 * float(Fq[:, :, k].max()), 2)
        out["area%d_pct" % dep] = int(round(100 * float((Fq[:, :, k] > 1e-6).mean())))
    T = np.asarray(d[tkey], dtype=float)
    for dep in (250, 300):
        k = int(np.argmin(abs(zk - dep)))
        out["T_max_%d" % dep] = int(round(float(T[:, :, k].max())))
        P = ph.lithostatic_pressure(dep * 1e3)
        out["T_dry_sol_%d" % dep] = int(round(float(ph.solidus(P))))
        out["T_wet_sol_%d" % dep] = int(round(float(ph.hydrous_solidus(P))))
    P100 = ph.lithostatic_pressure(1.0e5)
    out["P_100km_GPa"] = round(float(P100) / 1e9, 2)
    out["T_dry_sol_100"] = int(round(float(ph.solidus(P100))))
    out["T_wet_sol_100"] = int(round(float(ph.hydrous_solidus(P100))))
    out["wet_depression_K"] = int(round(float(ph.solidus(P100)
                                              - ph.hydrous_solidus(P100))))
    out["X_bulk_ppm"] = int(round(ph.X_BULK_H2O * 1e4))
    out["X_melt_wt_pct"] = round(float(min(ph.X_BULK_H2O / ph.D_H2O,
                                           ph.water_saturation(P100))), 1)
    # temperature at the base of the lithosphere vs the wet solidus there
    lab = np.asarray(d["lab"][:, :, 0], dtype=float)
    out["lab_min_km"] = int(round(float(lab.min()) / 1e3))
    out["lab_max_km"] = int(round(float(lab.max()) / 1e3))
    Plab = ph.lithostatic_pressure(lab.min())
    out["T_lab_base"] = int(round(float(ph.T_POTENTIAL
                                        * np.exp(ph.GRAVITY * ph.ALPHA
                                                 * lab.min() / ph.CP))))
    out["T_wet_sol_labmin"] = int(round(float(ph.hydrous_solidus(Plab))))
    out["subsolidus_K"] = out["T_wet_sol_labmin"] - out["T_lab_base"]
    return out


def axis_flow():
    """Along-chain velocity, positive toward the continent, in two layers."""
    out = {}
    ptag, title, a, b = NF.tracks()[0]
    for tag, name, _ in LIDS:
        s = F.load(tag)
        d = NF.sample_track(s, a, b)
        up = d["u_par"] * YR * 1e3
        zk = np.asarray(s["z"]) / 1e3
        Z = np.broadcast_to(zk, up.shape)
        L = np.repeat(d["lab"][:, None], len(zk), 1)
        sub = (Z >= L) & (Z <= L + 60)
        deep = (Z >= 250) & (Z <= 400)
        out[name] = dict(sub=round(float(up[sub].mean()), 3),
                         deep=round(float(up[deep].mean()), 3))
    return out


def melt_ratio(tag):
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    G = np.maximum(np.asarray(s["Gamma"], dtype=float), 0.0)
    md = np.trapezoid(G, s["z"], axis=2)
    vals = [md[int(np.argmin(abs(lon - lo))), int(np.argmin(abs(lat - la)))]
            for _, la, lo in F.CVL]
    return round(float(np.mean(vals) / md.mean()), 1)


def residuals():
    out = {}
    for tag, name, lid in LIDS:
        t = V._terms(tag, 1e-7, n=12000, lid=lid)
        sp = t["speed"] * YR * 1e3
        s = F.load(tag)
        k300 = F.kidx(s, 300.0)
        w300 = float((-np.asarray(s["uz"], dtype=float)[:, :, k300]
                      * YR * 1e3).max())
        b = budget(tag)
        out[name] = dict(
            mom=round(float(np.sqrt((t["Rmom"] ** 2).mean())), 3),
            rheo=round(float(np.sqrt((t["Rrheo"] ** 2).mean())), 3),
            solid=round(float(V.relative(t["divu"], t["gradu"])), 2),
            umed=round(float(np.median(sp)), 3),
            umax=round(float(sp.max()), 2),
            w300=round(w300, 2),
            Mdot=int(round(b["Mdot"])),
            work=round(V.buoyancy_work(tag), 2),
            wratio=round(volcano_ratio(tag), 1))
    return out


def main():
    n = {}
    n.update(melt_geometry())
    n["delam"] = NF.body_stats()
    n["lid"] = residuals()
    n["axis_flow"] = axis_flow()
    n["mdot_ratio"] = melt_ratio(MAIN)
    n["budget_A"] = budget(MAIN)
    n.update(lab_crossing(MAIN))
    s = F.load(MAIN)
    w = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    n["w_up_max"] = round(float(w.max()), 2)
    n["Mdot_A"] = n["lid"]["free"]["Mdot"]
    n["mdot_peak"] = round(n["budget_A"]["mdot_peak"], 1)
    n["phi_max"] = float(np.asarray(s["phi"]).max())

    n.update(deep_anomaly())
    n["body"] = body_suppression(MAIN)
    n["arms"] = arms(MAIN)
    n["notch"] = delam_notch(MAIN)

    with open(os.path.join(OUT, "numbers.json"), "w") as f:
        json.dump(n, f, indent=1, sort_keys=True)
    print(json.dumps(n, indent=1, sort_keys=True))




# --------------------------------------------------------- added diagnostics
def deep_anomaly():
    """Depth extent of the thermal anomaly relative to the adiabat."""
    d = NF.bg()
    z = np.asarray(d["z"], dtype=float)
    zk = z / 1e3
    T = np.asarray(d["T"], dtype=float)
    dT = T - ph.adiabat(z)[None, None, :]
    lon, lat = NF.latlon_grid(dT[:, :, 0])
    LO, LA = np.meshgrid(lon, lat, indexing="ij")
    out = {}
    for dep in (200, 300, 400, 500, 600):
        k = int(np.argmin(abs(zk - dep)))
        out["dT_max_%d" % dep] = int(round(float(dT[:, :, k].max())))
        out["area_dT50_%d" % dep] = round(100 * float((dT[:, :, k] > 50).mean()), 1)
    k4 = int(np.argmin(abs(zk - 400)))
    m = dT[:, :, k4] > 50
    out["deep_lon_c"] = round(float(LO[m].mean()), 1)
    out["deep_lat_c"] = round(float(LA[m].mean()), 1)
    return out


def body_suppression(tag="A"):
    """Vertical velocity inside and outside the fast body, below the LAB."""
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    LO, LA = np.meshgrid(lon, lat, indexing="ij")
    w = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    zk = np.asarray(s["z"]) / 1e3
    d = NF.bg()
    labg = np.asarray(d["lab"][:, :, 0], dtype=float)
    gx = np.linspace(0, fl.LX, labg.shape[0])
    gy = np.linspace(0, fl.LY, labg.shape[1])
    XX, YY = np.meshgrid(s["x"], s["y"], indexing="ij")
    lab = fl.bilinear(labg, gx, gy, XX.ravel(), YY.ravel()).reshape(XX.shape) / 1e3
    blon, blat, bm, _, _ = NF.fast_body_mask()
    ii = np.clip(np.searchsorted(blon, lon) - 1, 0, len(blon) - 1)
    jj = np.clip(np.searchsorted(blat, lat) - 1, 0, len(blat) - 1)
    mi = bm[np.ix_(ii, jj)]
    Z = np.broadcast_to(zk, w.shape)
    band = (Z >= lab[:, :, None]) & (Z <= lab[:, :, None] + 120)
    inb = mi[:, :, None] & band
    out = ((~mi[:, :, None]) & band
           & ((LO > 8) & (LO < 16) & (LA > 0) & (LA < 8))[:, :, None])
    return dict(w_in=round(float(w[inb].mean()), 3),
                w_out=round(float(w[out].mean()), 3),
                down_in=round(100 * float((w[inb] < 0).mean()), 1),
                down_out=round(100 * float((w[out] < 0).mean()), 1),
                supp_pct=int(round(100 * (1 - w[inb].mean() / w[out].mean()))))


ARMS = {"south": (3.0, 11.5, -4.0, 7.0),
        "adamawa": (12.0, 17.0, 5.5, 9.0),
        "biu": (10.0, 15.0, 9.0, 12.5)}


def arms(tag="A"):
    s = F.load(tag)
    lon, lat = F.lonlat_axes(s)
    LO, LA = np.meshgrid(lon, lat, indexing="ij")
    G = np.maximum(np.asarray(s["Gamma"], dtype=float), 0.0)
    md = np.trapezoid(G, s["z"], axis=2)
    tot = md.sum()
    w = -np.asarray(s["uz"], dtype=float) * YR * 1e3
    k = F.kidx(s, 300.0)
    i, j = np.unravel_index(np.argmax(w[:, :, k]), w[:, :, k].shape)
    out = dict(centre_lon=round(float(lon[i]), 1),
               centre_lat=round(float(lat[j]), 1))
    for nm, (l0, l1, b0, b1) in ARMS.items():
        m = (LO >= l0) & (LO <= l1) & (LA >= b0) & (LA <= b1)
        out["melt_%s_pct" % nm] = round(100 * float(md[m].sum() / tot))
    return out


def delam_notch(tag="A", dmin=1580.0, dmax=1850.0, zcap=200.0):
    """The melt-free block outlined on the along-chain vertical-velocity
    panel: its extent, and how it differs from the mantle on either side."""
    s = F.load(tag)
    tks = NF.tracks()
    a, b = tks[0][2], tks[0][3]
    d = NF.sample_track(s, a, b)
    zk = np.asarray(s["z"], dtype=float) / 1e3
    dist = d["dist"]
    nn = len(dist)
    tt = np.linspace(0.0, 1.0, nn)
    px = a[0] + (b[0] - a[0]) * tt
    py = a[1] + (b[1] - a[1]) * tt
    lo, la = fl.xy_to_lonlat(px - fl.LX / 2, py - fl.LY / 2)

    bgd = NF.bg()
    zb = np.asarray(bgd["z"], dtype=float)
    X = np.repeat(px[:, None], len(zb), 1).ravel()
    Y = np.repeat(py[:, None], len(zb), 1).ravel()
    Z = np.tile(zb, nn)
    DL = fl.trilinear(np.asarray(bgd["dlnvs"], dtype=float), bgd["x"],
                      bgd["y"], zb, X, Y, Z).reshape(nn, len(zb)) * 100
    TT = fl.trilinear(np.asarray(bgd["T"], dtype=float), bgd["x"], bgd["y"],
                      zb, X, Y, Z).reshape(nn, len(zb))
    zbk = zb / 1e3

    Fm = np.asarray(d["F"], dtype=float)
    lab = np.asarray(d["lab"], dtype=float)
    base = np.array([zk[Fm[i] > 1e-5].max() if (Fm[i] > 1e-5).any() else np.nan
                     for i in range(nn)])

    core = (dist > dmin) & (dist < dmax)
    side = (((dist > dmin - 260) & (dist < dmin))
            | ((dist > dmax) & (dist < dmax + 260)))
    band = (zbk > 100.0) & (zbk < zcap - 15.0)
    return dict(d0=int(round(float(dist[core].min()))),
                d1=int(round(float(dist[core].max()))),
                lon0=round(float(lo[core].min()), 1),
                lon1=round(float(lo[core].max()), 1),
                lat0=round(float(la[core].min()), 1),
                lat1=round(float(la[core].max()), 1),
                ztop=int(round(float(np.nanmin(base[core])))),
                zbot=int(round(zcap)),
                lab=int(round(float(lab[core].mean()))),
                dlnvs_in=round(float(DL[np.ix_(core, band)].mean()), 2),
                dlnvs_out=round(float(DL[np.ix_(side, band)].mean()), 2),
                T_in=int(round(float(TT[np.ix_(core, band)].mean()))),
                T_out=int(round(float(TT[np.ix_(side, band)].mean()))),
                gap_min=int(round(float(np.nanmin(base[core] - lab[core])))))


if __name__ == "__main__":
    main()
