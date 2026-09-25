"""
Background (data-derived) fields for the CVL PINN.

Builds, on a regular Cartesian working grid:
    Vs        interpolated shear-wave speed
    dlnVs     relative anomaly w.r.t. the 1-D model average
    T         temperature from potential_temperature.cc
    F_eq      Katz (2003) equilibrium melt fraction
    grad F_eq melt productivity gradient (drives the melting source term)

The spherical chunk (lon 2.5-18 E, lat -4 to 12.5 N, 0-660 km) is mapped to a
locally Cartesian domain by an equirectangular projection about the domain
centre.  Over this aperture the induced metric distortion is < 2 %.
"""
import os
import numpy as np
import physics as ph

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

LON0 = 0.5 * (ph.LON_MIN + ph.LON_MAX)
LAT0 = 0.5 * (ph.LAT_MIN + ph.LAT_MAX)
COSLAT0 = np.cos(np.radians(LAT0))

LX = ph.R_OUTER * np.radians(ph.LON_MAX - ph.LON_MIN) * COSLAT0
LY = ph.R_OUTER * np.radians(ph.LAT_MAX - ph.LAT_MIN)
LZ = ph.D_MAX


def lonlat_to_xy(lon, lat):
    x = ph.R_OUTER * np.radians(lon - LON0) * COSLAT0
    y = ph.R_OUTER * np.radians(lat - LAT0)
    return x, y


def xy_to_lonlat(x, y):
    lon = LON0 + np.degrees(x / (ph.R_OUTER * COSLAT0))
    lat = LAT0 + np.degrees(y / ph.R_OUTER)
    return lon, lat


# ------------------------------------------------------------ interpolation
def _interp1_weights(q, axis_vals):
    n = len(axis_vals)
    i = np.clip(np.searchsorted(axis_vals, q) - 1, 0, n - 2)
    t = (q - axis_vals[i]) / (axis_vals[i + 1] - axis_vals[i])
    return i, np.clip(t, 0.0, 1.0)


def trilinear(grid, ax, ay, az, qx, qy, qz):
    """grid[ix, iy, iz] sampled at scattered (qx, qy, qz)."""
    i, u = _interp1_weights(qx, ax)
    j, v = _interp1_weights(qy, ay)
    k, w = _interp1_weights(qz, az)
    out = 0.0
    for di, wu in ((0, 1 - u), (1, u)):
        for dj, wv in ((0, 1 - v), (1, v)):
            for dk, ww in ((0, 1 - w), (1, w)):
                out = out + grid[i + di, j + dj, k + dk] * (wu * wv * ww)
    return out


def _boxcar(a, w, axis):
    """Moving average of width w along `axis`, with edge replication."""
    if w < 2:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (w // 2, w - 1 - w // 2)
    b = np.pad(a, pad, mode="edge")
    zpad = [(0, 0)] * a.ndim
    zpad[axis] = (1, 0)
    c = np.cumsum(np.pad(b, zpad, mode="constant"), axis=axis)
    sl, sl0 = [slice(None)] * a.ndim, [slice(None)] * a.ndim
    sl[axis] = slice(w, None)
    sl0[axis] = slice(0, -w)
    return (c[tuple(sl)] - c[tuple(sl0)]) / w


def bilinear(grid, ax, ay, qx, qy):
    i, u = _interp1_weights(qx, ax)
    j, v = _interp1_weights(qy, ay)
    return (grid[i, j] * ((1 - u) * (1 - v)) + grid[i + 1, j] * (u * (1 - v))
            + grid[i, j + 1] * ((1 - u) * v) + grid[i + 1, j + 1] * (u * v))


# --------------------------------------------------------------- build grid
class Background:
    def __init__(self, nx=110, ny=116, nz=133, mode="selfconsistent",
                 ):
        self.mode = mode
        g = np.load(os.path.join(DATA, "grids.npz"))

        self.x = np.linspace(0.0, LX, nx)
        self.y = np.linspace(0.0, LY, ny)
        self.z = np.linspace(0.0, LZ, nz)          # depth, positive down
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        lon, lat = xy_to_lonlat(X - LX / 2, Y - LY / 2)

        # ---- shear-wave speed --------------------------------------------
        vs_grid = np.asarray(g["vs"], dtype=np.float64)       # [r, lon, lat]
        vs_grid = np.transpose(vs_grid, (1, 2, 0))            # [lon, lat, r]
        r_ax = np.asarray(g["r"], dtype=np.float64)
        dep_ax = (ph.R_OUTER - r_ax)[::-1]                    # ascending depth
        vs_grid = vs_grid[:, :, ::-1]

        vs = trilinear(vs_grid, np.asarray(g["vs_lon"], dtype=np.float64),
                       np.asarray(g["vs_lat"], dtype=np.float64), dep_ax,
                       lon, lat, Z)

        vs_ref_prof = np.asarray(g["vs_ref"], dtype=np.float64)[::-1]
        vs_ref = np.interp(Z, dep_ax, vs_ref_prof)
        self.vs = vs
        self.dlnvs = (vs - vs_ref) / vs_ref

        # ---- lithosphere-asthenosphere boundary ---------------------------
        self.lab = bilinear(np.asarray(g["lab"], dtype=np.float64),
                            np.asarray(g["lab_lon"], dtype=np.float64),
                            np.asarray(g["lab_lat"], dtype=np.float64),
                            lon, lat)

        # ---- temperature, pressure, melt ----------------------------------
        self.P = ph.lithostatic_pressure(Z)
        self.T, self.F = ph.temperature_field(Z, self.lab, self.dlnvs,
                                              mode=mode)
        # the literal ASPECT-plugin conversion, kept for comparison
        self.T_plugin, self.F_plugin = ph.temperature_field(
            Z, self.lab, self.dlnvs, mode="plugin")

        # productivity gradient (Cartesian, m^-1)
        gx, gy, gz = np.gradient(self.F, self.x, self.y, self.z, edge_order=2)
        self.dFdx, self.dFdy, self.dFdz = gx, gy, gz
        self.gradT = list(np.gradient(self.T, self.x, self.y, self.z, edge_order=2))

        # LAB surface and its horizontal gradients, for the rigid-lid mask.
        # psi = z - z_LAB is the signed distance to the LAB (>0 = asthenosphere)
        lab2d = self.lab[:, :, 0]
        gx2, gy2 = np.gradient(lab2d, self.x, self.y, edge_order=2)
        self.dlabx = np.repeat(gx2[:, :, None], len(self.z), axis=2)
        self.dlaby = np.repeat(gy2[:, :, None], len(self.z), axis=2)

        # reference (horizontally averaged) temperature for buoyancy
        self.Tbar = self.T.mean(axis=(0, 1))
        self.dT = self.T - self.Tbar[None, None, :]

        self._build_phi_ref()

        self.shape = (nx, ny, nz)

    # -------------------------------------------------- porosity precondition
    def _build_phi_ref(self, w_ref=3.17e-11, h=5.0e4, k0=1.0e-7,
                       phi_floor=1.0e-6, pond_km=10.0):
        """Analytic 1-D steady melt-retention porosity used as a preconditioner.

        Balancing melt production against Darcy escape over a height h,
            (rho_s/rho_f) w |grad F| = d/dz [ k0 phi^3 / mu * drho g ] ~ q/h
        gives   phi_ref = [ (rho_s/rho_f) w |grad F| h mu / (k0 drho g) ]^(1/3),
        which is O(2e-3) for the CVL.  The network learns a bounded
        multiplicative correction to this field, which removes the vanishing
        gradient that phi^3 permeability otherwise produces near phi = 0.
        """
        gF = np.sqrt(self.dFdx ** 2 + self.dFdy ** 2 + self.dFdz ** 2)
        num = (ph.RHO_S / ph.RHO_F) * w_ref * gF * h * ph.MU_F
        den = k0 * ph.DRHO * ph.GRAVITY
        phi_ref = np.cbrt(np.maximum(num / den, 0.0))
        # melt ponds above where it is produced: propagate the value upward
        nz = max(1, int(pond_km * 1e3 / (self.z[1] - self.z[0])))
        for _ in range(nz):
            phi_ref[:, :, :-1] = np.maximum(phi_ref[:, :, :-1], phi_ref[:, :, 1:])

        # The preconditioner must be smooth, because d(phi)/dx = phi_ref
        # d(ln phi_ref)/dx feeds the melt advection term.  The quantity that
        # has to be bounded is grad(phi_ref) itself, so smooth in LINEAR
        # space: this caps grad(phi_ref) at phi_max / L_smooth while leaving
        # the far field near zero.  (Smoothing in log space instead raises the
        # far-field floor, which injects spurious melt over the whole domain
        # and drives the solver to phi -> 0.)
        for _ in range(2):
            phi_ref = _boxcar(_boxcar(_boxcar(phi_ref, 5, 0), 5, 1), 9, 2)
        self.phi_ref = np.maximum(phi_ref, phi_floor)
        lg = np.log(self.phi_ref)
        self.dlpx, self.dlpy, self.dlpz = np.gradient(
            lg, self.x, self.y, self.z, edge_order=2)

    # ------------------------------------------------------------- sampling
    def sample_melt(self, n, rng):
        """Points drawn from the melting column and the layer just above it."""
        if not hasattr(self, "_mcells"):
            gz = np.abs(self.dFdz) + np.abs(self.dFdx) + np.abs(self.dFdy)
            active = (self.F > 1e-6) | (gz > gz.max() * 1e-3)
            # include the 40 km directly above, where melt ponds and refreezes
            for _ in range(int(40e3 / (self.z[1] - self.z[0]))):
                active[:, :, :-1] |= active[:, :, 1:]
            self._mcells = np.argwhere(active)
        idx = self._mcells[rng.integers(0, len(self._mcells), n)]
        hx, hy, hz = (self.x[1] - self.x[0], self.y[1] - self.y[0],
                      self.z[1] - self.z[0])
        return (np.clip(self.x[idx[:, 0]] + rng.uniform(-hx, hx, n), 0, LX),
                np.clip(self.y[idx[:, 1]] + rng.uniform(-hy, hy, n), 0, LY),
                np.clip(self.z[idx[:, 2]] + rng.uniform(-hz, hz, n), 0, LZ))

    def sample(self, qx, qy, qz, names=("T", "dT", "F", "dFdx", "dFdy", "dFdz",
                                        "lab", "dlnvs")):
        out = {}
        for n in names:
            arr = getattr(self, n)
            if arr.ndim == 3:
                out[n] = trilinear(arr, self.x, self.y, self.z, qx, qy, qz)
            else:
                out[n] = np.interp(qz, self.z, arr)
        return out


def report(bg, tag):
    print("\n=== %s ===" % tag)
    print("T    range: %.0f .. %.0f K   F_eq max: %.4f" %
          (bg.T.min(), bg.T.max(), bg.F.max()))
    print("  depth   F_max   F_mean(F>0)   area(F>0)   T range (K)      Tsol")
    for d in (60, 80, 100, 120, 150, 200, 250, 300, 400):
        k = np.argmin(abs(bg.z - d * 1e3))
        f = bg.F[:, :, k]
        m = f > 1e-5
        print("  %3d km  %.4f   %s   %6.1f%%   %4.0f-%4.0f   %5.0f/%5.0f"
              % (d, f.max(),
                 ("%.4f" % f[m].mean()) if m.any() else "  --  ",
                 100 * m.mean(), bg.T[:, :, k].min(), bg.T[:, :, k].max(),
                 ph.solidus(ph.lithostatic_pressure(d * 1e3)),
                 ph.hydrous_solidus(ph.lithostatic_pressure(d * 1e3))))


if __name__ == "__main__":
    print("domain  Lx,Ly,Lz (km):", round(LX / 1e3), round(LY / 1e3), round(LZ / 1e3))

    bg = Background(mode="selfconsistent")
    print("dlnVs range: %.3f .. %.3f   LAB %.0f-%.0f km"
          % (bg.dlnvs.min(), bg.dlnvs.max(), bg.lab.min() / 1e3, bg.lab.max() / 1e3))
    report(bg, "MODEL A  hydrous Katz 2003, self-consistent Vs inversion")

    bgp = Background.__new__(Background)
    bgp.__dict__ = dict(bg.__dict__)
    bgp.T, bgp.F = bg.T_plugin, bg.F_plugin
    report(bgp, "PLUGIN   literal potential_temperature.cc conversion")

    np.savez_compressed(os.path.join(DATA, "background.npz"),
                        x=bg.x, y=bg.y, z=bg.z, T=bg.T, F=bg.F, lab=bg.lab,
                        dlnvs=bg.dlnvs, vs=bg.vs, dT=bg.dT,
                        T_plugin=bg.T_plugin, F_plugin=bg.F_plugin,
                        dFdx=bg.dFdx, dFdy=bg.dFdy, dFdz=bg.dFdz)
    print("\nsaved background.npz")
