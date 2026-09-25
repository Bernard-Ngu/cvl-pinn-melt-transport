"""Verification diagnostics for the trained CVL PINN solutions.

Every residual is normalised by the size of the terms that make it up, so the
numbers below are *relative* errors rather than the raw nondimensional losses
reported during training (which are keyed to the velocity scale U0 and so
flatter the solution by roughly the ratio U0/|u|).
"""
import os
import numpy as np
import fields as fl
import physics as ph
import model as md
import figures as F
import train

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.abspath(os.path.join(HERE, "..", "runs"))
YR = 3.15576e7


def _terms(tag, k0, n=20000, seed=7, lid="none", width=64):
    md.set_permeability(k0)
    md.set_lid(lid)
    bg = fl.Background(mode="selfconsistent")
    geo = md.Geometry()
    p, _ = train.load_params(tag, width, 5)
    rng = np.random.default_rng(seed)
    x, y, z = md.sample_points(bg, n, rng)
    out = {}
    for a in range(0, n, 5000):
        b = min(a + 5000, n)
        x0, J0, s = md.build_inputs(bg, x[a:b], y[a:b], z[a:b], geo)
        raw, draw = md.mlp_with_jac(p, x0, J0)
        o, do = md.apply_constraints(raw, draw, geo, x[a:b], y[a:b], z[a:b],
                                     s["pref"], s["dlp"], s.get("lid"))
        R, aux = md.residuals(o, do, s, k0)
        u = [c.v for c in o["u"]]
        du = [[c.v for c in r] for r in do["u"]]
        dq = [[c.v for c in r] for r in do["q"]]
        phi, dphi = o["phi"].v, [c.v for c in do["phi"]]
        d = dict(
            F=s["F"][:, None], z=z[a:b, None], phi=phi,
            Gam=aux["Gamma"] / ph.RHO_F,
            divq=dq[0][0] + dq[1][1] + dq[2][2],
            divphiu=sum(dphi[k] * u[k] + phi * du[k][k] for k in range(3)),
            divu=du[0][0] + du[1][1] + du[2][2],
            Rmom=np.hstack([r.v for r in R["mom"]]),
            Rrheo=np.hstack([r.v for r in R["rheo"]]),
            speed=np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2),
            gradu=np.sqrt(sum(du[i][j] ** 2 for i in range(3) for j in range(3))),
            eta=aux["eta"])
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def buoyancy_work(tag):
    """Normalised work integral; must be positive for any Stokes solution."""
    s = F.load(tag)
    x, y, z = [np.asarray(s[k], dtype=float) for k in ("x", "y", "z")]
    T = np.asarray(s["T"], dtype=float)
    phi = np.asarray(s["phi"], dtype=float)
    uz = np.asarray(s["uz"], dtype=float)
    dT = T - T.mean(axis=(0, 1))[None, None, :]
    bz = -(ph.RHO_S * ph.ALPHA * dT + phi * ph.DRHO) * ph.GRAVITY
    w = bz * uz
    num = np.trapezoid(np.trapezoid(np.trapezoid(w, z, axis=2), y, axis=1), x)
    den = np.trapezoid(np.trapezoid(np.trapezoid(np.abs(w), z, axis=2),
                                    y, axis=1), x)
    return float(num / max(den, 1e-300))


def relative(a, ref):
    return np.sqrt((a ** 2).mean()) / max(np.sqrt((ref ** 2).mean()), 1e-300)


def report(tag, k0, lid="none"):
    t = _terms(tag, k0, lid=lid)
    hot = (t["F"] > 1e-6).ravel()
    print("\n=========== Model %s  (k0 = %.0e m2) ==========="
          % (tag, k0))

    # 1. incompressibility, measured against the actual velocity gradients
    print("1. solid mass      rms|div u| / rms|grad u| = %.4f"
          % relative(t["divu"], t["gradu"]))

    # 2. momentum, measured against the buoyancy that drives it
    print("2. momentum        rms residual / rms driving term = %.3f"
          % relative(t["Rmom"], np.full_like(t["Rmom"], 1.0)))

    # 3. rheology
    print("3. rheology        rms(tau - 2 eta e) / rms(tau scale) = %.4f"
          % np.sqrt((t["Rrheo"] ** 2).mean()))

    # 4. melt transport inside the melting column
    r = (t["divphiu"] + t["divq"] - t["Gam"]).ravel()
    print("4. melt transport  rms residual / rms source, melting column = %.3f"
          % relative(r[hot], t["Gam"].ravel()[hot]))
    print("                   melt flux divergence carries %.0f%% of the source"
          % (100 * np.sqrt((t["divq"].ravel()[hot] ** 2).mean())
             / max(np.sqrt((t["Gam"].ravel()[hot] ** 2).mean()), 1e-300)))

    # 5. porosity and its parameterisation bound
    md.set_permeability(k0)
    bg = fl.Background(mode="selfconsistent")
    print("5. porosity        column median %.2e  max %.2e"
          % (np.median(t["phi"].ravel()[hot]), t["phi"].max()))
    print("                   preconditioner phi_ref column median %.2e; the"
          % np.median(bg.phi_ref[bg.F > 1e-6]))
    print("                   network correction is at its lower bound (x%.3f),"
          % np.exp(-md.SPAN))
    print("                   so the retained porosity is an UPPER bound.")

    # 6. velocities and viscosity
    sp = t["speed"] * YR * 1e3
    print("6. solid velocity  median %.2f  p99 %.2f  max %.2f mm/yr"
          % (np.median(sp), np.percentile(sp, 99), sp.max()))
    print("   viscosity       %.1e - %.1e Pa s" % (t["eta"].min(), t["eta"].max()))
    return t


def melt_by_depth(tag):
    s = np.load(os.path.join(RUNS, "solution_%s.npz" % tag))
    z = s["z"]
    print("\n  depth   F_eq max   phi max    phi mean   melt flux (mm/yr)")
    for d in (100, 150, 200, 250, 300):
        k = int(np.argmin(abs(z - d * 1e3)))
        print("  %3d km  %.2e   %.2e   %.2e   %.3e"
              % (d, s["F"][:, :, k].max(), s["phi"][:, :, k].max(),
                 s["phi"][:, :, k].mean(),
                 np.abs(s["qz"][:, :, k]).max() * YR * 1e3))


def permeability_scaling():
    """phi ~ k0^{-1/3} at fixed melting rate: the prm value raises retention."""
    print("\n--- permeability sensitivity (analytic, at fixed melting rate) ---")
    print("Steady retention obeys  k0 phi^3 drho g / mu = q = Gamma h / rho_f,")
    print("so phi ~ k0^{-1/3}.  Relative to k0 = 1e-7 m2:")
    for k0 in (1e-7, 1e-8, 1e-9, 1e-10):
        print("   k0 = %.0e  ->  phi x %.2f %s"
              % (k0, (k0 / 1e-7) ** (-1.0 / 3.0),
                 "   <- value set in original.prm" if k0 == 1e-10 else ""))


if __name__ == "__main__":
    report("A", 1e-7)
    melt_by_depth("A")
    permeability_scaling()
