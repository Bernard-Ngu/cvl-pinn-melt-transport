"""Correctness checks for the autodiff engine and the PINN residual assembly."""
import numpy as np
import autodiff as ad
import model as md
import fields as fl


def check_input_jacobian():
    """d(out)/dx from mlp_with_jac vs central finite differences."""
    rng = np.random.default_rng(1)
    p = md.init_params(5, 16, 3, seed=3)
    n = 7
    x = rng.uniform(0, fl.LX, n)
    y = rng.uniform(0, fl.LY, n)
    z = rng.uniform(0, fl.LZ, n)

    # a smooth analytic stand-in for the interpolated background features
    def feats(x, y, z):
        F = 0.01 * np.exp(-((z - 1.2e5) / 6e4) ** 2) * (1 + 0.3 * np.sin(x / 4e5))
        T = 1600 + 200 * np.sin(y / 5e5) - 3e-4 * z
        return F, T

    def inputs(x, y, z):
        F, T = feats(x, y, z)
        Xn, Yn, d = 2 * x / fl.LX - 1, 2 * y / fl.LY - 1, z / fl.LZ
        return np.stack([Xn, Yn, d, F / md.PHI0, (T - 1600) / 300.0], axis=1)

    h = 50.0
    J0 = []
    for k in range(3):
        dx = [np.zeros(n), np.zeros(n), np.zeros(n)]
        dx[k] = np.full(n, h)
        Jp = (inputs(x + dx[0], y + dx[1], z + dx[2])
              - inputs(x - dx[0], y - dx[1], z - dx[2])) / (2 * h)
        J0.append(Jp)

    out, J = md.mlp_with_jac(p, inputs(x, y, z), J0)
    err = 0.0
    for k in range(3):
        dx = [np.zeros(n), np.zeros(n), np.zeros(n)]
        dx[k] = np.full(n, h)
        op, _ = md.mlp_with_jac(p, inputs(x + dx[0], y + dx[1], z + dx[2]), J0)
        om, _ = md.mlp_with_jac(p, inputs(x - dx[0], y - dx[1], z - dx[2]), J0)
        fd = (op.v - om.v) / (2 * h)
        err = max(err, np.abs(fd - J[k].v).max() / (np.abs(fd).max() + 1e-30))
    print("input-Jacobian rel. error      : %.2e" % err)
    return err


def check_param_gradient():
    """dLoss/dparams from the tape vs central finite differences."""
    rng = np.random.default_rng(2)
    n_in, n = 5, 11
    p = md.init_params(n_in, 12, 3, seed=5)
    x0 = rng.normal(0, 1, (n, n_in))
    J0 = [rng.normal(0, 1e-6, (n, n_in)) for _ in range(3)]

    def loss_of(params):
        out, J = md.mlp_with_jac(params, x0, J0)
        return (out * out).mean() + sum((Jk * Jk).mean() * 1e12 for Jk in J)

    L = loss_of(p)
    L.backward()
    ana = [t.grad.copy() for t in p]

    eps, worst = 1e-6, 0.0
    for ti, t in enumerate(p):
        it = np.ndindex(t.v.shape)
        for _ in range(4):
            idx = next(it)
            orig = t.v[idx]
            t.v[idx] = orig + eps
            lp = loss_of(p).v
            t.v[idx] = orig - eps
            lm = loss_of(p).v
            t.v[idx] = orig
            fd = (lp - lm) / (2 * eps)
            rel = abs(fd - ana[ti][idx]) / (abs(fd) + 1e-12)
            worst = max(worst, rel)
    print("parameter-gradient rel. error  : %.2e" % worst)
    return worst


def check_hard_bcs():
    """Velocity and melt flux must vanish on the prescribed boundaries."""
    geo = md.Geometry()
    p = md.init_params(5, 16, 3, seed=7)
    rng = np.random.default_rng(4)
    n = 400
    faces = {
        "west  x=0":   (np.zeros(n), rng.uniform(0, fl.LY, n), rng.uniform(0, fl.LZ, n)),
        "east  x=Lx":  (np.full(n, fl.LX), rng.uniform(0, fl.LY, n), rng.uniform(0, fl.LZ, n)),
        "south y=0":   (rng.uniform(0, fl.LX, n), np.zeros(n), rng.uniform(0, fl.LZ, n)),
        "north y=Ly":  (rng.uniform(0, fl.LX, n), np.full(n, fl.LY), rng.uniform(0, fl.LZ, n)),
        "inner z=Lz":  (rng.uniform(0, fl.LX, n), rng.uniform(0, fl.LY, n), np.full(n, fl.LZ)),
        "outer z=0":   (rng.uniform(0, fl.LX, n), rng.uniform(0, fl.LY, n), np.zeros(n)),
    }
    worst = 0.0
    for name, (x, y, z) in faces.items():
        x0 = np.stack([2 * x / fl.LX - 1, 2 * y / fl.LY - 1, z / fl.LZ,
                       np.zeros(n), np.zeros(n)], axis=1)
        J0 = [np.zeros((n, 5)) for _ in range(3)]
        raw, draw = md.mlp_with_jac(p, x0, J0)
        out, _ = md.apply_constraints(raw, draw, geo, x, y, z)
        u = np.abs(np.concatenate([c.v for c in out["u"]], axis=1)).max() / md.U0
        if name == "outer z=0":
            u = np.abs(out["u"][2].v).max() / md.U0          # free slip: only u_z
        qn = {"west  x=0": 0, "east  x=Lx": 0, "south y=0": 1, "north y=Ly": 1,
              "inner z=Lz": 2, "outer z=0": 2}[name]
        qq = np.abs(out["q"][qn].v).max() / md.Q0
        print("  %-12s  |u|/U0 = %.2e   |q.n|/Q0 = %.2e" % (name, u, qq))
        worst = max(worst, u, qq)
    return worst


def check_katz():
    """Katz 2003 against values computed directly from the published formulae."""
    import physics as ph
    ok = True
    for P_GPa, T_C in ((1.0, 1250), (2.0, 1400), (3.0, 1500), (0.5, 1200)):
        P, T = P_GPa * 1e9, T_C + 273.15
        Ts = 1085.7 + 273.15 + 1.329e-7 * P - 5.1e-18 * P * P
        Tl = 1475.0 + 273.15 + 8.0e-8 * P - 3.2e-18 * P * P
        Tliq = 1780.0 + 273.15 + 4.50e-8 * P - 2.0e-18 * P * P
        F = 0.0 if T < Ts else (1.0 if T > Tl else ((T - Ts) / (Tl - Ts)) ** 1.5)
        Fmax = 0.15 / (0.5 + 8e-11 * P)
        if F > Fmax and T < Tliq:
            Tmax = Fmax ** (1 / 1.5) * (Tl - Ts) + Ts
            F = Fmax + (1 - Fmax) * ((T - Tmax) / (Tliq - Tmax)) ** 1.5
        mine = float(ph.melt_fraction(np.array(T), np.array(P)))
        ok &= abs(mine - F) < 1e-10
        print("  P=%.1f GPa T=%4d C : F_ref=%.6f  F_port=%.6f" % (P_GPa, T_C, F, mine))
    return ok


def check_lid():
    """The lid mask must kill the right velocity components in the lithosphere."""
    import model as md
    import numpy as np
    bg = fl.Background()
    geo = md.Geometry()
    p = md.init_params(5, 16, 3, seed=11)
    rng = np.random.default_rng(9)
    n = 3000
    x, y, z = md.sample_points(bg, n, rng)
    worst = {}
    for mode in ("locked", "slip"):
        md.set_lid(mode)
        x0, J0, s = md.build_inputs(bg, x, y, z, geo)
        raw, draw = md.mlp_with_jac(p, x0, J0)
        out, _ = md.apply_constraints(raw, draw, geo, x, y, z,
                                      s["pref"], s["dlp"], s["lid"])
        lith = (z < s["lab"] - 1.0)
        uh = np.hypot(out["u"][0].v[:, 0], out["u"][1].v[:, 0])
        uz = np.abs(out["u"][2].v[:, 0])
        print("  %-7s lithosphere: max|u_h|/U0 = %.2e   max|u_z|/U0 = %.2e"
              " (n=%d)" % (mode, uh[lith].max() / md.U0, uz[lith].max() / md.U0,
                           lith.sum()))
        print("          asthenosphere: max|u_h|/U0 = %.2e   max|u_z|/U0 = %.2e"
              % (uh[~lith].max() / md.U0, uz[~lith].max() / md.U0))
        worst[mode] = (uh[lith].max() / md.U0, uz[lith].max() / md.U0)
    md.set_lid("none")
    ok = (worst["locked"][0] == 0.0 and worst["locked"][1] == 0.0
          and worst["slip"][1] == 0.0 and worst["slip"][0] > 0.0)
    print("  locked kills all components, slip kills only u_z:", ok)
    return ok


if __name__ == "__main__":
    print("--- Katz 2003 port ---------------------------------------------")
    kok = check_katz()
    print("--- autodiff ---------------------------------------------------")
    e1 = check_input_jacobian()
    e2 = check_param_gradient()
    print("--- hard boundary conditions -----------------------------------")
    e3 = check_hard_bcs()
    print("--- rigid lid --------------------------------------------------")
    lok = check_lid()
    print("\nAll checks pass:",
          bool(kok) and bool(lok) and e1 < 1e-5 and e2 < 1e-5 and e3 < 1e-12)
