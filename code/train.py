"""Train the CVL two-phase PINN."""
import argparse
import os
import time
import numpy as np

import autodiff as ad
import fields as fl
import model as md
import physics as ph

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
os.makedirs(OUTD, exist_ok=True)


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(t.v) for t in params]
        self.v = [np.zeros_like(t.v) for t in params]
        self.t = 0

    def step(self, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, p in enumerate(self.p):
            g = p.grad
            if g is None:
                continue
            g = np.clip(g, -1e3, 1e3)
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p.v -= lr * mh / (np.sqrt(vh) + eps)
            p.grad = None


WORK_FLOOR = 0.10
GROUPS = ("mom", "rheo", "solid", "melt", "darcy")


def total_loss(R, w, work=None, w_work=0.0):
    terms, L = {}, None
    for g in GROUPS:
        s = None
        for r in R[g]:
            s = ad.mse(r) if s is None else s + ad.mse(r)
        s = s * (1.0 / len(R[g]))
        terms[g] = s
        L = s * w[g] if L is None else L + s * w[g]
    if work is not None and w_work:
        # Hinge on the buoyancy-work integral.  A true Stokes solution
        # dissipates energy, so the work is positive; the penalty is exactly
        # zero once it exceeds WORK_FLOOR and therefore biases nothing.
        d = work * (-1.0) + WORK_FLOOR
        pen = ad.where(d.v > 0.0, d, 0.0)
        L = L + pen * pen * w_work
    return L, terms


def load_params(tag, width, depth):
    d = np.load(os.path.join(OUTD, "params_%s.npz" % tag))
    ps = []
    for i in range(2 * (depth + 1)):
        ps.append(ad.Tensor(d["arr_%d" % i], requires_grad=True))
    return ps, d["hist"] if "hist" in d else np.zeros((0, 7))


def train(lid='locked', k0=1e-7, iters=4000, batch=2600, width=64,
          depth=5, lr=2e-3, seed=0, tag="A", log_every=100,
          start_it=0, total=None, resume=False, budget=None):
    total = total or iters
    rng = np.random.default_rng(seed + start_it)
    md.set_permeability(k0)
    md.set_lid(lid)
    bg = fl.Background(mode="selfconsistent")
    geo = md.Geometry()
    if resume and os.path.exists(os.path.join(OUTD, "params_%s.npz" % tag)):
        params, hist0 = load_params(tag, width, depth)
        if len(hist0):
            start_it = int(hist0[-1, 0])
        print("resumed from checkpoint at iteration", start_it)
    else:
        params = md.init_params(5, width, depth, seed=seed)
        hist0 = np.zeros((0, 7))
    opt = Adam(params, lr)

    w = dict(mom=1.0, rheo=1.0, solid=1.0, melt=0.0, darcy=0.0)
    W_WORK = 30.0
    hist = []
    t0 = time.time()

    for it in range(start_it + 1, start_it + iters + 1):
        if budget and time.time() - t0 > budget:
            print("time budget reached at iteration", it)
            break
        frac = it / total
        # curriculum -----------------------------------------------------
        visc_pow = min(1.0, 0.35 + 0.65 * min(1.0, frac / 0.35))
        if frac > 0.12:
            ramp = min(1.0, (frac - 0.12) / 0.13)
            w["melt"], w["darcy"] = ramp, ramp
        cur_lr = lr * (0.5 * (1 + np.cos(np.pi * min(1.0, frac))) * 0.99 + 0.01)

        x, y, z = md.sample_points(bg, batch, rng)
        # the last block of sample_points() is drawn uniformly over the box
        n_u = batch - int(batch * 0.3) - int(batch * 0.45)
        wmask = np.zeros((len(x), 1))
        wmask[-n_u:] = 1.0
        x0, J0, s = md.build_inputs(bg, x, y, z, geo)
        raw, draw = md.mlp_with_jac(params, x0, J0)
        out, dout = md.apply_constraints(raw, draw, geo, x, y, z, s['pref'], s['dlp'], s['lid'])
        R, aux = md.residuals(out, dout, s, k0, visc_pow=visc_pow,
                              work_mask=wmask)
        L, terms = total_loss(R, w, work=aux["work"], w_work=W_WORK)
        L.backward()
        opt.step(cur_lr)

        if it % log_every == 0 or it == 1:
            rec = dict(it=it, loss=float(L.v),
                       **{g: float(terms[g].v) for g in GROUPS},
                       eta_med=float(np.median(aux["eta"])),
                       phi_max=float(out["phi"].v.max()),
                       work=float(aux["work"].v.mean()),
                       umax_mmyr=float(np.abs(np.concatenate(
                           [c.v for c in out["u"]], 1)).max() * 3.15576e10))
            hist.append(rec)
            print("it %5d  L=%9.3e  mom=%8.2e rheo=%8.2e solid=%8.2e "
                  "melt=%8.2e darcy=%8.2e | eta=%.1e phi_max=%.2e "
                  "|u|max=%.2f mm/yr  [%.0fs]"
                  % (it, rec["loss"], rec["mom"], rec["rheo"], rec["solid"],
                     rec["melt"], rec["darcy"], rec["eta_med"], rec["phi_max"],
                     rec["umax_mmyr"], time.time() - t0), flush=True)
            print("        buoyancy work (must be > 0): %+.3e" % rec["work"],
                  flush=True)

    path = os.path.join(OUTD, "params_%s.npz" % tag)
    hnew = np.array([[r[k] for k in ("it", "loss", "mom", "rheo", "solid",
                                     "melt", "darcy")] for r in hist])
    hall = np.vstack([hist0, hnew]) if len(hnew) else hist0
    np.savez_compressed(path, *[t.v for t in params], hist=hall,
                        meta=np.array([width, depth, k0]))
    print("saved", path, " total logged iterations:", len(hall))
    return params, bg, hall


def evaluate(params, bg, k0, tag, lid='locked', nx=90, ny=94, nz=100, chunk=6000):
    """Dense evaluation on a regular grid; writes runs/solution_<tag>.npz."""
    geo = md.Geometry()
    md.set_permeability(k0)
    md.set_lid(lid)
    gx = np.linspace(0, fl.LX, nx)
    gy = np.linspace(0, fl.LY, ny)
    gz = np.linspace(0, fl.LZ, nz)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    x, y, z = X.ravel(), Y.ravel(), Z.ravel()

    keys = ["ux", "uy", "uz", "phi", "P", "pc", "qx", "qy", "qz",
            "eta", "Gamma", "F", "T", "edot"]
    acc = {k: np.empty(len(x)) for k in keys}
    for a in range(0, len(x), chunk):
        b = min(a + chunk, len(x))
        x0, J0, s = md.build_inputs(bg, x[a:b], y[a:b], z[a:b], geo)
        raw, draw = md.mlp_with_jac(params, x0, J0)
        out, dout = md.apply_constraints(raw, draw, geo, x[a:b], y[a:b], z[a:b], s['pref'], s['dlp'], s['lid'])
        R, aux = md.residuals(out, dout, s, k0)
        for i, k in enumerate(("ux", "uy", "uz")):
            acc[k][a:b] = out["u"][i].v[:, 0]
            acc["q" + k[1]][a:b] = out["q"][i].v[:, 0]
        acc["phi"][a:b] = out["phi"].v[:, 0]
        acc["P"][a:b] = out["P"].v[:, 0]
        acc["pc"][a:b] = out["pc"].v[:, 0]
        acc["eta"][a:b] = aux["eta"][:, 0]
        acc["Gamma"][a:b] = aux["Gamma"][:, 0]
        acc["edot"][a:b] = aux["edot"][:, 0]
        acc["F"][a:b] = s["F"]
        acc["T"][a:b] = s["T"]

    sol = {k: v.reshape(nx, ny, nz).astype(np.float32) for k, v in acc.items()}
    sol.update(x=gx, y=gy, z=gz)
    np.savez(os.path.join(OUTD, "solution_%s.npz" % tag), **sol)
    print("saved solution_%s.npz" % tag)
    return sol


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="A")
    ap.add_argument("--lid", default="locked",
                    choices=["none", "locked", "slip"])
    ap.add_argument("--k0", type=float, default=1e-7)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2600)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=float, default=0)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    p, bg, h = train(lid=a.lid, k0=a.k0, iters=a.iters,
                     batch=a.batch, width=a.width, depth=a.depth, tag=a.tag,
                     start_it=a.start, total=a.total or None, resume=a.resume,
                     budget=a.budget or None)
    if not a.no_eval:
        evaluate(p, bg, a.k0, a.tag, lid=a.lid)
