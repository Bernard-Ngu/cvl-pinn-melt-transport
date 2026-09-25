"""Reviewer experiments for the CVL manuscript.

Two things the reviewers asked for:

  1. an architecture sweep  -- does the reported solution depend on the
     particular choice of 5 hidden layers of 64 neurons?
  2. a model without the physics -- what do the conservation residuals
     actually contribute?

Every configuration is trained for the SAME number of iterations with the
SAME seed, batch size, optimiser and curriculum as the reported run, and
every one is then evaluated on the SAME fixed set of collocation points, so
the residuals in the table are comparable across rows.

Modes
-----
  pinn      the reported objective: five conservation residuals plus the
            one-sided buoyancy-work penalty
  nowork    identical, but with the buoyancy-work penalty switched off
  dataonly  no conservation residuals at all.  The network is trained
            against the only label that exists without invoking the
            governing equations: the porosity estimated from the tomography
            by the 1-D melt-production / Darcy-drainage balance.  Nothing
            constrains the velocity, because nothing observational does.

Runs are resumable: --budget stops the loop after that many seconds and
saves a checkpoint, and --resume picks it up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# every module lives beside this file
sys.path.insert(0, HERE)

import autodiff as ad          # noqa: E402
import fields as fl            # noqa: E402
import model as md             # noqa: E402
import physics as ph           # noqa: E402
from train import Adam, total_loss, GROUPS   # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "experiments", "results"))
os.makedirs(RES, exist_ok=True)

EVAL_N = 6000
EVAL_SEED = 12345
W_WORK = 30.0


# ----------------------------------------------------------------- helpers
def n_params(width, depth):
    dims = [5] + [width] * depth + [md.N_OUT]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def eval_points(bg):
    rng = np.random.default_rng(EVAL_SEED)
    return md.sample_points(bg, EVAL_N, rng)


def work_mask_for(n, batch):
    m = np.zeros((n, 1))
    n_u = batch - int(batch * 0.3) - int(batch * 0.45)
    m[-n_u:] = 1.0
    return m


def diagnose(params, bg, geo, pts, k0=1e-7):
    """Residuals of a trained network on the common evaluation batch."""
    x, y, z = pts
    x0, J0, s = md.build_inputs(bg, x, y, z, geo)
    raw, draw = md.mlp_with_jac(params, x0, J0)
    out, dout = md.apply_constraints(raw, draw, geo, x, y, z,
                                     s['pref'], s['dlp'], s['lid'])
    R, aux = md.residuals(out, dout, s, k0,
                          work_mask=work_mask_for(len(x), EVAL_N))
    d = {}
    for g in GROUPS:
        v = sum(float(ad.mse(r).v) for r in R[g]) / len(R[g])
        d[g] = float(np.sqrt(v))                    # r.m.s. relative residual
    pref = s["pref"][:, None]
    d["phi_misfit"] = float(np.sqrt(np.mean(((out["phi"].v - pref)
                                             / pref) ** 2)))
    d["work"] = float(aux["work"].v)
    d["umax_mmyr"] = float(np.abs(np.concatenate(
        [c.v for c in out["u"]], 1)).max() * 3.15576e10)
    d["phi_max"] = float(out["phi"].v.max())
    return d


# ------------------------------------------------------------------- train
def run(tag, mode="pinn", width=64, depth=5, iters=1500, batch=2600,
        lr=2e-3, seed=0, lid="none", k0=1e-7, budget=None, log_every=50):
    ck = os.path.join(RES, "ck_%s.npz" % tag)
    md.set_permeability(k0)
    md.set_lid(lid)
    bg = fl.Background(mode="selfconsistent")
    geo = md.Geometry()

    if os.path.exists(ck):
        z = np.load(ck)
        params = [ad.Tensor(z["arr_%d" % i], requires_grad=True)
                  for i in range(2 * (depth + 1))]
        start, elapsed = int(z["it"]), float(z["elapsed"])
        hist = list(z["hist"]) if len(z["hist"]) else []
    else:
        params = md.init_params(5, width, depth, seed=seed)
        start, elapsed, hist = 0, 0.0, []

    if start >= iters:
        return finish(tag, params, bg, geo, mode, width, depth, iters,
                      elapsed, hist)

    opt = Adam(params, lr)
    rng = np.random.default_rng(seed + start)
    t0 = time.time()

    for it in range(start + 1, iters + 1):
        if budget and time.time() - t0 > budget:
            break
        frac = it / iters
        visc_pow = min(1.0, 0.35 + 0.65 * min(1.0, frac / 0.35))
        w = dict(mom=1.0, rheo=1.0, solid=1.0, melt=0.0, darcy=0.0)
        if frac > 0.12:
            ramp = min(1.0, (frac - 0.12) / 0.13)
            w["melt"], w["darcy"] = ramp, ramp
        cur_lr = lr * (0.5 * (1 + np.cos(np.pi * min(1.0, frac))) * 0.99 + 0.01)

        x, y, z_ = md.sample_points(bg, batch, rng)
        x0, J0, s = md.build_inputs(bg, x, y, z_, geo)
        raw, draw = md.mlp_with_jac(params, x0, J0)
        out, dout = md.apply_constraints(raw, draw, geo, x, y, z_,
                                         s['pref'], s['dlp'], s['lid'])
        R, aux = md.residuals(out, dout, s, k0, visc_pow=visc_pow,
                              work_mask=work_mask_for(len(x), batch))

        if mode == "dataonly":
            pref = s["pref"][:, None]
            L = ad.mse((out["phi"] - ad.Tensor(pref)) * ad.Tensor(1.0 / pref))
            terms = {g: ad.Tensor(0.0) for g in GROUPS}
        else:
            ww = W_WORK if mode == "pinn" else 0.0
            L, terms = total_loss(R, w, work=aux["work"], w_work=ww)

        L.backward()
        opt.step(cur_lr)

        if it % log_every == 0 or it == 1:
            print("[%s] it %5d L=%9.3e  mom=%8.2e melt=%8.2e work=%+.3f "
                  "[%.0fs]" % (tag, it, float(L.v), float(terms["mom"].v),
                               float(terms["melt"].v), float(aux["work"].v),
                               elapsed + time.time() - t0), flush=True)
            hist.append([it, float(L.v)])

    done = it if it >= iters else it - 1
    elapsed += time.time() - t0
    np.savez_compressed(ck, *[p.v for p in params], it=done, elapsed=elapsed,
                        hist=np.array(hist) if hist else np.zeros((0, 2)))
    if done >= iters:
        return finish(tag, params, bg, geo, mode, width, depth, iters,
                      elapsed, hist)
    print("PROGRESS %s %d/%d  (%.0f s so far)" % (tag, done, iters, elapsed))
    return None


def finish(tag, params, bg, geo, mode, width, depth, iters, elapsed, hist):
    d = diagnose(params, bg, geo, eval_points(bg))
    rec = dict(tag=tag, mode=mode, width=width, depth=depth, iters=iters,
               n_params=n_params(width, depth), seconds=round(elapsed, 1), **d)
    with open(os.path.join(RES, "%s.json" % tag), "w") as f:
        json.dump(rec, f, indent=1)
    print("DONE %s" % tag)
    print(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="pinn",
                    choices=["pinn", "nowork", "dataonly"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--budget", type=float, default=0)
    a = ap.parse_args()
    run(tag=a.tag, mode=a.mode, width=a.width, depth=a.depth,
        iters=a.iters, budget=a.budget or None)
