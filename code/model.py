"""
Physics-informed neural network for steady two-phase (melt + solid) flow in the
asthenosphere beneath the Cameroon Volcanic Line.

Formulation
-----------
Mixed / first-order form.  The network outputs

    u   (3)  solid velocity                    [m/s]
    P   (1)  dynamic pressure (melt phase)     [Pa]
    pc  (1)  compaction pressure               [Pa]
    phi (1)  porosity = retained melt fraction [-]
    tau (6)  deviatoric stress                 [Pa]
    q   (3)  Darcy (segregation) flux          [m/s]

so that every residual contains only first derivatives:

 R1  d_j tau_ij - d_i(P + pc) + b_i                       = 0   momentum
 R2  tau_ij - 2 eta (e_ij - delta_ij div(u)/3)            = 0   rheology
 R3  div(u) + pc/xi - Gamma (1/rho_f - 1/rho_s)           = 0   solid mass
 R4  div(phi u) + div(q) - Gamma/rho_f                    = 0   melt mass
 R5  q + (k(phi)/mu)(grad P + drho g zhat)                = 0   Darcy

with buoyancy  b_z = -(rho_s alpha dT + phi drho) g,  b_x = b_y = 0, and the
melting rate from decompression through the equilibrium melt fraction,

 Gamma = rho_s [ max(u . grad F_eq, 0) - phi X_sub / tau_freeze ]

where X_sub = 1 where F_eq = 0 (sub-solidus, melt refreezes).

Boundary conditions are imposed *exactly* by construction (see `_constrain`),
matching original.prm: zero solid velocity on the inner, west, east, south and
north boundaries, free slip on the outer boundary, and no melt flux through
any boundary.

Viscosity and permeability are lagged (Picard), i.e. evaluated from the
current iterate without being differentiated.  This is the standard treatment
for the 5-order-of-magnitude viscosity contrast and keeps training stable.
"""
import numpy as np
import autodiff as ad
import physics as ph
import fields as fl

# ------------------------------------------------------------------- scales
# Scales are chosen so that the three momentum terms are all O(1):
#   P0 / L0  ~  rho_s alpha dT_char g   with dT_char = 150 K
L0 = fl.LZ                       # 660 km
ETA0 = 1.0e21
DT_CHAR = 150.0
B0 = ph.RHO_S * ph.ALPHA * DT_CHAR * ph.GRAVITY      # ~146 Pa/m
P0 = B0 * L0                                          # ~9.6e7 Pa
# The naive Stokes scale P0 L0 / eta0 (2 m/yr) overestimates the response by
# ~60x because the domain is closed (no slip on five faces) and the lid is
# 1e24 Pa s.  Residuals normalised by that scale look small while the relative
# error is 60x larger, so U0 is set from the realised flow instead.
U0 = 1.0e-9                                           # 3.15 cm/yr
PHI0 = 0.01
SPAN = 2.5                       # porosity correction range: x0.082 .. x12.2

# ------------------------------------------------------------- rigid lid
# The lithosphere is removed from the flow problem by a hard mask, exactly as
# the domain boundaries are.  With psi = z - z_LAB(x,y) the signed distance to
# the LAB (positive in the asthenosphere) and delta the transition thickness,
#     w(psi) = tanh^2( max(psi,0) / delta )
# is identically zero in the lid, has zero first derivative there, and rises to
# 1 below it.  Two variants are provided:
#     "locked" : all three velocity components carry w  -- the lid cannot move
#     "slip"   : only the vertical component carries w  -- the lid cannot rise
#                or sink but may translate horizontally, and the asthenosphere
#                shears freely up to the LAB
#     "none"   : the original model, lithosphere included in the flow
LID_MODE = "none"
LID_DELTA = 15.0e3


def set_lid(mode, delta=None):
    global LID_MODE, LID_DELTA
    assert mode in ("none", "locked", "slip"), mode
    LID_MODE = mode
    if delta:
        LID_DELTA = delta
    return LID_MODE


def lid_mask(z, lab, dlabx, dlaby):
    """w and its three spatial derivatives at the sample points."""
    if LID_MODE == "none":
        one = np.ones_like(z)
        return one, [np.zeros_like(z)] * 3
    psi = z - lab
    s = np.maximum(psi, 0.0) / LID_DELTA
    th = np.tanh(s)
    w = th * th
    # dw/dpsi = 2 tanh(s) sech^2(s) / delta, and zero where psi <= 0
    dwdpsi = np.where(psi > 0.0, 2.0 * th * (1.0 - th * th) / LID_DELTA, 0.0)
    # grad(psi) = (-dz_LAB/dx, -dz_LAB/dy, 1)
    dw = [dwdpsi * (-dlabx), dwdpsi * (-dlaby), dwdpsi]
    return w, dw
K0_REF = 1.0e-7                  # reference permeability actually in use
Q0 = K0_REF * PHI0 ** 3 / ph.MU_F * ph.DRHO * ph.GRAVITY
# Refreezing time for melt advected into sub-solidus mantle.  Melt segregates
# at ~1 m/yr, so this sets the distance (~30 km) over which melt leaving the
# melting column is reabsorbed; 1 Myr lets melt cross the whole domain first.
TAU_FREEZE = 3.0e4 * 3.15576e7   # 30 kyr


def set_permeability(k0):
    """Keep the Darcy-flux scale consistent with the permeability in use."""
    global K0_REF, Q0
    K0_REF = k0
    Q0 = k0 * PHI0 ** 3 / ph.MU_F * ph.DRHO * ph.GRAVITY
    return Q0

N_OUT = 15


# ============================================================ network
def init_params(n_in, width, depth, seed=0):
    rng = np.random.default_rng(seed)
    dims = [n_in] + [width] * depth + [N_OUT]
    params = []
    for i in range(len(dims) - 1):
        s = np.sqrt(1.0 / dims[i])
        W = ad.Tensor(rng.normal(0.0, s, (dims[i], dims[i + 1])), requires_grad=True)
        b = ad.Tensor(np.zeros((1, dims[i + 1])), requires_grad=True)
        params += [W, b]
    return params


def mlp_with_jac(params, x0, J0):
    """Forward pass returning outputs and d(out)/dx_k for k = 0,1,2.

    x0 : (N, n_in) numpy array of network inputs
    J0 : list of three (N, n_in) numpy arrays, d(input)/d(x_k) in metres^-1
    """
    z = x0
    J = list(J0)
    n_layer = len(params) // 2
    for li in range(n_layer):
        W, b = params[2 * li], params[2 * li + 1]
        A = [ad.matmul(Jk, W) for Jk in J]
        if li < n_layer - 1:
            zt = ad.tanh(ad.matmul(z, W) + b)
            sech2 = 1.0 - zt * zt                      # fully differentiable
            J = [sech2 * Ak for Ak in A]
            z = zt
        else:
            z = ad.matmul(z, W) + b
            J = A
    return z, J


# ==================================================== hard BC constraint layer
class Geometry:
    def __init__(self):
        self.Lx, self.Ly, self.Lz = fl.LX, fl.LY, fl.LZ

    def norm(self, x, y, z):
        return (2.0 * x / self.Lx - 1.0, 2.0 * y / self.Ly - 1.0, z / self.Lz)

    def masks(self, x, y, z):
        Xn, Yn, d = self.norm(x, y, z)
        sx, sy = 1.0 - Xn ** 2, 1.0 - Yn ** 2
        dsx, dsy = -2.0 * Xn * 2.0 / self.Lx, -2.0 * Yn * 2.0 / self.Ly
        Bh = sx * sy * (1.0 - d)
        Bv = sx * sy * 4.0 * d * (1.0 - d)
        dBh = [dsx * sy * (1.0 - d), sx * dsy * (1.0 - d), -sx * sy / self.Lz]
        dBv = [dsx * sy * 4 * d * (1 - d), sx * dsy * 4 * d * (1 - d),
               sx * sy * 4 * (1 - 2 * d) / self.Lz]
        Mq = [sx, sy, 4.0 * d * (1.0 - d)]
        dMq = [[dsx, 0.0 * dsx, 0.0 * dsx],
               [0.0 * dsy, dsy, 0.0 * dsy],
               [0.0 * d, 0.0 * d, 4.0 * (1 - 2 * d) / self.Lz]]
        return Bh, Bv, dBh, dBv, Mq, dMq


def _col(t, i):
    return t[:, i:i + 1]


def apply_constraints(raw, draw, geo, x, y, z, pref=None, dlp=None, lid=None):
    """Map raw network outputs to physical fields obeying the BCs exactly."""
    Bh, Bv, dBh, dBv, Mq, dMq = geo.masks(x, y, z)
    if pref is None:
        pref = np.full(len(x), 1.0e-3)
        dlp = [np.zeros(len(x))] * 3
    if lid is None:
        lid = (np.ones_like(x), [np.zeros_like(x)] * 3)
    w, dw = lid

    out, dout = {}, {}

    # --- solid velocity ------------------------------------------------
    # The lid mask multiplies the box mask; which components carry it is set
    # by LID_MODE (see lid_mask).
    if LID_MODE == "locked":
        carries = (True, True, True)
    elif LID_MODE == "slip":
        carries = (False, False, True)
    else:
        carries = (False, False, False)

    u, du = [], [[], [], []]
    for i, (B, dB) in enumerate(((Bh, dBh), (Bh, dBh), (Bv, dBv))):
        r = _col(raw, i)
        if carries[i]:
            M = B * w
            dM = [dB[k] * w + B * dw[k] for k in range(3)]
        else:
            M, dM = B, dB
        ui = ad.Tensor(M[:, None] * U0) * r
        u.append(ui)
        for k in range(3):
            du[k].append(ad.Tensor(dM[k][:, None] * U0) * r
                         + ad.Tensor(M[:, None] * U0) * _col(draw[k], i))
    out["u"], dout["u"] = u, du

    # --- pressures ------------------------------------------------------
    for name, idx in (("P", 3), ("pc", 4)):
        out[name] = ad.Tensor(P0) * _col(raw, idx)
        dout[name] = [ad.Tensor(P0) * _col(draw[k], idx) for k in range(3)]

    # --- porosity ---------------------------------------------------------
    # phi = phi_ref(x) * exp(SPAN tanh(g)):  a bounded multiplicative
    # correction to the analytic melt-retention porosity.  Learning phi
    # directly fails because k ~ phi^3 gives vanishing gradients near zero.
    g = _col(raw, 5)
    t = ad.tanh(g)
    phi = ad.exp(t * SPAN) * pref[:, None]
    out["phi"] = phi
    dout["phi"] = [phi * ((1.0 - t * t) * SPAN * _col(draw[k], 5)
                          + dlp[k][:, None]) for k in range(3)]

    # --- deviatoric stress ----------------------------------------------
    tau, dtau = [], [[], [], []]
    for j in range(6):
        tau.append(ad.Tensor(P0) * _col(raw, 6 + j))
        for k in range(3):
            dtau[k].append(ad.Tensor(P0) * _col(draw[k], 6 + j))
    out["tau"], dout["tau"] = tau, dtau

    # --- Darcy flux ------------------------------------------------------
    q, dq = [], [[], [], []]
    for i in range(3):
        r = _col(raw, 12 + i)
        M = Mq[i][:, None] if np.ndim(Mq[i]) else Mq[i]
        qi = ad.Tensor(M * Q0) * r
        q.append(qi)
        for k in range(3):
            dm = dMq[i][k]
            dm = dm[:, None] if np.ndim(dm) else np.zeros_like(M)
            dq[k].append(ad.Tensor(dm * Q0) * r + ad.Tensor(M * Q0) * _col(draw[k], i + 12))
    out["q"], dout["q"] = q, dq
    return out, dout


# =================================================================== residuals
IDX = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def residuals(out, dout, bg_s, k0, freeze=True, visc_pow=1.0,
              work_mask=None):
    """Return a dict of scaled PDE residual Tensors."""
    u, du = out["u"], dout["u"]
    tau, dtau = out["tau"], dout["tau"]
    q, dq = out["q"], dout["q"]
    phi, dphi = out["phi"], dout["phi"]
    P, dP = out["P"], dout["P"]
    pc, dpc = out["pc"], dout["pc"]

    dT = bg_s["dT"][:, None]
    dF = [bg_s["dFdx"][:, None], bg_s["dFdy"][:, None], bg_s["dFdz"][:, None]]
    Fe = bg_s["F"][:, None]

    # ---- strain rate and lagged coefficients ---------------------------
    eps = [[0.5 * (du[j][i].v + du[i][j].v) for j in range(3)] for i in range(3)]
    divu_v = du[0][0].v + du[1][1].v + du[2][2].v
    e2 = 0.0
    for i in range(3):
        for j in range(3):
            dev = eps[i][j] - (divu_v / 3.0 if i == j else 0.0)
            e2 = e2 + 0.5 * dev * dev
    edot = np.sqrt(np.maximum(e2, 1e-40))
    Tv, Pv = bg_s["T"][:, None], bg_s["Plith"][:, None]
    eta = ph.viscosity(Tv, Pv, edot, phi.v)
    if visc_pow != 1.0:                     # curriculum: compress the contrast
        eta = ETA0 * (eta / ETA0) ** visc_pow
    kmob = k0 * np.maximum(phi.v, ph.PHI_MIN) ** 3 / ph.MU_F
    xi = ph.XI_REF

    # ---- melting rate (differentiable; the max() mask is held fixed) -----
    prod_v = u[0].v * dF[0] + u[1].v * dF[1] + u[2].v * dF[2]
    mask = (prod_v > 0.0)
    prod = ad.where(mask, u[0] * dF[0] + u[1] * dF[1] + u[2] * dF[2], 0.0)
    sub = (Fe <= 1e-7).astype(float) if freeze else np.zeros_like(Fe)
    Gam = (prod - phi * (sub / TAU_FREEZE)) * ph.RHO_S
    Gam_v = ph.RHO_S * (np.maximum(prod_v, 0.0) - phi.v * sub / TAU_FREEZE)

    R = {}

    # ---- R1 momentum ----------------------------------------------------
    tmap = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (1, 0): 3,
            (0, 2): 4, (2, 0): 4, (1, 2): 5, (2, 1): 5}
    bz = (phi * ph.DRHO + (ph.RHO_S * ph.ALPHA * dT)) * (-ph.GRAVITY)
    mom = []
    for i in range(3):
        acc = dtau[0][tmap[(i, 0)]] + dtau[1][tmap[(i, 1)]] + dtau[2][tmap[(i, 2)]]
        acc = acc - dP[i] - dpc[i]
        if i == 2:
            acc = acc + bz
        mom.append(acc * (L0 / P0))
    R["mom"] = mom

    # ---- energy admissibility -------------------------------------------
    # For steady Stokes flow driven only by buoyancy, the work done by the
    # buoyancy force equals the viscous dissipation and is therefore strictly
    # positive.  A network that minimises the momentum residual by letting the
    # pressure absorb the buoyancy can settle on a branch in which the
    # velocity is anti-correlated with the density anomaly; that branch is
    # excluded by requiring the batch estimate of the work integral to be
    # positive.  Both components carry the same sign convention (z positive
    # downwards), so the integrand is b_z u_z.
    # The collocation points are deliberately biased toward the melting
    # column, so the work integral is estimated only over the uniformly drawn
    # part of the batch, selected by work_mask.
    m = (np.ones_like(bz.v) if work_mask is None
         else np.asarray(work_mask, dtype=float).reshape(bz.v.shape))
    _den = float(np.sum(np.abs(bz.v * u[2].v) * m)) + 1e-300
    work = (bz * u[2] * ad.Tensor(m)).sum() * (1.0 / _den)

    # ---- R2 rheology -----------------------------------------------------
    con = []
    for n, (i, j) in enumerate(IDX):
        e_ij = (du[j][i] + du[i][j]) * 0.5
        if i == j:
            e_ij = e_ij - (du[0][0] + du[1][1] + du[2][2]) * (1.0 / 3.0)
        con.append((tau[n] - ad.Tensor(2.0 * eta) * e_ij) * (1.0 / P0))
    R["rheo"] = con

    # ---- R3 solid mass ---------------------------------------------------
    divu = du[0][0] + du[1][1] + du[2][2]
    R["solid"] = [(divu + pc * (1.0 / xi)
                   - Gam * (1.0 / ph.RHO_F - 1.0 / ph.RHO_S)) * (L0 / U0)]

    # ---- R4 melt mass ----------------------------------------------------
    div_phiu = (dphi[0] * u[0] + phi * du[0][0]
                + dphi[1] * u[1] + phi * du[1][1]
                + dphi[2] * u[2] + phi * du[2][2])
    divq = dq[0][0] + dq[1][1] + dq[2][2]
    R["melt"] = [(div_phiu + divq - Gam * (1.0 / ph.RHO_F)) * (L0 / Q0)]

    # ---- R5 Darcy --------------------------------------------------------
    dar = []
    for i in range(3):
        g = ph.DRHO * ph.GRAVITY if i == 2 else 0.0
        dar.append((q[i] + ad.Tensor(kmob) * (dP[i] + g)) * (1.0 / Q0))
    R["darcy"] = dar

    return R, dict(eta=eta, edot=edot, Gamma=Gam_v, kmob=kmob,
                   work=work)


# =================================================================== sampling
def sample_points(bg, n, rng, shallow_frac=0.45, melt_frac=0.3):
    """Collocation points, strongly oversampled in the melting region.

    The melting column occupies only ~2 % of the domain volume, so uniform
    sampling leaves the melt-transport equations essentially unconstrained
    exactly where they matter.
    """
    n_m = int(n * melt_frac)
    n_s = int(n * shallow_frac)
    n_u = n - n_m - n_s

    xm, ym, zm = bg.sample_melt(n_m, rng)
    x = np.concatenate([xm, rng.uniform(0, fl.LX, n_s + n_u)])
    y = np.concatenate([ym, rng.uniform(0, fl.LY, n_s + n_u)])
    z = np.concatenate([zm,
                        rng.uniform(0.0, 300e3, n_s),
                        rng.uniform(0.0, fl.LZ, n_u)])
    return x, y, z


def build_inputs(bg, x, y, z, geo):
    """Network inputs and their spatial derivatives."""
    s = bg.sample(x, y, z, names=("T", "dT", "F", "dFdx", "dFdy", "dFdz",
                                  "phi_ref", "dlpx", "dlpy", "dlpz",
                                  "lab", "dlabx", "dlaby"))
    s["lid"] = lid_mask(z, s["lab"], s["dlabx"], s["dlaby"])
    gT = [fl.trilinear(g, bg.x, bg.y, bg.z, x, y, z) for g in bg.gradT]
    Xn, Yn, d = geo.norm(x, y, z)
    F_h, T_h = s["F"] / PHI0, (s["T"] - 1600.0) / 300.0
    s["pref"] = s["phi_ref"]
    s["dlp"] = [s["dlpx"], s["dlpy"], s["dlpz"]]

    x0 = np.stack([Xn, Yn, d, F_h, T_h], axis=1)
    J0 = []
    for k, (sc, gF, gTk) in enumerate(zip(
            (2.0 / fl.LX, 2.0 / fl.LY, 1.0 / fl.LZ),
            (s["dFdx"], s["dFdy"], s["dFdz"]), gT)):
        J = np.zeros((len(x), 5))
        J[:, k] = sc
        J[:, 3] = gF / PHI0
        J[:, 4] = gTk / 300.0
        J0.append(J)
    s["Plith"] = ph.lithostatic_pressure(z)
    return x0, J0, s
