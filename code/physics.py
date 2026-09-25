"""
Physical closures for the CVL PINN, ported from the ASPECT plugins in this
project so that the network solves the same system the ASPECT model defines.

Sources:
  katz2003_mantle_melting.cc  -> melt_fraction()          (Katz et al. 2003)
  visco_plastic.cc + prm      -> viscosity()              (Hirth & Kohlstedt 2004)
  potential_temperature.cc    -> temperature_field()      (Vs -> T conversion)
  original.prm                -> all parameter values

Every routine is written with numpy ufuncs only, so it works both on plain
arrays (for field construction / plotting) and on the autodiff Tensors used
inside the PINN residuals (see autodiff.py).
"""
import numpy as np

# ------------------------------------------------------------------ geometry
R_OUTER = 6378137.0
R_INNER = 5718137.0
LON_MIN, LON_MAX = 2.5, 18.0
LAT_MIN, LAT_MAX = -4.0, 12.5
D_MAX = R_OUTER - R_INNER            # 660 km

# --------------------------------------------------- thermodynamic constants
GRAVITY = 9.81
R_GAS = 8.314462618
RHO_S = 3300.0                       # Reference solid density        [prm]
RHO_F = 2600.0                       # Reference melt density         [prm]
DRHO = RHO_S - RHO_F
ALPHA = 3.0e-5                       # Thermal expansion coefficient  [prm]
CP = 1250.0                          # Reference specific heat        [prm]
K_THERM = 4.7                        # Thermal conductivity           [prm]
T_POTENTIAL = 1573.0                 # Mantle potential temperature   [prm]
T_SURFACE = 293.0                    # plugin default
VS_TO_DENSITY = 0.20                 # Vs to density                  [prm]

# ----------------------------------------------------------- melt transport
MU_F = 10.0                          # Reference melt viscosity       [prm]
K0 = 1.0e-10                         # Reference permeability         [prm]
XI_REF = 1.0e22                      # Reference bulk viscosity       [prm]
ALPHA_PHI = 27.0                     # Exponential melt weakening     [prm]
PHI_MIN = 1.0e-6

# ------------------------------------------------------- Katz 2003 constants
A1, A2, A3 = 1085.7, 1.329e-7, -5.1e-18
B1, B2, B3 = 1475.0, 8.0e-8, -3.2e-18
C1, C2, C3 = 1780.0, 4.50e-8, -2.0e-18
R1, R2 = 0.5, 8e-11
BETA = 1.5
M_CPX = 0.15

# -------------------------------------------------------- rheology constants
A_DIFF, N_DIFF, E_DIFF, V_DIFF = 4.5e-15, 1.0, 375e3, 6e-6
A_DISL, N_DISL, E_DISL, V_DISL = 7.4e-15, 3.5, 530e3, 14e-6
GRAIN_SIZE, M_GRAIN = 0.01, 3.0
ETA_MIN, ETA_MAX = 1.0e19, 1.0e24


# ============================================================ background state
def lithostatic_pressure(depth):
    """Adiabatic/lithostatic reference pressure [Pa]."""
    return RHO_S * GRAVITY * depth


def adiabat(depth):
    """Adiabatic temperature profile used by potential_temperature.cc."""
    return T_POTENTIAL * np.exp(GRAVITY * ALPHA * depth / CP)


# =================================================================== melting
def melt_fraction(T, P):
    """Anhydrous equilibrium melt fraction, Katz et al. (2003).

    Exact port of Katz2003MantleMelting<dim>::melt_fraction.
    """
    T_sol = A1 + 273.15 + A2 * P + A3 * P * P
    T_lherz = B1 + 273.15 + B2 * P + B3 * P * P
    T_liq = C1 + 273.15 + C2 * P + C3 * P * P

    x = np.clip((T - T_sol) / (T_lherz - T_sol), 0.0, 1.0)
    F = x ** BETA
    F = np.where(T > T_lherz, 1.0, F)

    F_max = M_CPX / (R1 + R2 * np.maximum(P, 0.0))
    T_max = F_max ** (1.0 / BETA) * (T_lherz - T_sol) + T_sol
    y = np.clip((T - T_max) / np.maximum(T_liq - T_max, 1.0), 0.0, 1.0)
    F_cpxout = F_max + (1.0 - F_max) * y ** BETA
    F = np.where((F > F_max) & (T < T_liq), F_cpxout, F)

    F = np.where((T < T_sol) | (P > 1.3e10), 0.0, F)
    return np.clip(F, 0.0, 1.0)


def solidus(P):
    return A1 + 273.15 + A2 * P + A3 * P * P


# ==================================================== hydrous melting (Katz 2003)
# Katz, Spiegelman & Langmuir (2003) is a HYDROUS parameterisation.  The
# anhydrous relations above are its X_H2O = 0 limit.  Water enters as a
# depression of all three characteristic temperatures,
#     dT(X_H2O) = K * X_H2O ** GAMMA_H2O,
# where X_H2O is the water content OF THE MELT in wt %.  Water behaves as an
# incompatible trace element with a constant bulk partition coefficient, so
# for a source of bulk water content X_bulk the melt carries
#     X_melt = X_bulk / (D_H2O + F (1 - D_H2O)),
# capped at the saturation value X_sat(P).  F and X_melt are therefore coupled
# and we solve them together by fixed-point iteration.
K_H2O = 43.0            # deg C / (wt %)**GAMMA        Katz et al. (2003)
GAMMA_H2O = 0.75        # exponent of the solidus depression
D_H2O = 0.01            # bulk partition coefficient, melt/solid
CHI1_H2O = 12.00        # wt % / GPa**LAMBDA_H2O       saturation
CHI2_H2O = 1.00         # wt % / GPa
LAMBDA_H2O = 0.60

X_BULK_H2O = 0.02       # wt % bulk water in the source (200 ppm)


def water_saturation(P):
    """Saturation water content of the melt [wt %]; P in Pa."""
    Pg = np.maximum(P, 0.0) / 1.0e9
    return CHI1_H2O * Pg ** LAMBDA_H2O + CHI2_H2O * Pg


def solidus_depression(X_melt):
    """dT [K] by which water at X_melt wt % lowers the melting temperatures."""
    return K_H2O * np.maximum(X_melt, 0.0) ** GAMMA_H2O


def _katz_F(T, P, dTw):
    """Katz melt fraction with all three temperatures lowered by dTw."""
    T_sol = A1 + 273.15 + A2 * P + A3 * P * P - dTw
    T_lherz = B1 + 273.15 + B2 * P + B3 * P * P - dTw
    T_liq = C1 + 273.15 + C2 * P + C3 * P * P - dTw

    x = np.clip((T - T_sol) / np.maximum(T_lherz - T_sol, 1.0), 0.0, 1.0)
    F = x ** BETA
    F = np.where(T > T_lherz, 1.0, F)

    F_max = M_CPX / (R1 + R2 * np.maximum(P, 0.0))
    T_max = F_max ** (1.0 / BETA) * (T_lherz - T_sol) + T_sol
    y = np.clip((T - T_max) / np.maximum(T_liq - T_max, 1.0), 0.0, 1.0)
    F_cpxout = F_max + (1.0 - F_max) * y ** BETA
    F = np.where((F > F_max) & (T < T_liq), F_cpxout, F)

    F = np.where((T < T_sol) | (P > 1.3e10), 0.0, F)
    return np.clip(F, 0.0, 1.0)


def _melt_fraction_hydrous_exact(T, P, X_bulk=None, n_iter=60):
    """Hydrous equilibrium melt fraction, Katz et al. (2003), solved exactly.

    The water content of the melt depends on F and the solidus depression
    depends on that water content, so F satisfies the implicit equation
        F = F_Katz(T, P, dT(X_melt(F))).
    The right-hand side decreases monotonically in F, because a larger melt
    fraction dilutes the water and raises the solidus back up, so the root is
    unique and we obtain it by bisection.  Reduces to melt_fraction() exactly
    when X_bulk = 0.
    """
    X_bulk = X_BULK_H2O if X_bulk is None else X_bulk
    if X_bulk <= 0.0:
        return melt_fraction(T, P)
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    X_sat = water_saturation(P)

    def rhs(F):
        X_melt = np.minimum(X_bulk / (D_H2O + F * (1.0 - D_H2O)), X_sat)
        return _katz_F(T, P, solidus_depression(X_melt))

    lo = np.zeros(np.broadcast(T, P).shape)
    hi = np.ones_like(lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        # rhs(F) - F is monotonically decreasing, so the root lies above mid
        # wherever rhs(mid) still exceeds mid
        up = rhs(mid) > mid
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.clip(0.5 * (lo + hi), 0.0, 1.0)


# --- fast lookup for the hydrous melt fraction -------------------------------
# The coupled (F, X_melt) solve is far too slow to run at every point of a
# 60-step bisection over a 1.7 million point grid, so we tabulate F on a fine
# (T, P) mesh once per water content and interpolate.  The table is built with
# the exact iteration above; its spacing is ~1.5 K and ~0.07 GPa, which is far
# finer than any structure in F.
_HYD_CACHE = {}
_XI_LO, _XI_HI, _NXI = -20.0, 900.0, 2400
_PP_LO, _PP_HI, _NP = 0.0, 2.2e10, 340


def _hydrous_table(X_bulk):
    """F tabulated against (T - wet solidus, P) so the onset is grid-aligned."""
    key = round(float(X_bulk), 8)
    if key not in _HYD_CACHE:
        xg = np.linspace(_XI_LO, _XI_HI, _NXI)
        Pg = np.linspace(_PP_LO, _PP_HI, _NP)
        XX, PP = np.meshgrid(xg, Pg, indexing="ij")
        TT = XX + _hydrous_solidus_raw(PP, key)
        _HYD_CACHE[key] = (xg, Pg, _melt_fraction_hydrous_exact(TT, PP, key))
    return _HYD_CACHE[key]


def melt_fraction_hydrous(T, P, X_bulk=None, exact=False):
    """Hydrous equilibrium melt fraction, Katz et al. (2003).

    Reduces to melt_fraction() exactly when X_bulk = 0.  Uses a tabulated
    interpolant in (T - wet solidus, P) unless exact=True.
    """
    X_bulk = X_BULK_H2O if X_bulk is None else X_bulk
    if X_bulk <= 0.0:
        return melt_fraction(T, P)
    if exact:
        return _melt_fraction_hydrous_exact(T, P, X_bulk)
    xg, Pg, tab = _hydrous_table(X_bulk)
    T = np.asarray(T, dtype=float)
    P = np.clip(np.asarray(P, dtype=float), _PP_LO, _PP_HI)
    xi = np.clip(T - _hydrous_solidus_raw(P, X_bulk), xg[0], xg[-1])
    dx = xg[1] - xg[0]
    dp = Pg[1] - Pg[0]
    fi = (xi - xg[0]) / dx
    fj = (P - Pg[0]) / dp
    i = np.clip(fi.astype(np.int64), 0, len(xg) - 2)
    j = np.clip(fj.astype(np.int64), 0, len(Pg) - 2)
    a = fi - i
    b = fj - j
    F = ((1 - a) * (1 - b) * tab[i, j] + a * (1 - b) * tab[i + 1, j]
         + (1 - a) * b * tab[i, j + 1] + a * b * tab[i + 1, j + 1])
    F = np.where(np.asarray(P) > 1.3e10, 0.0, F)
    return np.clip(F, 0.0, 1.0)


def _hydrous_solidus_raw(P, X_bulk):
    dry = solidus(P)
    if X_bulk <= 0.0:
        return dry
    X_melt = np.minimum(X_bulk / D_H2O, water_saturation(P))
    return dry - solidus_depression(X_melt)


def hydrous_solidus(P, X_bulk=None):
    """Wet solidus [K]: the temperature at which F first exceeds zero.

    At vanishing melt fraction the melt carries X_bulk / D_H2O, capped at
    saturation, so the depression takes its largest value there.
    """
    X_bulk = X_BULK_H2O if X_bulk is None else X_bulk
    return _hydrous_solidus_raw(np.asarray(P, dtype=float), X_bulk)


# ============================ self-consistent Vs -> (temperature, melt) ======
# The ASPECT plugin maps the whole velocity anomaly to temperature via
#   dlnVs = -(alpha / R_vs) dT,      R_vs = dln(rho)/dln(Vs) = 0.20
# giving 1 % Vs = 67 K.  Where the mantle is partially molten this
# over-predicts temperature badly, because melt is a far more efficient
# reducer of Vs than temperature.  We therefore solve the partition
#   dlnVs = (dlnVs/dT) dT + (dlnVs/dphi) phi,   phi = F_Katz(T_ad + dT, P)
# for dT, which keeps the inferred state pinned near the solidus.
DLNVS_DT = -ALPHA / VS_TO_DENSITY        # -1.5e-4 K^-1, identical to plugin
DLNVS_DPHI = -2.5                        # melt sensitivity (tube geometry)


def invert_temperature_melt(dlnvs, depth, n_iter=60):
    """Bisection for the self-consistent temperature excess and melt fraction."""
    P = lithostatic_pressure(depth)
    T_ad = adiabat(depth)
    mf = melt_fraction_hydrous

    def residual(dT):
        return DLNVS_DT * dT + DLNVS_DPHI * mf(T_ad + dT, P) - dlnvs

    lo = np.full_like(np.asarray(dlnvs, dtype=float), -600.0)
    hi = np.full_like(lo, 1600.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        # residual is monotonically decreasing in dT
        neg = residual(mid) > 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    dT = 0.5 * (lo + hi)
    return dT, mf(T_ad + dT, P)


# ================================================================= rheology
def viscosity(T, P, edot, phi):
    """Composite diffusion + dislocation creep with melt weakening.

    Harmonic average, as set by 'Viscosity averaging scheme = harmonic'.
    """
    T = np.maximum(T, 300.0)
    edot = np.maximum(edot, 1.0e-20)

    eta_diff = (0.5 / A_DIFF
                * np.exp((E_DIFF + P * V_DIFF) / (R_GAS * T))
                * GRAIN_SIZE ** M_GRAIN)
    eta_disl = (0.5 * A_DISL ** (-1.0 / N_DISL)
                * np.exp((E_DISL + P * V_DISL) / (N_DISL * R_GAS * T))
                * edot ** ((1.0 - N_DISL) / N_DISL))

    eta = 1.0 / (1.0 / eta_diff + 1.0 / eta_disl)
    eta = eta * np.exp(-ALPHA_PHI * phi)
    return np.clip(eta, ETA_MIN, ETA_MAX)


def permeability(phi):
    """k = k0 * phi^3  (Katz2003MantleMelting reference_darcy_coefficient)."""
    return K0 * np.maximum(phi, 0.0) ** 3


def darcy_mobility(phi):
    """k(phi) / mu_f  [m^2 / (Pa s)]"""
    return permeability(phi) / MU_F


# =========================================================== temperature field
def temperature_field(depth, lab_depth, dlnvs, mode="selfconsistent"):
    """Temperature and equilibrium melt fraction of the CVL upper mantle.

    mode = "plugin"          exact port of PotentialTemperature<dim>, i.e. the
                             whole velocity anomaly is read as temperature.
    mode = "selfconsistent"  the anomaly is partitioned between temperature and
                             melt (recommended; see invert_temperature_melt).

    Above the LAB a linear conductive geotherm is used in both cases, exactly
    as in the ASPECT plugin.
    """
    lab_depth = np.maximum(lab_depth, 1.0)
    T_lab = T_POTENTIAL * np.exp(GRAVITY * ALPHA * lab_depth / CP)
    T_lith = T_SURFACE + (depth / lab_depth) * (T_lab - T_SURFACE)

    if mode == "plugin":
        dT = -1.0 / ALPHA * VS_TO_DENSITY * dlnvs
    else:
        dT, _ = invert_temperature_melt(dlnvs, depth)

    T_sub = adiabat(depth) + dT
    T = np.where(depth < lab_depth, T_lith, T_sub)
    T = np.maximum(T, 0.0)

    mf = melt_fraction_hydrous
    F = np.where(depth < lab_depth, 0.0, mf(T, lithostatic_pressure(depth)))
    return T, F
