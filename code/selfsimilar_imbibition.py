"""
Self-similar non-isothermal capillary imbibition with an adsorbing additive.

Implements the scheme of Section 4 / Appendix D of the manuscript
"Non-Equilibrium Two-Phase Filtration with Adsorption and Heat Transfer in
Porous Media" (A.A. Mussina):

    s_sigma = (F P_xi)_xi + xi/2 s_xi
    A c_sigma = (xi/2 A - u) c_xi,            A = s + Gamma_a/m,  u = -F P_xi
    (s+beta) T_sigma = kappa T_xixi + (xi/2 (s+beta) - u) T_xi,   kappa = varkappa/alpha^2

on a uniform grid xi_i = i*dxi, i = 0..N, xi_max = 30, with central
differences for the capillary/conductive terms and one-sided (upwind)
differences for all first-order terms (eqs. D1-D3).  The stationary discrete
equations are solved by a linearised implicit pseudo-time iteration
(coefficients frozen at the previous iterate, one tridiagonal solve per
unknown), which has the same fixed point as the explicit scheme.

Usage:
    python selfsimilar_imbibition.py                  # grid study, parameters of Table 2
    python selfsimilar_imbibition.py --lam 2.5        # thermal conductivity in W/(m C)
"""
import argparse
import numpy as np
from scipy.linalg import solve_banded


# --------------------------------------------------------------------------
# parameters (Table 2 of the manuscript)
# --------------------------------------------------------------------------
class Params:
    def __init__(self, lam=36.0):
        self.m = 0.2                # porosity
        self.Gamma_a = 0.1          # adsorption constant, a(c) = Gamma_a c
        self.n_a = 1.0              # exponent in f1
        self.mu_ratio0 = 5.0        # mu_2^0 / mu_1
        self.gamma_mu = 1.0         # mu_2 = mu_2^0 exp(-gamma_mu T), T scaled by T^0
        self.s_inj = 0.75           # s^0
        self.s_init = 0.25          # s_0
        self.c_inj = 1.0            # c^0
        self.T_inj = 1.0            # T^0
        self.xi_max = 30.0
        # thermal parameters (dimensional -> scaled)
        C1, C2, Cr = 1000e3, 300e3, 2000e3      # J/(m^3 C)
        alpha2 = 1e-5                           # m^2/s  (K = 1 D, mu_1 = 1 mPa s, E = 10 kPa)
        self.lam = lam                          # W/(m C)
        self.beta = (self.m * C2 + (1 - self.m) * Cr) / (self.m * (C1 - C2))
        self.kappa = lam / (self.m * (C1 - C2)) / alpha2   # varkappa / alpha^2


# --------------------------------------------------------------------------
# closure relations
# --------------------------------------------------------------------------
def f1(s, c, p):
    return ((s - 0.2) / 0.8) ** 3 * (1.0 / (1.0 + p.Gamma_a * c)) ** p.n_a


def f2(s):
    return ((0.8 - s) / 0.6) ** 3


def Fcoef(s, c, T, p):
    mu = p.mu_ratio0 * np.exp(-p.gamma_mu * T)          # mu = mu_2 / mu_1
    a, b = f1(s, c, p), f2(s)
    return -a * b / (p.m * (mu * a + b))


def Pcap(s):
    return 1.0 / (s - 0.2) ** 2                          # P = P_k / E


def dPcap(s):
    return -2.0 / (s - 0.2) ** 3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system; lower[0] and upper[-1] are ignored."""
    n = len(diag)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def fluxes(s, c, T, d, p):
    """Half-node capillary conductance K_{i+1/2} (>0) and fluxes G_{i+1/2}."""
    Fn = Fcoef(s, c, T, p)
    Fh = 0.5 * (Fn[1:] + Fn[:-1])                       # F_{i+1/2}: half-sum of nodal values
    ds = s[1:] - s[:-1]
    dP = Pcap(s[1:]) - Pcap(s[:-1])
    chord = np.where(np.abs(ds) > 1e-14, dP / np.where(ds == 0, 1, ds),
                     dPcap(0.5 * (s[1:] + s[:-1])))
    K = Fh * chord                                      # F * dP/ds  (> 0)
    G = Fh * dP / d                                     # G_{i+1/2}
    return K, G


def velocity(G):
    """u_i = -(G_{i-1/2} + G_{i+1/2}) / 2 at interior nodes (zero-padded ends)."""
    u = np.zeros(len(G) + 1)
    u[1:-1] = -0.5 * (G[:-1] + G[1:])
    u[0], u[-1] = -G[0], -G[-1]
    return u


def concentration(W, p):
    """Steady state of the upwind advection equation for c.

    Where W_i < 0 the one-sided difference is taken to the left (c_i = c_{i-1}),
    where W_i >= 0 to the right (c_i = c_{i+1}).  Starting from c_0 = c^0 and
    c_N = 0 this gives a step between the last node with W < 0 reached from
    the left and the next node.
    """
    N = len(W) - 1
    c = np.zeros(N + 1)
    c[0] = p.c_inj
    i = 1
    while i < N and W[i] < 0:
        c[i] = p.c_inj
        i += 1
    return c


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def solve(N, p, tol=1e-10, maxit=200000, dsig=None, verbose=False):
    d = p.xi_max / N
    xi = np.arange(N + 1) * d

    # initial guess: undisturbed reservoir with injection values at xi = 0
    s = p.s_init + (p.s_inj - p.s_init) * np.exp(-xi)
    T = p.T_inj * np.exp(-xi / 3.0)
    c = np.where(xi < 1.0, p.c_inj, 0.0)
    if dsig is None:
        dsig = 0.5 * d                                   # pseudo-time step of the implicit iteration

    for it in range(maxit):
        K, G = fluxes(s, c, T, d, p)
        u = velocity(G)

        # ---- saturation: (K+ (s_{i+1}-s_i) - K- (s_i-s_{i-1}))/d^2 + xi/2 (s_{i+1}-s_i)/d
        Km, Kp = K[:-1], K[1:]
        lo = np.zeros(N + 1); di = np.ones(N + 1); up = np.zeros(N + 1); rhs = np.zeros(N + 1)
        lo[1:-1] = -dsig * Km / d**2
        up[1:-1] = -dsig * (Kp / d**2 + xi[1:-1] / (2 * d))
        di[1:-1] = 1 + dsig * ((Kp + Km) / d**2 + xi[1:-1] / (2 * d))
        rhs[1:-1] = s[1:-1]
        rhs[0], rhs[-1] = p.s_inj, p.s_init
        s_new = tridiag(lo, di, up, rhs)
        s_new = np.clip(s_new, 0.2 + 1e-6, 0.8 - 1e-6)
        ii = slice(1, N)

        # ---- concentration: c_sigma = W c_xi, upwind by the sign of W = xi/2 - u/A.
        # c is advanced as a continuous variable (as in the explicit scheme D2):
        # at a node where W ~ 0 the steady state may carry an intermediate value.
        A = s_new + p.Gamma_a / p.m
        W = xi / 2 - u / A
        Wp = np.maximum(W[ii], 0.0)
        Wm = np.minimum(W[ii], 0.0)
        lo = np.zeros(N + 1); di = np.ones(N + 1); up = np.zeros(N + 1); rhs = np.zeros(N + 1)
        lo[ii] = dsig * Wm / d
        up[ii] = -dsig * Wp / d
        di[ii] = 1 + dsig * (Wp - Wm) / d
        rhs[ii] = c[ii]
        rhs[0], rhs[-1] = p.c_inj, 0.0
        c_new = tridiag(lo, di, up, rhs)

        # ---- temperature: kappa/(s+beta) T_xixi + V T_xi, upwind first-order term
        V = xi / 2 - u / (s_new + p.beta)
        e = p.kappa / (s_new + p.beta)
        lo = np.zeros(N + 1); di = np.ones(N + 1); up = np.zeros(N + 1); rhs = np.zeros(N + 1)
        Vp = np.maximum(V[ii], 0.0)                      # forward difference if V >= 0
        Vm = np.minimum(V[ii], 0.0)                      # backward difference if V < 0
        lo[ii] = -dsig * (e[ii] / d**2 - Vm / d)
        up[ii] = -dsig * (e[ii] / d**2 + Vp / d)
        di[ii] = 1 + dsig * (2 * e[ii] / d**2 + Vp / d - Vm / d)
        rhs[ii] = T[ii]
        rhs[0], rhs[-1] = p.T_inj, 0.0
        T_new = tridiag(lo, di, up, rhs)

        change = max(np.max(np.abs(s_new - s)), np.max(np.abs(T_new - T)),
                     np.max(np.abs(c_new - c)))
        s, c, T = s_new, c_new, T_new
        if verbose and it % 2000 == 0:
            print(f"  it {it:6d}  change {change:.3e}")
        if change < tol * dsig and it > 10:
            break

    res = residual(s, c, T, xi, d, p)
    return xi, s, c, T, it, res


def residual(s, c, T, xi, d, p):
    """Max-norm of the right-hand sides of the explicit scheme (D1)-(D3)."""
    K, G = fluxes(s, c, T, d, p)
    u = velocity(G)
    ii = slice(1, len(s) - 1)
    rs = (G[1:] - G[:-1]) / d + xi[ii] / 2 * (s[2:] - s[1:-1]) / d
    V = xi / 2 - u / (s + p.beta)
    Tf = (T[2:] - T[1:-1]) / d
    Tb = (T[1:-1] - T[:-2]) / d
    rT = p.kappa / (s[ii] + p.beta) * (T[2:] - 2 * T[1:-1] + T[:-2]) / d**2 \
        + V[ii] * np.where(V[ii] >= 0, Tf, Tb)
    return max(np.max(np.abs(rs)), np.max(np.abs(rT)))


# --------------------------------------------------------------------------
# front positions
# --------------------------------------------------------------------------
def crossing(xi, y, level):
    """First xi where y (decreasing) crosses `level`, linear interpolation."""
    k = np.where((y[:-1] >= level) & (y[1:] < level))[0][0]
    return xi[k] + (y[k] - level) / (y[k] - y[k + 1]) * (xi[k + 1] - xi[k])


def fronts(xi, s, c, T, p):
    k = np.where(c > 0.5 * p.c_inj)[0][-1]              # last node with c = c^0
    xi_c = 0.5 * (xi[k] + xi[k + 1])                    # step located mid-cell
    xi_s = crossing(xi, s, 0.5 * (p.s_inj + p.s_init))
    xi_T = crossing(xi, T, 0.5 * p.T_inj)
    return xi_c, xi_s, xi_T


# --------------------------------------------------------------------------
def grid_study(p, Ns=(30, 60, 120, 240, 480, 960, 1920), save_prefix=None):
    rows = []
    for N in Ns:
        xi, s, c, T, it, res = solve(N, p)
        xc, xs, xT = fronts(xi, s, c, T, p)
        rows.append((N, p.xi_max / N, xc, xs, xT, it, res))
        if save_prefix is not None:
            np.savetxt(f"{save_prefix}_N{N}.dat", np.column_stack([xi, s, c, T]),
                       header="xi  s  c/c0  T/T0", fmt="%.8e")
    ext = [2 * rows[-1][k] - rows[-2][k] for k in (2, 3, 4)]   # Richardson, first order
    print(f"lambda = {p.lam} W/(m C):  beta = {p.beta:.3f},  kappa/alpha^2 = {p.kappa:.3f}")
    print(f"{'N':>5} {'dxi':>9} {'xi_c':>8} {'err%':>6} {'xi_s':>8} {'err%':>6} "
          f"{'xi_T':>8} {'err%':>6} {'iters':>7} {'resid':>9}")
    for N, d, xc, xs, xT, it, res in rows:
        e = [abs(v - r) / r * 100 for v, r in zip((xc, xs, xT), ext)]
        print(f"{N:5d} {d:9.5f} {xc:8.4f} {e[0]:6.1f} {xs:8.4f} {e[1]:6.1f} "
              f"{xT:8.4f} {e[2]:6.1f} {it:7d} {res:9.1e}")
    print(f"{'extrapolated':>15} {ext[0]:8.4f} {'':6} {ext[1]:8.4f} {'':6} {ext[2]:8.4f}")
    # observed orders from successive differences
    for k, name in ((3, 'xi_s'), (4, 'xi_T')):
        v = [r[k] for r in rows]
        orders = [np.log2(abs(v[i] - v[i - 1]) / abs(v[i + 1] - v[i]))
                  for i in range(1, len(v) - 1)]
        print(f"observed orders {name}: " + ", ".join(f"{o:.2f}" for o in orders))
    return rows, ext


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=36.0, help="thermal conductivity, W/(m C)")
    ap.add_argument("--save", type=str, default=None, help="prefix for profile files")
    ap.add_argument("--Nmax", type=int, default=1920)
    a = ap.parse_args()
    Ns = [n for n in (30, 60, 120, 240, 480, 960, 1920) if n <= a.Nmax]
    grid_study(Params(lam=a.lam), Ns, a.save)
