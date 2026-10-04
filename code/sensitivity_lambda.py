"""
Sensitivity of the front positions to the thermal conductivity lambda.

For each lambda the problem is solved on the grids N = 960 and N = 1920 and the
front positions are Richardson-extrapolated assuming first-order convergence
(as in Table 3).  Profiles on the finest grid are written to out/.

    python sensitivity_lambda.py
"""
import numpy as np
from selfsimilar_imbibition import Params, solve, fronts

LAMBDAS = (2.5, 5.0, 10.0, 20.0, 36.0)

rows = []
for lam in LAMBDAS:
    p = Params(lam=lam)
    res = []
    for N in (960, 1920):
        xi, s, c, T, it, r = solve(N, p)
        res.append(fronts(xi, s, c, T, p))
    np.savetxt(f"out/lam{lam:g}_N1920.dat", np.column_stack([xi, s, c, T]),
               header="xi  s  c/c0  T/T0", fmt="%.8e")
    f960, f1920 = np.array(res[0]), np.array(res[1])
    ext = 2 * f1920 - f960
    rows.append((lam, p.kappa, *f1920, *ext))
    print(f"lambda={lam:5.1f}  kappa/alpha2={p.kappa:7.3f}  "
          f"N=1920: xi_c={f1920[0]:.4f} xi_s={f1920[1]:.4f} xi_T={f1920[2]:.4f}   "
          f"extrap: xi_c={ext[0]:.4f} xi_s={ext[1]:.4f} xi_T={ext[2]:.4f}", flush=True)

np.savetxt("out/sensitivity_lambda.txt", np.array(rows),
           header="lambda kappa/alpha2 xi_c xi_s xi_T (N=1920)  xi_c xi_s xi_T (extrapolated)",
           fmt="%.5f")
