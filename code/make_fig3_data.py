"""
Data file for Fig. 3: relative errors of the front positions versus the grid step.

Reads the profiles of the grid study (lambda = 36 W/(m C)) written by
    python selfsimilar_imbibition.py --save out/lam36
determines the front positions, extrapolates them as in Table 3 and writes
out/fig3_errors.dat.

    python make_fig3_data.py
"""
import numpy as np
from selfsimilar_imbibition import Params, fronts

NS = (30, 60, 120, 240, 480, 960, 1920)

p = Params(lam=36.0)
rows = []
for N in NS:
    xi, s, c, T = np.loadtxt(f"out/lam36_N{N}.dat", unpack=True)
    rows.append((N, p.xi_max / N, *fronts(xi, s, c, T, p)))
ext = np.array([2 * rows[-1][k] - rows[-2][k] for k in (2, 3, 4)])   # Richardson, first order

with open("out/fig3_errors.dat", "w") as fh:
    fh.write("N h err_c err_s err_T\n")
    for N, h, *x in rows:
        e = np.abs(np.array(x) - ext) / ext
        fh.write(f"{N} {h:.6f} {e[0]:.6e} {e[1]:.6e} {e[2]:.6e}\n")
