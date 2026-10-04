# Self-similar computations for non-isothermal capillary imbibition with adsorption

Code and data accompanying the article

> A. A. Mussina, *Non-Equilibrium Two-Phase Filtration with Adsorption and Heat Transfer
> in Porous Media: Well-Posedness, Stefan-Type Limit and Self-Similar Computations*,
> submitted to Transport in Porous Media.

The code implements the upwind finite-difference scheme of Section 4 / Appendix D.1 of the
article for the self-similar problem (variable ξ = x / √(α²t)) of non-isothermal capillary
imbibition with an adsorbing additive (equilibrium, linear isotherm). It reproduces
Table 3 (grid convergence), Table 4 (sensitivity to the thermal conductivity λ), Fig. 2
(profiles) and Fig. 3 (errors of the front positions).

## Requirements

Python 3 with NumPy and SciPy (`pip install -r requirements.txt`).
Figures: a LaTeX distribution with PGFPlots/TikZ.

## Contents

| Path | Description |
|---|---|
| `code/selfsimilar_imbibition.py` | solver and grid study (Table 3) |
| `code/sensitivity_lambda.py` | front positions for λ = 2.5, 5, 10, 20, 36 W/(m °C) (Table 4) |
| `code/make_fig3_data.py` | errors of the front positions for Fig. 3 |
| `code/out/` | computed results (see below) |
| `figures/` | LaTeX/PGFPlots sources of Figs. 1–3 |

## Reproducing the results

Run all commands from the `code/` directory.

```bash
python selfsimilar_imbibition.py --save out/lam36 > out/grid_lam36.txt
python selfsimilar_imbibition.py --lam 2.5 --save out/lam2.5 > out/grid_lam2.5.txt
python sensitivity_lambda.py
python make_fig3_data.py
```

1. Grid study with the parameters of Table 2 (λ = 36 W/(m °C)), grids N = 30 … 1920:
   front positions ξ_c (mid-cell position of the concentration step), ξ_s
   (s = (s⁰ + s₀)/2), ξ_T (T = T⁰/2), their errors relative to the Richardson-extrapolated
   values and the observed orders (Table 3). Every grid is iterated until the residual of
   the stationary discrete equations is below 10⁻¹⁰. The finest grids take a few minutes.
2. The same grid study for λ = 2.5 W/(m °C).
3. Sensitivity to λ (Table 4): writes `out/sensitivity_lambda.txt` and the profiles
   `out/lam*_N1920.dat`.
4. Errors for Fig. 3: writes `out/fig3_errors.dat` (requires the profiles of step 1).

Use `--Nmax 480` for a quick test.

## Output files (`code/out/`)

* `lam<λ>_N<N>.dat` – profiles: columns ξ, s, c/c⁰, T/T⁰
* `grid_lam36.txt`, `grid_lam2.5.txt` – grid studies
* `sensitivity_lambda.txt` – front positions for different λ (N = 1920 and extrapolated)
* `fig3_errors.dat` – relative errors of the front positions (λ = 36 W/(m °C))
* `profiles_lambda_comparison.png` – quick-look plot of the profiles for λ = 36 and 2.5

## Figures

Compile in the `figures/` directory with `pdflatex Fig2_profiles.tex` etc.;
Figs. 2 and 3 read their data from `../code/out/`. Fig. 1 is a schematic drawn with TikZ.

## License

MIT, see `LICENSE`.
