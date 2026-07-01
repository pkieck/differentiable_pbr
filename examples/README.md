# Examples

Runnable example cases for the 2D axisymmetric packed-bed solver. Each script
sets up a physical scenario (mirroring one of the verification tests), solves to
steady state, and writes a `figures/*.png` panel of **(r,z) heatmaps for every
state field** — `u_r`, `u_z`, `p`, `T_f`, `T_s`, all species `Y_k`, and (for the
compressible cases) the density `ρ`.

Run any of them from the project root:

```bash
julia --project examples/example_reactor.jl
julia --project examples/example_thermal_expansion.jl
julia --project examples/example_methanation.jl
julia --project examples/example_uneven_catalyst.jl
julia --project examples/example_wall_heatloss.jl
```

| Script | Physics | Solver | Mirrors |
|--------|---------|--------|---------|
| `example_reactor.jl` | Exothermic A→B in a liquid-filled packed bed (two-T heat-up + conversion) | incompressible two-phase split | Test 5b (`main.jl`) |
| `example_thermal_expansion.jl` | Exothermic A→B heats a gas → it expands and accelerates (ρ↓, u↑) | low-Mach variable-density | Test C2 (`main_compressible.jl`) |
| `example_methanation.jl` | Catalytic methanation (CO/CO₂ + H₂ → CH₄ + H₂O) over Ni, CHEMKIN kinetics | low-Mach variable-density, 6 species | `main_methanation.jl` |
| `example_uneven_catalyst.jl` | Same A→B reactor with a **graded catalyst field** φ_cat(r,z) (inert entry, core-weighted) — the reaction follows the active zone | incompressible two-phase split | — |
| `example_wall_heatloss.jl` | Same A→B reactor with **convective wall heat loss** α(T−T_amb), T_amb = 293.5 K — hot core, cooled near-wall layer | incompressible two-phase split | — |

### Surrogate

[`surrogate/`](surrogate/) trains a **differentiable-solver-backed neural
surrogate** of the wall-heatloss reactor that takes the **inlet temperature** and
**inlet velocity profile** `(T_in, U, m)` and predicts all six steady fields. It
uses a ViT-patch-embed / DiT / Perceiver-IO architecture (Lux.jl) and a **Sobolev**
loss that matches the solver's exact input-sensitivities (`∂fields/∂θ` from
ForwardDiff) for gradient-accurate, inverse-design-ready predictions. See
[`surrogate/README.md`](surrogate/README.md).

### Notes on the two non-uniform cases

- **Uneven catalyst** needs no solver changes: `φ_cat` already multiplies the
  reaction heat and species sources in `src/rhs.jl`, so it's set purely by
  building a `PorousMedium` with a spatially-varying `phi_cat` field. The φ_cat
  field is drawn as an extra panel.
- **Wall heat loss** adds a Robin wall condition the validated solver doesn't
  have (its wall is adiabatic/Neumann). It does so *without touching* `src/`: a
  thin `rhs_wallloss!` wrapper calls the original `rhs!` and adds the
  α(T−T_amb) sink to the fluid/solid energy rows of the outer (i = nr) cells,
  converting the wall-face flux to a volumetric sink exactly as the interphase
  term is handled in `src/rhs.jl`.

## Plotting helper

`plotting.jl` provides `plot_state(u, s, g; species_names, title, density)`,
which returns a `Plots` figure with one heatmap panel per state field, and
`save_fig(fig, name)` to write it under `examples/figures/`. Reuse it to plot the
state from any case of your own — the layout adapts to the number of species.

Figures are written to `examples/figures/` (git-ignored).
