# Weekend vibe-coding project on differentiable reactive flow

A Julia research codebase for a 2D axisymmetric packed-bed flow/reactor solver, including:

- Incompressible-style verification workflows
- Low-Mach variable-density (compressible) workflows
- Reactive transport and heat transfer in porous media
- CHEMKIN-driven methanation example
- AD/sensitivity scaffolding

## Status

**This project is not ready yet.**

It is under active research/development. Expect breaking changes, evolving numerics, and incomplete docs.

## Repository Layout

- `test/test_main.jl`: Verification-style runs for the base packed-bed solver path.
- `test/test_main_compressible.jl`: Verification runs for the variable-density compressible path.
- `test/test_main_methanation.jl`: End-to-end methanation example with CHEMKIN kinetics.
- `src/`: Core numerics and model modules.
- `data/methanation.inp`: CHEMKIN mechanism input used by the methanation case.
- `test/`: Test entrypoint and focused test files.
- `Project.toml`, `Manifest.toml`: Julia environment definitions.

## Numerics

The core PDE system is discretized on a 2D axisymmetric finite-volume mesh (`r-z`) with state layout:

- `[u_r | u_z | p | T_f | T_s | Y_1..Y_Ns]`

Key modeling and discretization choices:

- Pressure-velocity coupling uses artificial compressibility pseudo-time marching.
- Base solver enforces steady `div(eps*rho*u)=0` with constant `rho` from `FluidProps`.
- Compressible solver is low-Mach, isobaric EOS: `rho = p0*Mbar/(R*T_f)`, and enforces steady `div(rho*u)=0` in the mass-flux sense.
- Momentum uses Darcy-Brinkman-Forchheimer drag with effective viscosity closures.
- Scalar transport (temperature/species) uses upwind advection and central diffusion in axisymmetric FVM form.
- Inlet Dirichlet boundaries are handled with separate advection and diffusion ghost values for scalars.
- Uniform-field-preserving correction is applied in advection terms (`div(u*q) - q*div(u)` style) to avoid spurious source terms in under-converged pseudo-transients.

Numerics at a glance (equations):

| Block | Governing form (continuous) | Discrete treatment in this code |
|---|---|---|
| Continuity / pressure coupling | `dp/dt = -beta^2 div(eps rho u)` (base) or `dp/dt = -beta^2 eps div(rho u)` (compressible) | Axisymmetric FVM divergence with face fluxes; pseudo-time marched to steady residual |
| r-momentum | `(rho/eps) du_r/dt = -dp/dr + mu_eff(grad^2(u_r) - u_r/r^2) - D u_r` | Central differences for pressure/viscous terms, porous drag closure, axisymmetric geometric factors |
| z-momentum | `(rho/eps) du_z/dt = -dp/dz + mu_eff grad^2(u_z) - D u_z - rho g` | Same as r-momentum plus gravity source in z |
| Fluid energy (`T_f`) | `eps rho c_p dT_f/dt = -rho c_p (u*grad(T_f)) + div(lambda_eff grad(T_f)) + h_fs a_v (T_s-T_f)` | Upwind advection + central diffusion; inlet advection/diffusion ghosts split; interphase heat-transfer source |
| Solid energy (`T_s`) | `(1-eps)rho_s c_ps dT_s/dt = div(lambda_s,eff grad(T_s)) - h_fs a_v (T_s-T_f) + phi_cat sum(-dH)r` | Central diffusion with reaction heat source and interphase coupling |
| Species (`Y_k`) | `eps rho dY_k/dt = -rho (u*grad(Y_k)) + div(rho D_k,eff grad(Y_k)) + phi_cat M_k sum(nu_k) r` | Upwind advection + central diffusion + reaction source per species |
| EOS (compressible path) | `rho = p0 Mbar/(R T_f)`, `Mbar^{-1}=sum(Y_k/M_k)` | Cellwise density update from `T_f` and composition; face density by interpolation |

The scalar advection operators use a uniform-field-preserving correction (`div(F q) - q div(F)`) so a spatially uniform scalar is not spuriously created/destroyed in under-converged pseudo-transients.

Time integration / steady strategy:

- Default pseudo-transient integrator: `Tsit5` (explicit RK45-style adaptive method).
- Rationale in this repo: avoids dense Jacobian construction cost from implicit methods on larger grids.
- Optional Newton-Krylov steady solve is available via `NonlinearSolve` + GMRES (`solve_steady_nk`) for implicit root-finding of steady residuals.
- A two-phase operator-split path is provided (`solve_steady_two_phase`):
	- Phase 1: equilibrate momentum/pressure with high artificial sound speed (`beta_ac`).
	- Phase 2: freeze momentum/pressure and advance species/energy.

Practical tuning knobs used in examples/tests:

- `beta_ac` controls pseudo-acoustic stiffness and CFL balance.
- `t_end` (or phase durations) is chosen from dominant physical relaxation scales (viscous, drag, residence/reaction).
- PASS/FAIL thresholds in verification scripts are empirical and still being refined.

## Quick Start

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite:

```bash
julia --project=. test/runtests.jl
```

Run the main verification/example entrypoints:

```bash
julia --project=. main.jl
julia --project=. main_compressible.jl
julia --project=. main_methanation.jl
```

## Notes

- This code is intended for experimentation and verification, not production deployment.
- Numerical stability, closure choices, and runtime behavior depend strongly on setup.
- Output and PASS/FAIL thresholds are part of ongoing validation work.

## Symbol Legend

| Symbol | Meaning | Dimension / SI unit |
|---|---|---|
| `r, z` | Radial and axial coordinates (axisymmetric domain) | length `[m]` |
| `u_r, u_z` | Radial and axial superficial velocity components | velocity `[m/s]` |
| `p` | Dynamic (artificial-compressibility) pressure variable | pressure `[Pa]` |
| `beta_ac`, `beta` | Artificial compressibility wave-speed parameter | velocity `[m/s]` |
| `eps`, `eps` | Bed porosity | dimensionless `[-]` |
| `rho`, `rho` | Fluid density | mass density `[kg/m^3]` |
| `mu`, `mu` | Dynamic viscosity | `[Pa*s]` |
| `mu_eff`, `mu_eff` | Effective Brinkman viscosity | `[Pa*s]` |
| `D` | Darcy-Forchheimer drag coefficient | momentum sink factor `[kg/(m^3*s)]` |
| `g` | Gravitational acceleration (axial source in momentum) | `[m/s^2]` |
| `T_f`, `T_s` | Fluid and solid temperatures | temperature `[K]` |
| `c_p`, `c_pf`, `c_ps` | Heat capacity (fluid/solid as indicated by subscript) | specific heat `[J/(kg*K)]` |
| `lambda`, `lambda` | Thermal conductivity | `[W/(m*K)]` |
| `lambda_eff`, `lambda_eff` | Effective thermal conductivity | `[W/(m*K)]` |
| `h_fs` | Fluid-solid interphase heat-transfer coefficient | `[W/(m^2*K)]` |
| `a_v` | Interfacial area density (surface area per bulk volume) | `[1/m]` |
| `phi_cat`, `phi_cat` | Catalytically active volume fraction / masking factor | dimensionless `[-]` |
| `Y_k` | Mass fraction of species `k` | dimensionless `[-]` |
| `D_k` | Species diffusivity for species `k` | `[m^2/s]` |
| `M_k` | Molar mass of species `k` | `[kg/mol]` |
| `nu_k`, `nu_k` | Stoichiometric coefficient of species `k` | dimensionless `[-]` |
| `r_j` | Rate of reaction `j` | molar source `[mol/(m^3*s)]` |
| `DeltaH`, `DeltaH` | Reaction enthalpy change | molar enthalpy `[J/mol]` |
| `p0` | Thermodynamic reference pressure (compressible path) | pressure `[Pa]` |
| `Mbar`, `Mbar` | Mean molar mass of the gas mixture | `[kg/mol]` |
| `R` | Universal gas constant | `[J/(mol*K)]` |
