# Wall-heatloss reactor surrogate

A neural surrogate that maps the **inlet temperature** and **inlet velocity
profile** of the non-adiabatic packed-bed reactor (the `example_wall_heatloss.jl`
physics) to its full steady (r,z) field solution — built to exploit the solver's
**differentiability** for data-efficient, gradient-accurate training.

```
  θ = [ T_in , U , m ]   ──►   surrogate   ──►   6 fields on the (nr×nz) grid
       │     │   │                                u_r, u_z, T_f, T_s, Y_A, Y_B
       │     │   └ inlet velocity-profile shape exponent
       │     └ area-averaged inlet velocity
       └ inlet temperature
```

The inlet velocity is the one-parameter radial family
`uz_in(r) = U·(m+2)/m·(1−(r/R)^m)` (area-averaged to `U` for any `m`: `m=2` is
parabolic/Poiseuille, large `m` → plug, `m<2` → centre-peaked). This required a
small, backward-compatible solver change: `BoundaryConditions.uz_in` may now be a
length-`nr` radial profile (see `bc_at` in `src/params.jl`); the scalar path is
bit-for-bit unchanged (all 118 solver tests still pass, C0 reduction 4.8e-16).

## Pipeline

| File | Role |
|------|------|
| `wallloss_problem.jl` | Parametric, **differentiable** physics: `run_case(θ) → (nr,nz,6)`. Wraps the validated two-phase wall-loss solver; a `Dual` θ propagates ForwardDiff partials through the whole pseudo-transient solve. |
| `gen_data.jl` | LHS-samples the `(T_in,U,m)` box; for each point stores the fields **and** `∂fields/∂θ` (the Sobolev targets) via one `ForwardDiff.jacobian!` pass. |
| `model.jl` | The surrogate (Lux.jl): coordinate **ViT patch-embedding** → **DiT** (adaLN-Zero) processor blocks → **Perceiver-IO** cross-attention decoder → 6 fields. |
| `train.jl` | **Sobolev** training: `MSE(field) + λ·MSE(∂field/∂input)`, both normalised. |
| `validate.jl` | Test-set accuracy, **surrogate-vs-solver gradient check**, one inverse-design round, and a true-vs-predicted figure. |

```bash
julia --project -t auto examples/surrogate/gen_data.jl 256 64   # ~50 min (ForwardDiff-heavy)
julia --project           examples/surrogate/train.jl 250       # ~70 min on CPU
julia --project           examples/surrogate/validate.jl
```

Artifacts land in `examples/surrogate/data/` (`wallloss_dataset.jls`,
`surrogate.jld2`) and `examples/figures/surrogate_compare.png` (all git-ignored).

## Leveraging differentiability

This is the point of building the surrogate on a differentiable solver:

1. **Sobolev / gradient-matched training.** Each label ships with the exact
   `∂(fields)/∂θ` from the solver (3 cheap forward-mode derivatives). Training the
   surrogate to match these — not just the field values — sharply improves data
   efficiency and, critically, makes the surrogate's *own* gradient accurate. The
   surrogate's `∂ŷ/∂θ` is taken by a central difference in normalised input space
   (`2·n_in` extra forward passes), so the loss differentiates with plain
   first-order Zygote — no nested AD.

2. **Gradient verification.** `validate.jl` compares `∂(outlet conversion)/∂θ`
   from the surrogate (AD through the net) against the solver (ForwardDiff through
   `run_case`) — the cosine-similarity is the headline metric for inverse-design
   readiness.

3. **Inverse design.** The accurate surrogate gradient is used to solve a small
   `argmin_θ (X(θ) − X_target)²`, and the chosen θ is verified against the full
   solver.

A future extension is a **PDE-residual** regularizer (feed `model(θ)` back through
the differentiable `rhs!` and penalize `‖rhs!(ŷ)‖²`) for physics-consistent
extrapolation; the code is structured to add it as an extra loss term.

## Architecture (model.jl)

- **Encoder.** Build a `(nr,nz,Cin)` image = per-cell coordinates `[r,z]` plus the
  conditioning θ broadcast over the grid; a strided `Conv(patch)` patch-embeds it
  into a grid of tokens (+ learned positional embedding). Conditioning enters both
  here and via adaLN (standard for conditional ViT/DiT).
- **Conditioning.** `c = MLP(θ)` drives **adaLN-Zero** modulation (shift/scale/gate
  from `c`, zero-initialised so each block starts at identity).
- **Processor.** `n_proc` **DiT** blocks: self-attention + MLP, each adaLN-modulated.
- **Decoder.** `n_dec` **Perceiver-IO** blocks: one query token per output cell
  (learned positional embedding + conditioning) cross-attends the processor latent,
  adaLN-modulated; an adaLN head maps to the 6 channels.

All tensors are feature-first, batch-last; attention uses `NNlib.dot_product_attention`.
