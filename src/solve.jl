"""
Solver layer.

Design rationale for solver choice
────────────────────────────────────
The packed-bed equations are an ODE system (via artificial compressibility).
The implicit Rosenbrock solvers (Rodas5P) build a dense O(N²) Jacobian via
ForwardDiff on every step — this is O(N³) per time step and unacceptable for
grids larger than ~10×10.

Instead we use **Tsit5** (explicit, 4th/5th order Runge-Kutta):
  • No Jacobian required — each step is O(N)
  • Adaptive step control; handles moderate stiffness well
  • Works when β_ac ≈ U_flow (acoustic CFL ≈ convective CFL)

Setting β_ac ≈ max(U_flow) balances the acoustic and physical time scales so
the explicit solver can take time steps ≈ dr / U without stability issues.

For production inverse design (large N), the plan §4 path is:
  • Newton-Krylov steady-state solve (no pseudo-time at all)
  • IFT adjoint: one GMRES solve for the gradient regardless of N
  • Sparse Jacobian via SparseConnectivityTracer + SparseDiffTools
This is tracked in ad.jl and is the next implementation milestone.
"""

include("rhs.jl")

using DifferentialEquations: ODEProblem, solve
using DifferentialEquations: Tsit5, Rodas5P
import NonlinearSolve as NLS
import LinearSolve as LS

"""Integrate over [0, t_end]. Default: Tsit5 (explicit, fast for moderate stiffness).
Switch to Rodas5P for small grids or very stiff systems.
"""
function solve_transient(u0::AbstractVector, params::ReactorParams;
                         tspan=(0.0, 1e3), solver=Tsit5(), kwargs...)
    prob = ODEProblem(rhs!, u0, tspan, params)
    solve(prob, solver; abstol=1e-6, reltol=1e-5,
          save_everystep=false, kwargs...)
end

"""
    solve_steady(u0, params; t_end, solver, kwargs...)

Pseudo-transient solve to steady state using Tsit5.

Choose `t_end` >= the longest physical relaxation time in the problem:
  - Open-tube viscous: `t_end ~ 5 rho R^2 / mu`
  - Packed bed drag: `t_end ~ 5 rho / (eps A mu)` where `A` is the Ergun A coefficient
  - Species reaction: `t_end ~ 5 L / U_in`

Set `params.beta_ac ~ U_mean` to balance acoustic and convective CFL numbers,
so the explicit solver can take steps on the order of `dr / U_mean`.

Returns the solution; `.u[end]` is the final steady-state vector.
"""
function solve_steady(u0::AbstractVector, params::ReactorParams;
                      t_end=1e3, solver=Tsit5(), kwargs...)
    prob = ODEProblem(rhs!, u0, (0.0, Float64(t_end)), params)
    solve(prob, solver; abstol=1e-6, reltol=1e-5,
          save_everystep=false, kwargs...)
end

"""
    apply_inlet_bc!(u, s, uz_in, ur_in, Tf_in, Ts_in, Ys_in)

Stamp Dirichlet inlet values into the j=1 cells of the state vector.
The ghost-cell BCs in rhs! use params.bcs for the boundary values,
so these two must be consistent.
"""
function apply_inlet_bc!(u::AbstractVector, s::StateLayout,
                         uz_in::Real, ur_in::Real=0.0,
                         Tf_in::Real=300.0, Ts_in::Real=300.0,
                         Ys_in::AbstractVector=Float64[])
    uz_f = field_mat(u, s, F_UZ)
    ur_f = field_mat(u, s, F_UR)
    Tf   = field_mat(u, s, F_TF)
    Ts   = field_mat(u, s, F_TS)
    for i in 1:s.nr
        uz_f[i,1] = uz_in
        ur_f[i,1] = ur_in
        Tf[i,1]   = Tf_in
        Ts[i,1]   = Ts_in
    end
    for k in 1:min(s.Ns, length(Ys_in))
        Y = field_mat(u, s, 5+k)
        for i in 1:s.nr
            Y[i,1] = Ys_in[k]
        end
    end
    u
end

# 3-argument residual wrapper for NonlinearProblem / IFT adjoint in ad.jl
_residual!(du, u, p) = rhs!(du, u, p, zero(eltype(u)))

"""
    solve_steady_nk(u0, params; tol, maxiters, verbose)

Newton-Krylov steady-state solve using NonlinearSolve.jl.

Solves F(u; params) = 0 where F = rhs! at t=0 (steady residual).
Uses Newton-Raphson with GMRES inner iterations so no dense Jacobian is
built — cost per Newton step scales as O(N) (matrix-free JVPs via ForwardDiff).

Prefer this over the pseudo-transient `solve_steady` when:
  • You need the IFT adjoint (implicit-function-theorem gradient), since the
    IFT requires a converged Newton root, not just a well-relaxed pseudo-time.
  • The grid is too large for the pseudo-transient to converge in reasonable time.

Tip: warm-start with `u0 = solve_steady(...).u[end]` from a pseudo-transient run
on a coarse grid to get the NK solver into the basin of convergence quickly.
"""
function solve_steady_nk(u0::AbstractVector, params::ReactorParams;
                          tol=1e-8, maxiters=200, verbose=false)
    prob = NLS.NonlinearProblem(_residual!, copy(u0), params)
    NLS.solve(prob,
              NLS.NewtonRaphson(linsolve=NLS.KrylovJL_GMRES());
              abstol=tol, reltol=tol, maxiters=maxiters, verbose=verbose)
end

"""
    solve_steady_two_phase(u0, params; β_flow, t_flow, t_species)

Operator-split steady-state solve (plan §2, "operator-split chemistry"):

  Phase 1 — Equilibrate flow (short pseudo-time, high β_ac):
    Uses `β_flow` (default 100 m/s) so acoustic waves traverse the domain in
    L/β_flow ≈ 0.01 s, much faster than drag relaxation.  Tsit5 is stable.

  Phase 2 — Transport species + energy with frozen velocity:
    Velocity/pressure derivatives are zeroed after each RHS call.
    Without momentum coupling, the only stiff mode is species advection
    (λ ~ U/dz + k_rxn), giving large Tsit5 steps (Δt ~ dz/U ~ seconds).

This mirrors the plan's recommendation to operator-split the stiff reaction
step from the momentum-pressure step, giving O(N) scaling per phase.

Returns the final state vector `u*`.
"""
function solve_steady_two_phase(u0::AbstractVector, params::ReactorParams;
                                β_flow=100.0, t_flow=2.0, t_species=600.0)
    g    = params.grid
    Ns   = length(params.D_species)
    s    = StateLayout(g.nr, g.nz, Ns)
    T    = eltype(u0)

    # Phase 1: equilibrate momentum + pressure with high β_ac
    p_flow = ReactorParams(g, params.porous, params.fluid, params.solid,
                           params.reactions, T(β_flow), params.gravity,
                           params.D_species, params.bcs)
    sol1 = solve_steady(u0, p_flow; t_end=t_flow)
    u1   = sol1.u[end]

    # Phase 2: species + energy transport with frozen velocity/pressure
    function frozen_rhs!(du, u, p, t)
        rhs!(du, u, p, t)
        # Freeze momentum and pressure rows
        field_mat(du, s, F_UR) .= zero(T)
        field_mat(du, s, F_UZ) .= zero(T)
        field_mat(du, s, F_P)  .= zero(T)
    end
    prob2 = ODEProblem(frozen_rhs!, u1, (0.0, Float64(t_species)), params)
    sol2  = solve(prob2, Tsit5(); abstol=1e-6, reltol=1e-5, save_everystep=false)
    sol2.u[end]
end
