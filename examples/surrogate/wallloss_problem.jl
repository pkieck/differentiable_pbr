"""
wallloss_problem.jl — Parametric, differentiable wall-heatloss reactor problem.

This is the *data-generating physics* for the surrogate.  It wraps the validated
incompressible two-phase solver (via the wall-heatloss Robin wrapper from
`example_wall_heatloss.jl`) into a single function

    run_case(θ) -> (nr, nz, 6) array of the steady fields
                   [u_r, u_z, T_f, T_s, Y_A, Y_B]

parametrised by the 3-vector

    θ = [T_in, U, m]
        T_in : inlet temperature                [K]
        U    : area-averaged inlet axial velocity [m/s]
        m    : inlet velocity-profile shape exponent (dimensionless)

The inlet axial velocity is a one-parameter radial family

    uz_in(r) = U · (m+2)/m · (1 − (r/R)^m)

which is *area-averaged to exactly U* for any m (axisymmetric mean of
1−(r/R)^m over the disk is m/(m+2)).  m=2 is the parabolic (Poiseuille) profile
with centreline 2U; large m → flat plug ≈ U; m<2 → centre-peaked beyond
parabolic.  This gives the surrogate a continuous "magnitude + shape" knob.

`run_case` is written so it differentiates cleanly through ForwardDiff: pass a
`Dual`-typed θ and the whole pseudo-transient solve carries the partials, which
is how we get the Sobolev (input-sensitivity) targets for training.
"""

include("../../src/ad.jl")        # → solve.jl → rhs!  (StateLayout, ReactorParams, solver)

using DifferentialEquations: ODEProblem, solve, Tsit5
import ForwardDiff

# Primal (derivative-free) value of a possibly-Dual number — used for the
# pseudo-time integration horizons, which must be plain Float64 (the steady
# state is independent of how long we integrate, so they carry no partials).
primal_value(x::Real) = Float64(x)
primal_value(x::ForwardDiff.Dual) = primal_value(ForwardDiff.value(x))

# ── Fixed problem geometry / properties (everything except θ) ────────────────
Base.@kwdef struct WallLossSetup
    nr::Int     = 8
    nz::Int     = 40
    R::Float64  = 0.05
    L::Float64  = 1.0
    eps_v::Float64 = 0.4
    dp_v::Float64  = 3e-3
    μ::Float64  = 1e-3
    ρ::Float64  = 1000.0
    cpf::Float64 = 1000.0
    λf::Float64  = 0.03
    M_A::Float64 = 0.028
    M_B::Float64 = 0.028
    dH::Float64  = -5000.0
    Da::Float64  = 0.1          # Damköhler at the reference velocity U_ref and T_ref
    U_ref::Float64 = 0.01       # reference velocity that sets A_pre (= Da·U_ref/L)
    E_a::Float64  = 33256.0     # activation energy [J/mol] (Arrhenius γ=E_a/(R·T_ref)≈10)
    T_ref::Float64 = 400.0      # reference temperature at which Da is defined [K]
    α_wall::Float64 = 20.0      # wall heat-transfer coeff [W/(m²·K)]
    T_amb::Float64  = 293.5     # ambient (wall sink) temperature [K]
    β_flow::Float64 = 100.0     # phase-1 artificial sound speed
end

const NFIELDS_OUT = 6           # u_r, u_z, T_f, T_s, Y_A, Y_B  (pressure dropped)

"Radial inlet velocity profile, area-averaged to U.  Returns a length-nr vector."
function inlet_profile(U, m, g::Grid2D)
    R = g.R
    return [U * (m + 2) / m * (1 - (g.r[i] / R)^m) for i in 1:g.nr]
end

"""
    run_case(θ, setup=WallLossSetup()) -> Array{T,3}  (nr, nz, 6)

Solve the steady non-adiabatic reactor for θ = [T_in, U, m] and return the six
physical fields stacked on the last axis.  `T = eltype(θ)`, so a Dual θ yields a
Dual field array (ForwardDiff sensitivities).
"""
function run_case(θ::AbstractVector, setup::WallLossSetup=WallLossSetup())
    T_in, U, m = θ[1], θ[2], θ[3]
    Tq = promote_type(eltype(θ), Float64)

    g     = Grid2D(setup.nr, setup.nz, setup.R, setup.L)
    pm    = uniform_bed(setup.nr, setup.nz, setup.eps_v, setup.dp_v, 1.0)
    fluid = FluidProps(setup.ρ, setup.μ, setup.cpf, setup.λf)
    solid = SolidProps(2000.0, 800.0, 1.0)
    Ns    = 2
    # Rate constant Da·U_ref/L is the *effective* rate at T_ref; fold the
    # Arrhenius factor back out so A_pre·exp(−E_a/(R·T_ref)) hits that target.
    R_gas = 8.314462618
    A_pre = (setup.Da * setup.U_ref / setup.L) * exp(setup.E_a / (R_gas * setup.T_ref))
    rxn   = Reaction(A_pre, setup.E_a, (-1.0, 1.0), setup.dH,
                     (setup.M_A, setup.M_B), (true, false))

    uz_in = inlet_profile(U, m, g)            # length-nr Dual/Float vector
    params = ReactorParams(g, pm, fluid, solid, (rxn,), setup.β_flow, 0.0,
                           fill(1e-5, Ns);
                           uz_in=uz_in, Tf_in=T_in, Ts_in=T_in, Y_in=[1.0, 0.0])

    s  = StateLayout(setup.nr, setup.nz, Ns)
    u0 = zeros(Tq, length(s))
    field_mat(u0, s, F_UZ) .= reshape(uz_in, :, 1)   # seed each row with its inlet uz
    field_mat(u0, s, F_TF) .= T_in
    field_mat(u0, s, F_TS) .= T_in
    species_mat(u0, s, 1)  .= one(Tq)

    # time scales (use the mean velocity U for the heuristics)
    U0     = primal_value(U)
    D_erg  = setup.μ * ergun_A(setup.eps_v, setup.dp_v) +
             setup.ρ * ergun_B(setup.eps_v, setup.dp_v) * U0
    t_flow = 50 * setup.ρ / (setup.eps_v * D_erg)
    t_spec = 5 * setup.L / U0

    u_ss = solve_wallloss_two_phase(u0, params, s;
                                    alpha=setup.α_wall, T_amb=setup.T_amb,
                                    β_flow=setup.β_flow, t_flow=t_flow,
                                    t_species=t_spec)

    # Stack the 6 physical fields (drop pressure) into (nr, nz, 6).
    out = Array{Tq,3}(undef, setup.nr, setup.nz, NFIELDS_OUT)
    out[:, :, 1] .= field_mat(u_ss, s, F_UR)
    out[:, :, 2] .= field_mat(u_ss, s, F_UZ)
    out[:, :, 3] .= field_mat(u_ss, s, F_TF)
    out[:, :, 4] .= field_mat(u_ss, s, F_TS)
    out[:, :, 5] .= species_mat(u_ss, s, 1)
    out[:, :, 6] .= species_mat(u_ss, s, 2)
    return out
end

const OUT_FIELD_NAMES = ["u_r", "u_z", "T_f", "T_s", "Y_A", "Y_B"]

# ── Wall-loss two-phase solve (same physics as example_wall_heatloss.jl, kept
#    self-contained so this module does not depend on that script being run) ──
const T_AMBIENT_DEFAULT = 293.5

struct WallLossParams{P}
    base::P
    s::StateLayout
    alpha::Float64
    T_amb::Float64
end

function rhs_wallloss!(du, u, wp::WallLossParams, t)
    p = wp.base
    rhs!(du, u, p, t)
    g  = p.grid; s = wp.s; pm = p.porous; nr = g.nr
    AoverV = 2 * g.R / (g.R^2 - g.rf[nr]^2)
    dTf = field_mat(du, s, F_TF); Tf = field_mat(u, s, F_TF)
    dTs = field_mat(du, s, F_TS); Ts = field_mat(u, s, F_TS)
    i = nr
    for j in 1:g.nz
        eps = pm.eps[i,j]
        qf = wp.alpha * (Tf[i,j] - wp.T_amb) * AoverV
        dTf[i,j] -= qf / (eps * p.fluid.rho * p.fluid.cp)
        qs = wp.alpha * (Ts[i,j] - wp.T_amb) * AoverV
        solid_cap = max((1 - eps) * p.solid.rho * p.solid.cp, 1e-20)
        dTs[i,j] -= qs / solid_cap
    end
    nothing
end

function solve_wallloss_two_phase(u0, params, s; alpha, T_amb,
                                  β_flow=100.0, t_flow=2.0, t_species=600.0)
    g = params.grid
    Tq = eltype(u0)
    # β_ac / gravity / D_species are Float64 (tied to the Float64 grid); only the
    # state and the BCs carry ForwardDiff partials.
    p_flow = ReactorParams(g, params.porous, params.fluid, params.solid,
                           params.reactions, Float64(β_flow), params.gravity,
                           params.D_species, params.bcs)
    u1 = solve_steady(u0, p_flow; t_end=t_flow).u[end]

    wp = WallLossParams(params, s, Float64(alpha), Float64(T_amb))
    function frozen_wallloss!(du, u, wp, t)
        rhs_wallloss!(du, u, wp, t)
        field_mat(du, s, F_UR) .= zero(eltype(du))
        field_mat(du, s, F_UZ) .= zero(eltype(du))
        field_mat(du, s, F_P)  .= zero(eltype(du))
    end
    prob = ODEProblem(frozen_wallloss!, u1, (zero(Tq), Tq(t_species)), wp)
    solve(prob, Tsit5(); abstol=1e-6, reltol=1e-5, save_everystep=false).u[end]
end
