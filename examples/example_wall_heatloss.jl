"""
example_wall_heatloss.jl — Non-adiabatic packed-bed reactor with convective
heat loss through the wall:  q_wall = α (T − T_ambient),  T_ambient = 293.5 K.

The validated solver treats the wall as adiabatic (Neumann, zero radial heat
flux). Here we add a Robin wall condition WITHOUT touching the validated RHS:
a thin wrapper `rhs_wallloss!` calls the original `rhs!` and then adds the wall
heat-loss sink to the fluid- and solid-energy rows of the outermost (i = nr)
cells only. The flux α(T−T_amb) [W/m²] through the wall face (area 2πR·dz) is
turned into a volumetric sink [W/m³] by dividing by the cell volume, exactly as
the interphase term h·aᵥ·(T_s−T_f) is handled in src/rhs.jl.

Same exothermic A→B chemistry as example_reactor.jl. With the wall cooled, the
flat radial temperature profile of the adiabatic case is replaced by a hot core
and a cooled near-wall layer — the classic non-adiabatic reactor radial profile.

Run from the project root:
    julia --project examples/example_wall_heatloss.jl

Produces examples/figures/wall_heatloss_fields.png.
"""

include("../src/ad.jl")        # → solve.jl → rhs!  (+ ODEProblem/Tsit5/solve)
include("plotting.jl")

using DifferentialEquations: ODEProblem, solve, Tsit5
using Printf
using Statistics: mean

const T_AMBIENT = 293.5        # K

"""
Wrapper params bundling the base ReactorParams with the wall-loss coefficient.
"""
struct WallLossParams{P}
    base::P
    s::StateLayout
    alpha::Float64             # wall heat-transfer coefficient [W/(m²·K)]
    T_amb::Float64
end

"""
    rhs_wallloss!(du, u, wp, t)

Validated RHS plus a Robin wall heat-loss sink on the i = nr cells.
"""
function rhs_wallloss!(du, u, wp::WallLossParams, t)
    p = wp.base
    rhs!(du, u, p, t)

    g  = p.grid
    s  = wp.s
    pm = p.porous
    nr = g.nr

    # wall area / cell volume for an outer-ring cell:  2πR·dz / (π(R²−rf[nr]²)·dz)
    AoverV = 2 * g.R / (g.R^2 - g.rf[nr]^2)

    dTf = field_mat(du, s, F_TF); Tf = field_mat(u, s, F_TF)
    dTs = field_mat(du, s, F_TS); Ts = field_mat(u, s, F_TS)

    i = nr
    for j in 1:g.nz
        eps = pm.eps[i,j]
        # fluid phase loses heat through the wall
        qf = wp.alpha * (Tf[i,j] - wp.T_amb) * AoverV
        dTf[i,j] -= qf / (eps * p.fluid.rho * p.fluid.cp)
        # solid phase loses heat through the wall
        qs = wp.alpha * (Ts[i,j] - wp.T_amb) * AoverV
        solid_cap = max((1 - eps) * p.solid.rho * p.solid.cp, 1e-20)
        dTs[i,j] -= qs / solid_cap
    end
    nothing
end

"""
Two-phase solve (equilibrate flow, then frozen-velocity energy+species) with the
wall-loss term active in phase 2 — mirrors solve_steady_two_phase but swaps the
phase-2 RHS for `rhs_wallloss!`.
"""
function solve_wallloss_two_phase(u0, params, s; alpha, T_amb,
                                  β_flow=100.0, t_flow=2.0, t_species=600.0)
    g = params.grid
    # Phase 1: equilibrate momentum + pressure (wall loss does not affect flow).
    p_flow = ReactorParams(g, params.porous, params.fluid, params.solid,
                           params.reactions, β_flow, params.gravity,
                           params.D_species, params.bcs)
    u1 = solve_steady(u0, p_flow; t_end=t_flow).u[end]

    # Phase 2: frozen velocity/pressure, energy + species with wall loss.
    wp = WallLossParams(params, s, alpha, T_amb)
    function frozen_wallloss!(du, u, wp, t)
        rhs_wallloss!(du, u, wp, t)
        field_mat(du, s, F_UR) .= 0.0
        field_mat(du, s, F_UZ) .= 0.0
        field_mat(du, s, F_P)  .= 0.0
    end
    prob = ODEProblem(frozen_wallloss!, u1, (0.0, Float64(t_species)), wp)
    solve(prob, Tsit5(); abstol=1e-6, reltol=1e-5, save_everystep=false).u[end]
end

function run_wall_heatloss()
    println("="^60)
    println("EXAMPLE: non-adiabatic reactor — wall heat loss α(T−T_ambient)")
    println("="^60)

    nr, nz = 8, 40
    R, L   = 0.05, 1.0
    U_in   = 0.01
    eps_v  = 0.4
    dp_v   = 3e-3
    T0     = 400.0
    μ      = 1e-3
    ρ      = 1000.0
    M_A, M_B = 0.028, 0.028
    cpf    = 1000.0
    λf     = 0.03
    dH     = -5000.0
    Da     = 1.0
    A_pre  = Da * U_in / L
    α_wall = 20.0                  # W/(m²·K) — moderate wall cooling

    rxn = Reaction(Float64(A_pre), 0.0, (-1.0, 1.0), Float64(dH),
                   (M_A, M_B), (true, false))
    g     = Grid2D(nr, nz, Float64(R), Float64(L))
    pm    = uniform_bed(nr, nz, Float64(eps_v), Float64(dp_v), 1.0)
    fluid = FluidProps(ρ, μ, cpf, λf)
    solid = SolidProps(2000.0, 800.0, 1.0)
    Ns    = 2
    params = ReactorParams(g, pm, fluid, solid, (rxn,), U_in, 0.0, fill(1e-5, Ns);
                           uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[1.0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= 1.0

    D_erg  = μ * ergun_A(eps_v, dp_v) + ρ * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ρ / (eps_v * D_erg)
    t_spec = 5 * L / U_in
    @printf "  grid %d×%d,  α_wall=%.0f W/(m²·K),  T_ambient=%.1f K,  t_species=%.0f s\n" nr nz α_wall T_AMBIENT t_spec

    @time u_ss = solve_wallloss_two_phase(u0, params, s;
                                          alpha=α_wall, T_amb=T_AMBIENT,
                                          β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    Tf  = field_mat(u_ss, s, F_TF)
    X   = 1 - mean(species_mat(u_ss, s, 1)[:, nz])
    Tf_core = Tf[1, nz]      # axis, outlet
    Tf_wall = Tf[nr, nz]     # wall, outlet
    @printf "  outlet conversion X = %.3f\n" X
    @printf "  outlet T_f: core(axis) = %.1f K,  wall = %.1f K  (radial ΔT = %.1f K)\n" Tf_core Tf_wall (Tf_core - Tf_wall)

    fig = plot_state(u_ss, s, g;
                     species_names=["Y_A (reactant)", "Y_B (product)"],
                     title=@sprintf("Wall heat loss α(T−%.1f K), α=%.0f W/m²K  —  hot core, cooled wall (ΔT_r=%.0f K)",
                                    T_AMBIENT, α_wall, Tf_core - Tf_wall))
    save_fig(fig, "wall_heatloss_fields.png")
    return u_ss
end

run_wall_heatloss()
