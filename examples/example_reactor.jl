"""
example_reactor.jl — Non-isothermal exothermic packed-bed reactor (incompressible).

Mirrors the verification case Test 5b in main.jl: a first-order exothermic
reaction A→B (Da = 1, ΔH = −5000 J/mol) in a liquid-filled packed bed, solved
with the two-phase operator split (equilibrate flow, then transport species +
energy on the frozen velocity).

Run from the project root:
    julia --project examples/example_reactor.jl

Produces examples/figures/reactor_fields.png with (r,z) heatmaps of every
state field: u_r, u_z, p, T_f, T_s, Y_A, Y_B.
"""

include("../src/ad.jl")        # → solve.jl → rhs/fields/closures/params/grid
include("plotting.jl")

using Printf
using Statistics: mean

function run_reactor()
    println("="^60)
    println("EXAMPLE: non-isothermal exothermic packed-bed reactor (A→B)")
    println("="^60)

    nr, nz = 8, 40                # finer than the test grid → nicer fields
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
    YA0    = 1.0

    rxn = Reaction(Float64(A_pre), 0.0, (-1.0, 1.0), Float64(dH),
                   (M_A, M_B), (true, false))

    g     = Grid2D(nr, nz, Float64(R), Float64(L))
    pm    = uniform_bed(nr, nz, Float64(eps_v), Float64(dp_v), 1.0)
    fluid = FluidProps(ρ, μ, cpf, λf)
    solid = SolidProps(2000.0, 800.0, 1.0)
    Ns    = 2
    params = ReactorParams(g, pm, fluid, solid, (rxn,), U_in, 0.0, fill(1e-5, Ns);
                           uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[YA0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= YA0

    D_erg  = μ * ergun_A(eps_v, dp_v) + ρ * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ρ / (eps_v * D_erg)
    t_spec = 5 * L / U_in
    @printf "  grid %d×%d,  Da=%.1f,  ΔH=%.0f J/mol,  t_flow=%.3f s,  t_species=%.0f s\n" nr nz Da dH t_flow t_spec

    @time u_ss = solve_steady_two_phase(u0, params;
                                        β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    X    = 1 - mean(species_mat(u_ss, s, 1)[:, nz]) / YA0
    Tfo  = mean(field_mat(u_ss, s, F_TF)[:, nz])
    Tso  = mean(field_mat(u_ss, s, F_TS)[:, nz])
    @printf "  outlet conversion X = %.3f,  T_f = %.2f K,  T_s = %.2f K  (inlet %.0f K)\n" X Tfo Tso T0

    fig = plot_state(u_ss, s, g;
                     species_names=["Y_A (reactant)", "Y_B (product)"],
                     title=@sprintf("Exothermic packed-bed reactor A→B  (Da=%.1f, X=%.2f)", Da, X))
    save_fig(fig, "reactor_fields.png")
    return u_ss
end

run_reactor()
