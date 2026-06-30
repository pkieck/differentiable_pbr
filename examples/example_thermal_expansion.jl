"""
example_thermal_expansion.jl — Thermal expansion in an exothermic gas-phase bed
(low-Mach VARIABLE-DENSITY / compressible solver).

Mirrors verification case Test C2 in main_compressible.jl: cold gas (300 K)
enters a catalytic bed where the exothermic A→B reaction heats it; at fixed
thermodynamic pressure p₀ the gas expands (ρ↓, u↑) while ∇·(ρu)=0 holds.

Run from the project root:
    julia --project examples/example_thermal_expansion.jl

Produces examples/figures/thermal_expansion_fields.png — heatmaps of every
state field plus the density field ρ(r,z).
"""

include("../src/compressible.jl")
include("plotting.jl")

using Printf
using Statistics: mean

function run_thermal_expansion()
    println("="^60)
    println("EXAMPLE: thermal expansion in an exothermic gas bed (ρ↓ ⇒ u↑)")
    println("="^60)

    nr, nz = 6, 36                # finer than the 3×12 test grid
    R, L   = 0.02, 0.1
    U_in   = 0.1
    eps_v  = 0.5
    dp_v   = 5e-3
    T0     = 300.0
    μ      = 2.0e-5
    cp     = 1000.0
    λf     = 0.05
    Mbar   = 0.029
    p0     = 1.0e5
    M_A, M_B = 0.029, 0.029
    dH     = -1.0e4
    Da     = 1.5
    A_pre  = Da * U_in / L
    ρ_in   = p0 * Mbar / (R_GAS * T0)

    rxn = Reaction(Float64(A_pre), 0.0, (-1.0, 1.0), Float64(dH),
                   (M_A, M_B), (true, false))

    g     = Grid2D(nr, nz, R, L)
    pm    = uniform_bed(nr, nz, eps_v, dp_v, 1.0)
    gas   = IdealGas(Mbar, μ, cp, λf, p0)
    solid = SolidProps(50.0, 200.0, 1.0)        # light solid → fast steady state
    Ns    = 2
    Msp   = [M_A, M_B]
    β     = 50.0
    params = CompressibleParams(g, pm, gas, solid, (rxn,), β, 0.0, fill(1e-5, Ns), Msp;
                                uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[1.0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= 1.0

    D_erg  = μ * ergun_A(eps_v, dp_v) + ρ_in * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ρ_in / (eps_v * D_erg)
    t_end  = 10 * L / U_in
    @printf "  grid %d×%d,  ρ_in=%.3f kg/m³,  Da=%.1f,  t_flow=%.3f s,  t_end=%.1f s\n" nr nz ρ_in Da t_flow t_end

    @time u_ss = solve_steady_compressible_warmstart(u0, params;
                                                     β_flow=100.0, t_flow=t_flow, t_end=t_end)

    ρ_ss = density_field(u_ss, params)
    Tfo  = mean(field_mat(u_ss, s, F_TF)[:, nz])
    uzo  = mean(field_mat(u_ss, s, F_UZ)[:, nz])
    @printf "  outlet T_f = %.1f K  (inlet %.0f K),  u_z = %.4f m/s  (u_out/u_in = %.2f)\n" Tfo T0 uzo (uzo/U_in)

    fig = plot_state(u_ss, s, g;
                     species_names=["Y_A (reactant)", "Y_B (product)"],
                     density=ρ_ss,
                     title=@sprintf("Thermal expansion, exothermic gas bed  (T: %.0f→%.0f K, u↑×%.2f)",
                                    T0, Tfo, uzo/U_in))
    save_fig(fig, "thermal_expansion_fields.png")
    return u_ss
end

run_thermal_expansion()
