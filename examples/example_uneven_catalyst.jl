"""
example_uneven_catalyst.jl — Exothermic A→B reactor with a SPATIALLY GRADED
catalyst-activity field φ_cat(r,z).

Same chemistry/flow as example_reactor.jl, but the catalyst activity is no
longer uniform: it ramps up along the bed (inert entry length, then loaded)
and is concentrated near the axis (activity loss toward the wall). φ_cat
multiplies BOTH the reaction heat source and the species source in the RHS
(src/rhs.jl), so the reaction — and therefore the hot zone and the conversion
front — follows the catalyst. No solver changes are needed; only the
PorousMedium's φ_cat field is made non-uniform.

Run from the project root:
    julia --project examples/example_uneven_catalyst.jl

Produces examples/figures/uneven_catalyst_fields.png. The catalyst field itself
is shown alongside the state fields so the localisation is visible.
"""

include("../src/ad.jl")
include("plotting.jl")

using Printf
using Statistics: mean

function run_uneven_catalyst()
    println("="^60)
    println("EXAMPLE: exothermic reactor with a graded catalyst field φ_cat(r,z)")
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

    g = Grid2D(nr, nz, Float64(R), Float64(L))

    # Graded catalyst activity: inert entry (z < 0.2L), then ramps to full along
    # the bed, and decays radially toward the wall (concentrated near the axis).
    phi = zeros(nr, nz)
    for j in 1:nz, i in 1:nr
        axial  = clamp((g.z[j] - 0.2L) / (0.6L), 0.0, 1.0)   # 0 → 1 over 0.2–0.8 L
        radial = exp(-3.0 * (g.r[i] / R)^2)                  # core-weighted
        phi[i,j] = axial * radial
    end
    pm = PorousMedium(fill(eps_v, nr, nz), fill(dp_v, nr, nz), phi)

    rxn = Reaction(Float64(A_pre), 0.0, (-1.0, 1.0), Float64(dH),
                   (M_A, M_B), (true, false))
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
    @printf "  grid %d×%d,  φ_cat graded (axial ramp × core-weighted),  t_species=%.0f s\n" nr nz t_spec

    @time u_ss = solve_steady_two_phase(u0, params;
                                        β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    X   = 1 - mean(species_mat(u_ss, s, 1)[:, nz])
    Tfo = mean(field_mat(u_ss, s, F_TF)[:, nz])
    @printf "  outlet conversion X = %.3f,  T_f = %.2f K,  peak T_s = %.2f K\n" X Tfo maximum(field_mat(u_ss, s, F_TS))

    # The catalyst field is shown as an extra panel alongside the state fields.
    fig = plot_state(u_ss, s, g;
                     species_names=["Y_A (reactant)", "Y_B (product)"],
                     extras=["φ_cat  [activity]" => phi],
                     title=@sprintf("Graded catalyst φ_cat(r,z)  —  reaction follows the active zone  (X=%.2f)", X))
    save_fig(fig, "uneven_catalyst_fields.png")
    return u_ss
end

run_uneven_catalyst()
