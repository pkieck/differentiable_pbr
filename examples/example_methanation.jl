"""
example_methanation.jl — Catalytic methanation (Sabatier chemistry) on the
low-Mach variable-density solver, kinetics read from a CHEMKIN-II file.

Mirrors main_methanation.jl: dilute syngas (CO/H2/N2) over a Ni catalyst at
5 bar, T_in = 575 K. The exothermic surface reactions heat the solid → gas →
the gas expands; CO is consumed, CH4 + H2O produced along the bed.

Run from the project root:
    julia --project examples/example_methanation.jl

Produces examples/figures/methanation_fields.png — heatmaps of every state
field (u_r, u_z, p, T_f, T_s), all 6 species, and the density field ρ(r,z).
This is the richest example: it shows the reaction hot spot and the species
fronts developing down the bed.
"""

include("../src/compressible.jl")
include("../src/chemkin.jl")
include("plotting.jl")

using Printf
using Statistics: mean

function run_methanation_example()
    println("="^64)
    println("EXAMPLE: catalytic methanation on the compressible packed-bed solver")
    println("="^64)

    mech = read_chemkin(joinpath(@__DIR__, "..", "data", "methanation.inp"))
    println("Loaded ", mech)

    nr, nz = 6, 40               # finer than the 4×24 demo grid
    R, L   = 0.01, 0.15
    eps_v  = 0.4
    dp_v   = 2e-3
    p0     = 5.0e5
    Tf_in  = 575.0
    U_in   = 0.20
    μ      = 2.5e-5
    cp     = 1800.0
    λf     = 0.1

    g     = Grid2D(nr, nz, R, L)
    pm    = uniform_bed(nr, nz, eps_v, dp_v, 1.0)
    gas   = IdealGas(mech.M[end], μ, cp, λf, p0)
    solid = SolidProps(120.0, 600.0, 2.0)
    feed  = Dict("CO" => 0.04, "H2" => 0.12, "N2" => 0.84)

    params = mechanism_params(mech, g, pm, gas, solid, 50.0, 0.0;
                              D=2e-5, uz_in=U_in, Tf_in=Tf_in, Ts_in=Tf_in,
                              feed=feed)

    Ns   = nspecies(mech)
    s    = StateLayout(nr, nz, Ns)
    Y_in = params.bcs.Y_in
    rho_in = ideal_gas_density(p0, 1/sum(Y_in ./ mech.M), Tf_in)
    @printf "Feed (mass frac): %s\n" join(["$(mech.species[k])=$(round(Y_in[k],digits=3))" for k in 1:Ns], "  ")
    @printf "p0=%.1f bar  T_in=%.0f K  U_in=%.2f m/s  rho_in=%.3f kg/m^3\n" p0/1e5 Tf_in U_in rho_in

    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, Tf_in)
    set_field!(u0, s, F_TS, Tf_in)
    for k in 1:Ns; species_mat(u0, s, k) .= Y_in[k]; end

    D_erg  = μ * ergun_A(eps_v, dp_v) + rho_in * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * rho_in / (eps_v * D_erg)
    t_end  = 18 * L / U_in
    @printf "Solving (warm-start flow %.3fs, then coupled %.1fs)...\n" t_flow t_end
    @time u_ss = solve_steady_compressible_warmstart(u0, params;
                            β_flow=100.0, t_flow=t_flow, t_end=t_end)

    uz = field_mat(u_ss, s, F_UZ)
    Tf = field_mat(u_ss, s, F_TF)
    Ts = field_mat(u_ss, s, F_TS)
    ρ  = density_field(u_ss, params)
    area = [(g.rf[i+1]^2 - g.rf[i]^2) * π for i in 1:nr]
    Yk   = [field_mat(u_ss, s, 5+k) for k in 1:Ns]
    molflux(k, j) = sum(ρ[:,j] .* uz[:,j] .* Yk[k][:,j] .* area) / mech.M[k]
    iCO = species_index(mech, "CO")
    X_CO = 1 - molflux(iCO, nz) / molflux(iCO, 1)
    @printf "  CO conversion = %.1f %%,  peak solid T = %.1f K,  u_out/u_in = %.2f\n" 100*X_CO maximum(Ts) (mean(uz[:,nz])/U_in)

    fig = plot_state(u_ss, s, g;
                     species_names=copy(mech.species),
                     density=ρ,
                     title=@sprintf("Catalytic methanation  (CO conv %.0f%%, hot spot %.0f K)",
                                    100*X_CO, maximum(Ts)))
    save_fig(fig, "methanation_fields.png")
    return u_ss
end

run_methanation_example()
