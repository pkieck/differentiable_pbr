"""
main_methanation.jl - A literature reaction system (catalytic methanation /
Sabatier chemistry) running on the compressible (low-Mach, variable-density)
packed-bed solver, with the kinetics read from a CHEMKIN-II input file.

Chemistry (data/methanation.inp, global Ni-catalyst kinetics):
    R1  CO + 3 H2 -> CH4 + H2O      dH298 ~= -206 kJ/mol
    R2  CO2 + 4 H2 -> CH4 + 2 H2O    dH298 ~= -165 kJ/mol
dH and molar masses come from the NASA-7 THERMO cards in that file (GRI-Mech
3.0 thermodynamics), so nothing is hard-coded here.

This exercises the full reactive-compressible path:
  - CHEMKIN parsing -> Mechanism (chemkin.jl / kinetics.jl)
  - exothermic surface reaction heats the solid -> interphase HT -> gas heats
  - gas expands at fixed p0 (rho down, u up) - the low-Mach variable-density coupling
  - CO is consumed, CH4/H2O produced along the bed

Quantitative checks (printed PASS/FAIL):
  (V1) Element balance (C, H, O atom flows in = out)        - transport/stoich
  (V2) Mass-flux conservation integral(rho*u*dA) = const over z planes  - div(rho*u)=0
  (V3) Adiabatic energy balance  mdot*cp*dT_f ~= sum_rxn(-dH)*(rate*V)
  (V4) Thermal expansion  u_out/u_in > 1 with a reaction hot spot
"""

include("src/compressible.jl")   # solver + kinetics accessors
include("src/chemkin.jl")        # read_chemkin (kinetics.jl already loaded)

using Printf
using LinearAlgebra
using Statistics: mean

function run_methanation()
    println("="^64)
    println("CATALYTIC METHANATION on the compressible packed-bed solver")
    println("="^64)

    # ── 1. Kinetics from the CHEMKIN file ───────────────────────────────────
    mech = read_chemkin(joinpath(@__DIR__, "data", "methanation.inp"))
    println("Loaded ", mech)
    for (j, r) in enumerate(mech.reactions)
        @printf "  R%d:  A=%.2e  beta=%.1f  Ea=%.0f kJ/mol  dH=%+.1f kJ/mol\n" j r.A r.beta r.Ea/1e3 r.dH/1e3
    end

    # ── 2. Geometry, bed, gas ───────────────────────────────────────────────
    nr, nz  = 4, 24
    R, L    = 0.01, 0.15
    eps_v   = 0.4
    dp_v    = 2e-3
    p0      = 5.0e5            # 5 bar (methanation is run at elevated pressure)
    Tf_in   = 575.0           # K — onset of activity
    U_in    = 0.20            # m/s superficial
    μ       = 2.5e-5
    cp      = 1800.0          # J/(kg·K), H2-rich mixture (mass-based, constant)
    λf      = 0.1

    g     = Grid2D(nr, nz, R, L)
    pm    = uniform_bed(nr, nz, eps_v, dp_v, 1.0)
    gas   = IdealGas(mech.M[end], μ, cp, λf, p0)   # default Mbar unused (Ns>0)
    # Light solid thermal mass: steady state is independent of rhos*cps, and a
    # realistic (heavy) catalyst would need a very long pseudo-transient.
    solid = SolidProps(120.0, 600.0, 2.0)

    # Dilute syngas feed: H2:CO = 3 (stoichiometric for R1), heavy N2 dilution
    # to keep the adiabatic temperature rise to a tractable hot-spot.
    feed = Dict("CO" => 0.04, "H2" => 0.12, "N2" => 0.84)

    params = mechanism_params(mech, g, pm, gas, solid, 50.0, 0.0;
                              D=2e-5, uz_in=U_in, Tf_in=Tf_in, Ts_in=Tf_in,
                              feed=feed)

    Ns   = nspecies(mech)
    s    = StateLayout(nr, nz, Ns)
    Y_in = params.bcs.Y_in
    rho_in = ideal_gas_density(p0, 1/sum(Y_in ./ mech.M), Tf_in)
    @printf "\nFeed (mass frac): %s\n" join(["$(mech.species[k])=$(round(Y_in[k],digits=3))" for k in 1:Ns], "  ")
    @printf "p0=%.1f bar  T_in=%.0f K  U_in=%.2f m/s  rho_in=%.3f kg/m^3\n\n" p0/1e5 Tf_in U_in rho_in

    # -- 3. Initial state (cold feed everywhere) & solve -----------------
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

    # ── 4. Post-process ─────────────────────────────────────────────────────
    uz = field_mat(u_ss, s, F_UZ)
    Tf = field_mat(u_ss, s, F_TF)
    Ts = field_mat(u_ss, s, F_TS)
    ρ  = density_field(u_ss, params)
    Yk = [field_mat(u_ss, s, 5+k) for k in 1:Ns]

    area = [(g.rf[i+1]^2 - g.rf[i]^2) * π for i in 1:nr]
    A_tot = sum(area)

    # plane molar flux of species k: integral(rho u Y_k /M_k) dA   [mol/s]
    molflux(k, j) = sum(ρ[:,j] .* uz[:,j] .* Yk[k][:,j] .* area) / mech.M[k]
    plane_mass(j) = sum(ρ[:,j] .* uz[:,j] .* area)

    iCO, iH2, iCH4, iH2O, iN2, iCO2 = (species_index(mech, n) for n in
                                        ("CO","H2","CH4","H2O","N2","CO2"))

    F_CO_in,  F_CO_out  = molflux(iCO, 1),  molflux(iCO, nz)
    F_CH4_in, F_CH4_out = molflux(iCH4,1),  molflux(iCH4,nz)
    X_CO = 1 - F_CO_out / F_CO_in

    Tf_out = sum(Tf[:,nz] .* area) / A_tot
    uz_out = sum(uz[:,nz] .* area) / A_tot

    println("\n-- Reactor outlet --")
    @printf "  CO conversion          : %5.1f %%\n" 100*X_CO
    @printf "  CH4 produced           : %.3e mol/s (in %.3e)\n" F_CH4_out F_CH4_in
    @printf "  Peak solid T (hot spot): %6.1f K  at z=%.3f m\n" maximum(Ts) g.z[argmax(vec(maximum(Ts,dims=1)))]
    @printf "  Outlet fluid  T        : %6.1f K  (inlet %.0f K, dT=%.1f K)\n" Tf_out Tf_in Tf_out-Tf_in
    @printf "  Outlet u_z (mean)      : %.4f m/s (inlet %.4f, ratio %.2f)\n" uz_out U_in uz_out/U_in

    # ── V1: element balance (atoms/s) — C in CO,CO2,CH4 ; etc. ──────────────
    atoms = Dict("C"=>Dict(iCO=>1, iCO2=>1, iCH4=>1),
                 "H"=>Dict(iH2=>2, iCH4=>4, iH2O=>2),
                 "O"=>Dict(iCO=>1, iCO2=>2, iH2O=>1))
    println("\n-- (V1) Atom balance in->out --")
    elem_err = 0.0
    for (el, comp) in atoms
        fin  = sum(c*molflux(k,1)  for (k,c) in comp)
        fout = sum(c*molflux(k,nz) for (k,c) in comp)
        e = abs(fout-fin)/abs(fin)
        elem_err = max(elem_err, e)
        @printf "  %s : in=%.4e  out=%.4e  rel.err=%.2e\n" el fin fout e
    end

    # ── V2: mass-flux conservation over z planes ────────────────────────────
    md = [plane_mass(j) for j in 1:nz]
    mass_dev = (maximum(md)-minimum(md))/mean(md)

    # ── V3: adiabatic energy balance  ṁ cp ΔT  vs  Σ(−ΔH)·rate·V ────────────
    Qchem = 0.0
    for j in 1:nz, i in 1:nr
        Vc = area[i]*g.dz
        Yc = ntuple(l -> Yk[l][i,j], Ns)
        for rxn in mech.reactions
            Qchem += pm.phi_cat[i,j]*rxn_negdH(rxn)*rxn_rate(rxn, Ts[i,j], ρ[i,j], Yc)*Vc
        end
    end
    mdot   = plane_mass(nz)
    Qsens  = mdot * cp * (Tf_out - Tf_in)
    en_err = abs(Qchem - Qsens)/abs(Qchem)

    println("\n-- (V2/V3) Conservation --")
    @printf "  mass-flux plane spread : %.2e        (div(rho*u)=0)\n" mass_dev
    @printf "  chem heat release Qchem: %.2f W\n" Qchem
    @printf "  sensible gas  mdot cp dT  : %.2f W\n" Qsens
    @printf "  energy-balance rel.err : %.2e\n" en_err

    # ── Verdicts ────────────────────────────────────────────────────────────
    println("\n-- Verdict --")
    v1 = elem_err < 0.02
    v2 = mass_dev < 0.02
    v3 = en_err   < 0.10
    v4 = uz_out/U_in > 1.05 && maximum(Ts) > Tf_in + 20 && X_CO > 0.05
    for (name, ok) in (("V1 atom balance", v1), ("V2 mass-flux div(rho*u)=0", v2),
                       ("V3 energy balance", v3), ("V4 hot spot + expansion", v4))
        @printf "  %-26s %s\n" name (ok ? "PASS" : "FAIL")
    end
    println()
    return (; X_CO, Tf_out, elem_err, mass_dev, en_err, allpass = v1&&v2&&v3&&v4)
end

function test_gradient_sanity_methanation()
    println("="^64)
    println("METHANATION GRADIENT SANITY CHECK (A_pre of reaction 1)")
    println("="^64)

    mech = read_chemkin(joinpath(@__DIR__, "data", "methanation.inp"))

    nr, nz  = 3, 10
    R, L    = 0.01, 0.10
    eps_v   = 0.4
    dp_v    = 2e-3
    p0      = 5.0e5
    Tf_in   = 575.0
    U_in    = 0.20
    μ       = 2.5e-5
    cp      = 1800.0
    λf      = 0.1

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

    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, Tf_in)
    set_field!(u0, s, F_TS, Tf_in)
    for k in 1:Ns
        species_mat(u0, s, k) .= Y_in[k]
    end

    D_erg  = μ * ergun_A(eps_v, dp_v) + ideal_gas_density(p0, 1/sum(Y_in ./ mech.M), Tf_in) * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ideal_gas_density(p0, 1/sum(Y_in ./ mech.M), Tf_in) / (eps_v * D_erg)
    t_end  = 10 * L / U_in

    println("Step 1: warm-started forward solve")
    @time u_warm = solve_steady_compressible_warmstart(u0, params;
                                                      β_flow=100.0, t_flow=t_flow, t_end=t_end)

    println("Step 2: Newton-Krylov steady-state refinement")
    @time sol_nk = solve_steady_compressible_nk(u_warm, params; tol=1e-8, maxiters=40, verbose=false)
    u_star = sol_nk.u

    function loss_CH4(uvec)
        mean(field_mat(uvec, s, 5 + species_index(mech, "CH4"))[:, nz])
    end

    θ_getter(p) = [p.reactions[1].A]
    function θ_setter(p, θ_new)
        new_rxns = ntuple(k -> k == 1 ? GlobalReaction(θ_new[1], p.reactions[k].beta,
                                                       p.reactions[k].Ea, p.reactions[k].nu,
                                                       p.reactions[k].order, p.reactions[k].dH,
                                                       p.reactions[k].M) : p.reactions[k],
                          length(p.reactions))
        CompressibleParams(p.grid, p.porous, p.gas, p.solid, new_rxns,
                           p.beta_ac, p.gravity, copy(p.D_species), copy(p.M_species);
                           uz_in=p.bcs.uz_in, Tf_in=p.bcs.Tf_in, Ts_in=p.bcs.Ts_in,
                           Y_in=copy(p.bcs.Y_in))
    end

    println("Step 3: IFT gradient vs finite-difference re-solve")
    @time g_ift = ift_gradient(loss_CH4, u_star, params, θ_getter, θ_setter)

    h_fd = 1e-4 * θ_getter(params)[1]
    θ_p  = [θ_getter(params)[1] + h_fd]
    params_p = θ_setter(params, θ_p)
    u_p_warm = solve_steady_compressible_warmstart(u0, params_p;
                                                   β_flow=100.0, t_flow=t_flow, t_end=t_end)
    u_p = solve_steady_compressible_nk(u_p_warm, params_p; tol=1e-8, maxiters=40, verbose=false).u
    g_fd = [(loss_CH4(u_p) - loss_CH4(u_star)) / h_fd]

    rel_err = abs(g_ift[1] - g_fd[1]) / max(abs(g_fd[1]), 1e-12)
    @printf "  IFT  dCH4/dA_pre : %.4e\n" g_ift[1]
    @printf "  FD   dCH4/dA_pre : %.4e\n" g_fd[1]
    @printf "  relative error   : %.2e\n" rel_err
    println(rel_err < 0.10 ? "  PASS" : "  FAIL")
    println()
    return rel_err
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_methanation()
end
