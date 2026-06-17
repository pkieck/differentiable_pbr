"""
main_compressible.jl — Verification tests for the low-Mach VARIABLE-DENSITY
(compressible) packed-bed solver (src/compressible.jl).

  Test C0: Reduction check  — rhs_compressible! ≡ rhs! when T & composition are
                              uniform (ρ constant).  Rigorous: shares all of the
                              already-verified incompressible physics.
  Test C1: Isothermal flow  — Hagen-Poiseuille parabola recovered by the
                              compressible solver at uniform T (sanity / Test 1
                              parity through the new code path).
  Test C2: Thermal expansion — exothermic A→B heats the gas; the gas expands and
                              accelerates.  Checks (a) ∇·(ρu)=0 (plane mass-flux
                              conservation) and (b) ideal-gas u_out/u_in = T_out/T_in.
"""

include("src/compressible.jl")

using Printf
using LinearAlgebra
using Statistics: mean
using Random: MersenneTwister

# ── Test C0: reduction to the incompressible solver ────────────────────────
#
# With T_f and composition uniform, the EOS gives a constant ρ, and EVERY term
# of rhs_compressible! must reduce identically to rhs! evaluated at that ρ —
# for ARBITRARY velocity/pressure fields and ARBITRARY (spatially varying) ε.
# This inherits all of Tests 1–7's correctness for free.
# ──────────────────────────────────────────────────────────────────────────

function test_reduction()
    println("=" ^ 60)
    println("TEST C0: compressible RHS reduces to incompressible (uniform T)")
    println("=" ^ 60)

    nr, nz = 6, 14
    R, L   = 0.02, 0.1
    T0     = 400.0
    Mbar   = 0.029
    p0     = 1.0e5
    μ      = 2.0e-5
    cp     = 1000.0
    λf     = 0.05
    ρ0     = p0 * Mbar / (R_GAS * T0)      # exact EOS density at T0

    g  = Grid2D(nr, nz, R, L)

    # Spatially-varying ε (exercises the per-cell scalar / masking path too)
    eps_field = fill(0.4, nr, nz)
    for j in 1:nz, i in 1:nr
        if g.r[i] > 0.6R
            eps_field[i,j] = 0.15
        end
    end
    pm    = PorousMedium(eps_field, fill(3e-3, nr, nz), ones(nr, nz))
    solid = SolidProps(2000.0, 800.0, 1.0)

    M_A, M_B = 0.029, 0.029
    rxn = Reaction(0.7, 1000.0, (-1.0, 1.0), -5000.0, (M_A, M_B), (true, false))
    Ns  = 2
    Dsp = fill(1e-5, Ns)
    Msp = [M_A, M_B]
    U_in = 0.1
    β    = 50.0
    grav = 9.81

    fluid = FluidProps(ρ0, μ, cp, λf)
    gas   = IdealGas(Mbar, μ, cp, λf, p0)

    p_inc = ReactorParams(g, pm, fluid, solid, (rxn,), β, grav, Dsp;
                          uz_in=U_in, Tf_in=T0, Ts_in=350.0, Y_in=[1.0, 0.0])
    p_cmp = CompressibleParams(g, pm, gas, solid, (rxn,), β, grav, Dsp, Msp;
                               uz_in=U_in, Tf_in=T0, Ts_in=350.0, Y_in=[1.0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    rng = MersenneTwister(1234)
    u  = zero_state(s)
    # arbitrary velocity & pressure fields (so divergence, drag, viscous ≠ 0)
    field_mat(u, s, F_UR) .= 0.01 .* randn(rng, nr, nz)
    field_mat(u, s, F_UZ) .= U_in .+ 0.02 .* randn(rng, nr, nz)
    field_mat(u, s, F_P)  .= 10.0 .* randn(rng, nr, nz)
    # uniform T_f and composition ⇒ uniform ρ (the reduction hypothesis);
    # T_s may vary (it does not enter the EOS).
    field_mat(u, s, F_TF) .= T0
    field_mat(u, s, F_TS) .= 350.0 .+ 5.0 .* randn(rng, nr, nz)
    species_mat(u, s, 1)  .= 1.0
    species_mat(u, s, 2)  .= 0.0

    du_inc = similar(u); rhs!(du_inc, u, p_inc, 0.0)
    du_cmp = similar(u); rhs_compressible!(du_cmp, u, p_cmp, 0.0)

    # Per-field relative difference
    fields = ["u_r", "u_z", "p", "T_f", "T_s", "Y_A", "Y_B"]
    maxrel = 0.0
    for f in 1:(5+Ns)
        a = field_mat(du_inc, s, f)
        b = field_mat(du_cmp, s, f)
        scale = max(maximum(abs, a), maximum(abs, b), 1e-30)
        rel = maximum(abs, a .- b) / scale
        @printf "  %-4s row  max rel diff: %.2e\n" fields[f] rel
        maxrel = max(maxrel, rel)
    end
    @printf "  overall max rel diff: %.2e\n" maxrel
    pass = maxrel < 1e-10
    println(pass ? "  PASS" : "  FAIL")
    println()
    return maxrel
end

# ── Test C1: isothermal Poiseuille through the compressible code path ───────
function test_compressible_poiseuille()
    println("=" ^ 60)
    println("TEST C1: isothermal Poiseuille (compressible solver, uniform T)")
    println("=" ^ 60)

    nr, nz  = 8, 20
    R, L    = 0.01, 0.10
    U_mean  = 0.01
    μ       = 1.0e-3
    Mbar    = 0.029
    p0      = 1.0e5
    T0      = 300.0
    ρ0      = p0 * Mbar / (R_GAS * T0)
    # The artificial sound speed is c_ac = β·ε (independent of ρ).  For a
    # low-density gas, setting β = U_mean (the liquid recipe) makes the acoustic
    # transit L/(βε) = 10 s, so reaching steady state needs ~50 transits = 500 s.
    # Instead pick β ≫ U_mean so the pressure equilibrates in a few seconds; the
    # steady state is independent of β.
    β_ac    = 1.0
    t_end   = 5.0          # ≫ both the viscous time (0.12 s) and ~50 transits

    g      = Grid2D(nr, nz, R, L)
    pm     = PorousMedium(ones(nr,nz), ones(nr,nz), zeros(nr,nz))   # open tube
    gas    = IdealGas(Mbar, μ, 1000.0, 0.05, p0)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    params = CompressibleParams(g, pm, gas, solid, (), β_ac, 0.0, Float64[], Float64[];
                                uz_in=U_mean, Tf_in=T0, Ts_in=T0)

    s  = StateLayout(nr, nz, 0)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_mean)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)

    @printf "  ρ(EOS) = %.4f kg/m³,  t_end = %.3f s\n" ρ0 t_end
    @time sol = solve_steady_compressible(u0, params; t_end=t_end)
    println("  Return code: $(sol.retcode)")

    u_ss   = sol.u[end]
    uz_ss  = field_mat(u_ss, s, F_UZ)
    uz_sim = uz_ss[:, nz]
    uz_exact = @. 2 * U_mean * (1 - (g.r / R)^2)
    L2_err = norm(uz_sim .- uz_exact) / norm(uz_exact)

    @printf "  L2 error vs parabola: %.2e\n" L2_err
    println(L2_err < 0.05 ? "  PASS" : "  FAIL")
    println()
    return L2_err
end

# ── Test C2: thermal expansion in an exothermic packed bed ─────────────────
#
# Cold gas (300 K) enters a catalytic bed; A→B is exothermic so the gas heats
# and, at the fixed thermodynamic pressure p₀, EXPANDS: ρ↓, u↑.  Two checks:
#
#   (a) Mass conservation ∇·(ρu)=0  ⇒  the plane-integrated superficial mass
#       flux  ṁ(z) = ∮ ρ u_z dA  is independent of z.
#   (b) Ideal gas at constant p, constant M̄ (M_A = M_B):  u_out/u_in = T_out/T_in.
# ──────────────────────────────────────────────────────────────────────────
function test_thermal_expansion()
    println("=" ^ 60)
    println("TEST C2: thermal expansion (exothermic bed, ρ↓ ⇒ u↑)")
    println("=" ^ 60)

    nr, nz  = 3, 12
    R, L    = 0.02, 0.1
    U_in    = 0.1
    eps_v   = 0.5
    dp_v    = 5e-3
    T0      = 300.0
    μ       = 2.0e-5
    cp      = 1000.0
    λf      = 0.05
    Mbar    = 0.029
    p0      = 1.0e5
    M_A, M_B = 0.029, 0.029
    dH      = -1.0e4              # exothermic [J/mol]
    Da      = 1.5
    A_pre   = Da * U_in / L       # 1.5 s⁻¹
    ρ_in    = p0 * Mbar / (R_GAS * T0)

    rxn = Reaction(Float64(A_pre), 0.0, (-1.0, 1.0), Float64(dH),
                   (M_A, M_B), (true, false))

    g     = Grid2D(nr, nz, R, L)
    pm    = uniform_bed(nr, nz, eps_v, dp_v, 1.0)
    gas   = IdealGas(Mbar, μ, cp, λf, p0)
    # Solid heat capacity kept small ON PURPOSE: the steady state is independent
    # of ρ_s·c_ps (the solid is just a conduit for reaction heat into the gas),
    # but a realistic (large) solid thermal mass would need a very long
    # pseudo-transient to equilibrate.  A light solid reaches the SAME steady
    # state within ~10 residence times.
    solid = SolidProps(50.0, 200.0, 1.0)
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
    t_end  = 10 * L / U_in        # 10 residence times (fully coupled)
    @printf "  ρ_in = %.3f kg/m³,  β = %g,  t_flow = %.3f s,  t_end = %.1f s\n" ρ_in β t_flow t_end

    @time u_ss = solve_steady_compressible_warmstart(u0, params;
                                                     β_flow=100.0, t_flow=t_flow, t_end=t_end)

    uz_ss = field_mat(u_ss, s, F_UZ)
    Tf_ss = field_mat(u_ss, s, F_TF)
    ρ_ss  = density_field(u_ss, params)

    cell_area = [(g.rf[i+1]^2 - g.rf[i]^2) * π for i in 1:nr]

    # (a) plane mass flux ṁ(j) = Σ_i ρ u_z A_i
    mdot = [sum(ρ_ss[:,j] .* uz_ss[:,j] .* cell_area) for j in 1:nz]
    mdot_in = ρ_in * U_in * sum(cell_area)
    mass_dev = (maximum(mdot) - minimum(mdot)) / mean(mdot)
    inout_err = abs(mdot[nz] - mdot_in) / mdot_in

    # (b) ideal-gas velocity/temperature ratio (area-weighted means)
    uz_out = sum(uz_ss[:,nz] .* cell_area) / sum(cell_area)
    Tf_out = sum(Tf_ss[:,nz] .* cell_area) / sum(cell_area)
    u_ratio = uz_out / U_in
    T_ratio = Tf_out / T0
    ratio_err = abs(u_ratio - T_ratio) / T_ratio

    @printf "  Outlet T_f (mean)     : %.1f K   (inlet %.1f K)\n" Tf_out T0
    @printf "  Outlet u_z (mean)     : %.4f m/s (inlet %.4f m/s)\n" uz_out U_in
    @printf "  u_out/u_in            : %.3f\n" u_ratio
    @printf "  T_out/T_in            : %.3f\n" T_ratio
    @printf "  ideal-gas ratio error : %.2e   (u_out/u_in vs T_out/T_in)\n" ratio_err
    @printf "  plane mass-flux spread: %.2e   (max-min)/mean over z planes  ← ∇·(ρu)=0\n" mass_dev
    @printf "  inlet→outlet ṁ error  : %.2e\n" inout_err
    # plane mass-flux spread is the rigorous ∇·(ρu)=0 check; the ideal-gas ratio
    # and inlet→outlet errors are ~3% on this coarse 3×12 grid (radial-averaging
    # and first-order upwind), not a conservation defect.
    pass = mass_dev < 0.01 && ratio_err < 0.05 && inout_err < 0.05 && Tf_out > T0 + 50
    println(pass ? "  PASS" : "  FAIL")
    println()
    return max(mass_dev, ratio_err, inout_err)
end

# ── Entrypoint ─────────────────────────────────────────────────────────────
function main()
    println()
    println("2D Axisymmetric Packed-Bed — COMPRESSIBLE (low-Mach) Verification Suite")
    println()

    c0 = test_reduction()
    c1 = test_compressible_poiseuille()
    c2 = test_thermal_expansion()

    println("─" ^ 60)
    println("Summary")
    println("─" ^ 60)
    @printf "  Test C0 (reduction to incompressible) max rel : %.2e  %s\n" c0 (c0 < 1e-10 ? "PASS" : "FAIL")
    @printf "  Test C1 (isothermal Poiseuille)       L2 err  : %.2e  %s\n" c1 (c1 < 0.05  ? "PASS" : "FAIL")
    @printf "  Test C2 (thermal expansion)           max err : %.2e  %s\n" c2 (c2 < 0.05  ? "PASS" : "FAIL")
    println()
    return (; reduction=c0, isothermal_poiseuille=c1, thermal_expansion=c2)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
