"""
main.jl — Verification tests for the 2D axisymmetric packed-bed solver.

Verification roadmap (plan §8):
  Test 1: Hagen-Poiseuille tube flow     → grid, BCs, viscous fluxes
  Test 2: Ergun ΔP                       → Darcy-Brinkman-Forchheimer closures
  Test 3: First-order A→B reactor        → chemistry, species transport
  Test 4: Two-T heat-up (no flow)        → interphase HT coupling, solid energy eq.
  Test 5: Non-isothermal exothermic PFR  → energy conservation with chemistry
  Test 6: Spatially varying ε(r,z)       → Brinkman masking of solid internals
  Test 7: IFT gradient sanity check      → FD sensitivity vs IFT adjoint
"""

include("src/ad.jl")

using Printf
using LinearAlgebra
using Statistics: mean

# ── Test 1: Hagen-Poiseuille flow ─────────────────────────────────────────
#
# Exact: u_z(r) = 2 U (1 − (r/R)²)
# BC:   inlet uz=U (Dirichlet), wall uz=0 (no-slip), axis symmetry, outlet p=0
# β_ac = U_mean  →  acoustic CFL ≈ convective CFL  →  Tsit5 stable
# t_end = 5 ρ R² / μ  (5× viscous diffusion time)
# ──────────────────────────────────────────────────────────────────────────

function test_poiseuille()
    println("=" ^ 60)
    println("TEST 1: Hagen-Poiseuille flow")
    println("=" ^ 60)

    nr, nz  = 8, 20
    R, L    = 0.01, 0.10   # L > entrance length (≈ 0.05 Re D ≈ 0.04 m here)
    U_mean  = 0.01          # [m/s]
    μ       = 1e-3          # [Pa·s]
    ρ       = 1000.0        # [kg/m³]
    t_end   = 5 * ρ * R^2 / μ   # 5 × viscous diffusion time = 500 s

    g      = Grid2D(nr, nz, Float64(R), Float64(L))
    pm     = PorousMedium(ones(nr,nz), ones(nr,nz), zeros(nr,nz))
    fluid  = FluidProps(ρ, μ, 4182.0, 0.6)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    # β_ac = U_mean so acoustic CFL ≈ convective CFL
    params = ReactorParams(g, pm, fluid, solid, (), U_mean, 0.0, Float64[];
                           uz_in=U_mean, Tf_in=300.0, Ts_in=300.0)

    s  = StateLayout(nr, nz, 0)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_mean)
    set_field!(u0, s, F_TF, 300.0)
    set_field!(u0, s, F_TS, 300.0)

    println("  Integrating (t_end = $t_end s) ...")
    @time sol = solve_steady(u0, params; t_end=t_end)
    println("  Return code: $(sol.retcode)")

    u_ss     = sol.u[end]
    uz_ss    = field_mat(u_ss, s, F_UZ)
    r_vals   = g.r
    uz_sim   = uz_ss[:, nz]          # outlet plane (fully developed)
    uz_exact = @. 2 * U_mean * (1 - (r_vals / R)^2)

    L2_err   = norm(uz_sim .- uz_exact) / norm(uz_exact)
    mass_sim  = sum(uz_ss[:, nz] .* g.r .* g.dr) * 2π
    mass_exact = U_mean * π * R^2
    mass_err  = abs(mass_sim - mass_exact) / mass_exact

    @printf "  L2 error vs parabola: %.2e\n" L2_err
    @printf "  Mass flux error:      %.2e\n" mass_err
    println(L2_err < 0.05 ? "  PASS" : "  FAIL")
    println()
    return L2_err
end

# ── Test 2: Ergun pressure drop ────────────────────────────────────────────
#
# Analytical:  ΔP/L = μ A U + ρ B U²
# β_ac = U_in; t_end must be long enough for the pressure wave to traverse
# the bed: t_acoustic = L / β_ac = 50 s for β_ac = U_in = 0.01 m/s.
# The drag equilibrates much faster (τ_drag ≈ 0.1 s), so acoustic transit
# is the bottleneck.
# ──────────────────────────────────────────────────────────────────────────

function test_ergun_dp()
    println("=" ^ 60)
    println("TEST 2: Ergun pressure drop in uniform packed bed")
    println("=" ^ 60)

    nr, nz  = 4, 20
    R, L    = 0.05, 0.5
    U_in    = 0.01          # [m/s] superficial velocity
    eps_v   = 0.4
    dp_v    = 3e-3
    μ       = 1e-3
    ρ       = 1000.0

    A_erg = ergun_A(eps_v, dp_v)
    B_erg = ergun_B(eps_v, dp_v)
    # Use β_ac = 100 m/s so pressure propagates in L/β = 0.005 s,
    # well before drag relaxation τ_drag = ρ/(ε D_erg) ≈ 0.007 s.
    β_ac  = 100.0
    t_end = 5 * ρ / (eps_v * (μ * A_erg + ρ * B_erg * U_in))

    g      = Grid2D(nr, nz, Float64(R), Float64(L))
    pm     = uniform_bed(nr, nz, eps_v, dp_v)
    fluid  = FluidProps(ρ, μ, 4182.0, 0.6)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    params = ReactorParams(g, pm, fluid, solid, (), β_ac, 0.0, Float64[];
                           uz_in=U_in, Tf_in=300.0, Ts_in=300.0)

    s  = StateLayout(nr, nz, 0)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, 300.0)
    set_field!(u0, s, F_TS, 300.0)

    println("  β_ac = $(β_ac) m/s,  t_end = $(round(t_end, digits=4)) s  (5 × τ_drag)")
    @time sol = solve_steady(u0, params; t_end=t_end)
    println("  Return code: $(sol.retcode)")

    u_ss = sol.u[end]
    pp   = field_mat(u_ss, s, F_P)
    p_in  = mean(pp[:, 1])
    p_out = mean(pp[:, nz])
    dP_sim   = p_in - p_out
    dP_exact = (μ * A_erg * U_in + ρ * B_erg * U_in^2) * L

    rel_err = abs(dP_sim - dP_exact) / dP_exact
    @printf "  Ergun ΔP (exact)   : %.3f Pa\n" dP_exact
    @printf "  Simulated ΔP       : %.3f Pa\n" dP_sim
    @printf "  Relative error     : %.2e\n"    rel_err
    println(rel_err < 0.10 ? "  PASS" : "  FAIL")
    println()
    return rel_err
end

# ── Test 3: First-order plug-flow reactor ──────────────────────────────────
#
# A → B, rate r = A_pre * C_A, Da = A_pre * L / U_in = 1 (moderate conversion)
# Exact: X = 1 − exp(−Da)  (plug flow with Da = k τ)
# ──────────────────────────────────────────────────────────────────────────

function test_plug_flow_reactor()
    println("=" ^ 60)
    println("TEST 3: First-order A→B in plug flow (Da = 1)")
    println("=" ^ 60)
    println("  Uses two-phase operator splitting (plan §2):")
    println("  Phase 1: equilibrate flow  (β=100, t≈0.1 s, Tsit5)")
    println("  Phase 2: transport species (β=U_in, frozen velocity, Tsit5)")

    nr, nz  = 4, 40
    R, L    = 0.05, 1.0
    U_in    = 0.01
    eps_v   = 0.4
    dp_v    = 3e-3
    T0      = 500.0
    μ       = 1e-3
    ρ       = 1000.0   # liquid density; Da = A_pre*L/U is ρ-independent
    M_A     = 0.028
    M_B     = 0.028
    Da      = 1.0
    A_pre   = Da * U_in / L   # 0.01 s⁻¹ → X_exact = 1-exp(-1) ≈ 0.632
    YA0     = 1.0

    rxn = Reaction(Float64(A_pre), 0.0,
                   (-1.0, 1.0),
                   0.0,
                   (M_A, M_B),
                   (true, false))

    g      = Grid2D(nr, nz, Float64(R), Float64(L))
    pm     = uniform_bed(nr, nz, Float64(eps_v), Float64(dp_v), 1.0)
    fluid  = FluidProps(ρ, μ, 1000.0, 0.03)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    Ns     = 2
    # β_ac = U_in: used in phase 2 (species transport, acoustic not limiting)
    params = ReactorParams(g, pm, fluid, solid, (rxn,), U_in, 0.0, fill(1e-5, Ns);
                           uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[YA0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= YA0

    # t_flow: ~50 × drag relaxation; t_species: 5 × residence time.
    # NOTE: τ_drag = ρ/(ε D_erg) is the *local* velocity-to-drag relaxation time,
    # NOT the time for the global pressure field to drive the bed to a
    # divergence-free state. The latter needs ~50 τ_drag here; using only 5 τ_drag
    # freezes a velocity field with uz_out ~5% > uz_in (max|div(ερu)| ~ O(1)),
    # which corrupts the frozen-velocity energy budget in Phase 2.
    D_erg   = μ * ergun_A(eps_v, dp_v) + ρ * ergun_B(eps_v, dp_v) * U_in
    t_flow  = 50 * ρ / (eps_v * D_erg)
    t_spec  = 5 * L / U_in
    println("  A_pre = $A_pre s⁻¹,  t_flow = $(round(t_flow,digits=4)) s,  t_species = $t_spec s")

    @time u_ss = solve_steady_two_phase(u0, params;
                                         β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    YA_out = mean(species_mat(u_ss, s, 1)[:, nz])
    X_sim  = 1 - YA_out / YA0
    X_exact = 1 - exp(-Da)
    rel_err = abs(X_sim - X_exact) / X_exact

    @printf "  Exact conversion    : %.4f\n" X_exact
    @printf "  Simulated conversion: %.4f\n" X_sim
    @printf "  Relative error      : %.2e\n" rel_err
    println(rel_err < 0.05 ? "  PASS" : "  FAIL")
    println()
    return rel_err
end

# ── Test 4: Two-temperature heat-up (no flow) ─────────────────────────────
#
# A pre-heated solid (Ts=700K) equilibrates with a cooler fluid (Tf=400K)
# via interphase heat transfer h_fs · a_v.  No flow, no reactions, no spatial
# gradients in the initial condition → purely ODE dynamics.
#
# Analytical solution (per-cell, uniform field):
#   Cf  = ε ρ c_pf,   Cs = (1−ε) ρ_s c_ps
#   T_eq = (Cf Tf0 + Cs Ts0) / (Cf + Cs)
#   γ   = h_fs · a_v · (1/Cf + 1/Cs)
#   ΔT(t) = (Ts0 − Tf0) exp(−γ t)
#   Tf(t) = T_eq − [Cs/(Cf+Cs)] ΔT(t)
#   Ts(t) = T_eq + [Cf/(Cf+Cs)] ΔT(t)
#
# Boundary contamination (inlet diffusion) is negligible because the
# diffusion time ~ dr²/α >> simulation time 5/γ ~ 46ms.
# ──────────────────────────────────────────────────────────────────────────

function test_two_T_heatup()
    println("=" ^ 60)
    println("TEST 4: Two-T heat-up (no flow, interphase HT coupling)")
    println("=" ^ 60)

    nr, nz  = 4, 8
    R, L    = 0.05, 0.4
    eps_v   = 0.4
    dp_v    = 1e-2     # 1 cm particles
    Tf0     = 400.0    # initial fluid T [K]
    Ts0     = 700.0    # initial solid T [K] (pre-heated)
    ρ       = 1.0      # [kg/m³] (low-density gas for speed)
    μ       = 1e-3
    cpf     = 1000.0
    λf      = 0.6
    ρs      = 2000.0
    cps     = 800.0

    # At u=0: Nu_p = 2 + 1.1*0^0.6*Pr^(1/3) = 2
    h_val   = 2 * λf / dp_v    # 120 W/(m²·K)
    av_val  = 6 * (1 - eps_v) / dp_v   # 360 m²/m³
    Cf      = eps_v * ρ * cpf                  # 400 J/(m³·K)
    Cs      = (1 - eps_v) * ρs * cps          # 960000 J/(m³·K)
    T_eq    = (Cf * Tf0 + Cs * Ts0) / (Cf + Cs)
    γ       = h_val * av_val * (1/Cf + 1/Cs)
    t_end   = 5 / γ    # 5 × equilibration time ≈ 46 ms

    # Inlet BCs match the initial condition → no diffusion flux at t=0
    g      = Grid2D(nr, nz, R, L)
    pm     = uniform_bed(nr, nz, eps_v, dp_v)
    fluid  = FluidProps(ρ, μ, cpf, λf)
    solid  = SolidProps(ρs, cps, 1.0)
    params = ReactorParams(g, pm, fluid, solid, (), 0.01, 0.0, Float64[];
                           uz_in=0.0, Tf_in=Tf0, Ts_in=Ts0)

    s  = StateLayout(nr, nz, 0)
    u0 = zero_state(s)
    set_field!(u0, s, F_TF, Tf0)
    set_field!(u0, s, F_TS, Ts0)

    println("  h_fs = $(h_val) W/(m²·K),  a_v = $(av_val) m⁻¹")
    println("  Cf = $(Cf),  Cs = $(Cs),  T_eq = $(round(T_eq,digits=3)) K")
    @printf "  γ = %.1f s⁻¹,  τ = %.4f s,  t_end = %.4f s\n" γ (1/γ) t_end

    @time sol = solve_steady(u0, params; t_end=t_end)
    println("  Return code: $(sol.retcode)")

    u_ss = sol.u[end]
    Tf_ss = field_mat(u_ss, s, F_TF)
    Ts_ss = field_mat(u_ss, s, F_TS)

    # Check interior cell (avoid inlet boundary contamination)
    ΔT_t  = (Ts0 - Tf0) * exp(-γ * t_end)
    Tf_an = T_eq - (Cs / (Cf + Cs)) * ΔT_t
    Ts_an = T_eq + (Cf / (Cf + Cs)) * ΔT_t

    Tf_interior = mean(Tf_ss[2:end-1, 3:end-1])
    Ts_interior = mean(Ts_ss[2:end-1, 3:end-1])

    err_Tf = abs(Tf_interior - Tf_an) / (Ts0 - Tf0)
    err_Ts = abs(Ts_interior - Ts_an) / (Ts0 - Tf0)

    @printf "  Analytical Tf = %.3f K,  simulated Tf (interior mean) = %.3f K\n" Tf_an Tf_interior
    @printf "  Analytical Ts = %.3f K,  simulated Ts (interior mean) = %.3f K\n" Ts_an Ts_interior
    @printf "  Tf relative error: %.2e,  Ts relative error: %.2e\n" err_Tf err_Ts
    pass = max(err_Tf, err_Ts) < 0.02
    println(pass ? "  PASS" : "  FAIL")
    println()
    return max(err_Tf, err_Ts)
end

# ── Test 5: Non-isothermal exothermic packed-bed reactor ─────────────────
#
# Incremental sub-tests that build on the passing flow (Test 3) and
# heat-transfer (Test 4) tests:
#
#   5a – tiny dH (−50 J/mol, ΔTf ~ 1 K): checks whether the energy-balance
#        error is structural (present even at linear, low-ΔT regime) or only
#        emerges at large temperature excursions.
#
#   5b – full dH (−5000 J/mol): adds a middle diagnostic — the interphase
#        heat-transfer integral IHT = ∫ h_fs a_v (Ts−Tf) dV — which lets us
#        locate which link in the chain  Q_rxn → IHT → ΔH_fluid  breaks.
#
# At steady state the chain must close:
#   solid eq: Q_rxn  ≈ IHT          (all reaction heat leaves solid to fluid)
#   fluid eq: IHT    ≈ ΔH_fluid_out (all interphase heat exits with the fluid)
# ──────────────────────────────────────────────────────────────────────────

# Shared helper: solve the non-isothermal reactor and return diagnostics.
function _run_nonisothermal(dH; nr=4, nz=40)
    R, L    = 0.05, 1.0
    U_in    = 0.01
    eps_v   = 0.4
    dp_v    = 3e-3
    T0      = 400.0
    μ       = 1e-3
    ρ       = 1000.0
    M_A     = 0.028
    M_B     = 0.028
    cpf     = 1000.0
    lam_f   = 0.03
    Da      = 1.0
    A_pre   = Da * U_in / L
    YA0     = 1.0

    rxn = Reaction(Float64(A_pre), 0.0,
                   (-1.0, 1.0),
                   Float64(dH),
                   (M_A, M_B),
                   (true, false))

    g      = Grid2D(nr, nz, Float64(R), Float64(L))
    pm     = uniform_bed(nr, nz, Float64(eps_v), Float64(dp_v), 1.0)
    fluid  = FluidProps(ρ, μ, cpf, lam_f)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    Ns     = 2
    params = ReactorParams(g, pm, fluid, solid, (rxn,), U_in, 0.0, fill(1e-5, Ns);
                           uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[YA0, 0.0])

    s  = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= YA0

    D_erg  = μ * ergun_A(eps_v, dp_v) + ρ * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ρ / (eps_v * D_erg)   # 50 τ_drag: flow must reach div-free (see _run_nonisothermal)
    t_spec = 5 * L / U_in

    @time u_ss = solve_steady_two_phase(u0, params;
                                        β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    YA_ss = species_mat(u_ss, s, 1)
    Tf_ss = field_mat(u_ss, s, F_TF)
    Ts_ss = field_mat(u_ss, s, F_TS)
    uz_ss = field_mat(u_ss, s, F_UZ)
    ur_ss = field_mat(u_ss, s, F_UR)

    X_sim  = 1 - mean(YA_ss[:, nz]) / YA0

    # Three-way energy budget:
    #   Q_rxn    – heat released by reaction (from concentration field)
    #   IHT      – interphase heat transfer ∫ h_fs a_v (Ts−Tf) dV
    #   ΔH_fluid – convective enthalpy flux at outlet (net of inlet)
    Q_rxn   = 0.0
    IHT     = 0.0
    av      = 6 * (1 - eps_v) / dp_v
    for j in 1:nz, i in 1:nr
        r_c = g.r[i]
        dV  = r_c * g.dr * g.dz * 2π
        C_A = ρ * YA_ss[i,j] / M_A
        r_j = A_pre * max(C_A, 0.0)
        Q_rxn += (-dH) * r_j * dV
        h_val  = h_fs(eps_v, dp_v, ρ, μ, cpf, lam_f, ur_ss[i,j], uz_ss[i,j])
        IHT   += h_val * av * (Ts_ss[i,j] - Tf_ss[i,j]) * dV
    end
    ΔH_fluid = ρ * cpf * sum(uz_ss[:, nz] .* (Tf_ss[:, nz] .- T0) .* g.r .* g.dr) * 2π

    return (X=X_sim,
            Tf_out=mean(Tf_ss[:, nz]),
            Ts_out=mean(Ts_ss[:, nz]),
            Q_rxn=Q_rxn, IHT=IHT, ΔH_fluid=ΔH_fluid)
end

# ── Test 5a ────────────────────────────────────────────────────────────────
function test_nonisothermal_5a()
    println("=" ^ 60)
    println("TEST 5a: Non-isothermal reactor — tiny dH (linear regime)")
    println("=" ^ 60)
    dH = -50.0   # ΔTf ≈ 1.1 K at X≈0.63 — stays well within linear regime
    println("  dH = $dH J/mol  (ΔTf_expected ≈ $(round((-dH)*0.63*1.0/(0.028*1000.0), digits=2)) K)")
    d = _run_nonisothermal(dH)
    ref = max(d.Q_rxn, 1.0)
    err_solid  = abs(d.Q_rxn  - d.IHT)     / ref
    err_fluid  = abs(d.IHT    - d.ΔH_fluid) / ref
    err_total  = abs(d.Q_rxn  - d.ΔH_fluid) / ref
    @printf "  Conversion X          : %.4f\n"  d.X
    @printf "  Outlet Tf             : %.4f K\n" d.Tf_out
    @printf "  Outlet Ts             : %.4f K\n" d.Ts_out
    @printf "  Q_rxn   (solid src)   : %.4f W\n" d.Q_rxn
    @printf "  IHT     (interphase)  : %.4f W\n" d.IHT
    @printf "  ΔH_fluid(outlet flux) : %.4f W\n" d.ΔH_fluid
    @printf "  solid link err        : %.2e\n"   err_solid
    @printf "  fluid link err        : %.2e\n"   err_fluid
    @printf "  end-to-end err        : %.2e\n"   err_total
    pass = err_total < 0.15
    println(pass ? "  PASS" : "  FAIL")
    println()
    return err_total
end

# ── Test 5b ────────────────────────────────────────────────────────────────
function test_nonisothermal_5b()
    println("=" ^ 60)
    println("TEST 5b: Non-isothermal reactor — full dH, three-way budget")
    println("=" ^ 60)
    dH = -5000.0
    println("  dH = $dH J/mol,  Da = 1.0")
    println("  Using two-phase operator splitting (same as Test 3)")
    d = _run_nonisothermal(dH)
    ref = max(d.Q_rxn, 1.0)
    err_solid  = abs(d.Q_rxn  - d.IHT)     / ref
    err_fluid  = abs(d.IHT    - d.ΔH_fluid) / ref
    err_total  = abs(d.Q_rxn  - d.ΔH_fluid) / ref
    @printf "  Conversion X          : %.4f\n"  d.X
    @printf "  Outlet Tf             : %.2f K\n" d.Tf_out
    @printf "  Outlet Ts             : %.2f K\n" d.Ts_out
    @printf "  Q_rxn   (solid src)   : %.1f W\n" d.Q_rxn
    @printf "  IHT     (interphase)  : %.1f W\n" d.IHT
    @printf "  ΔH_fluid(outlet flux) : %.1f W\n" d.ΔH_fluid
    @printf "  solid link err (Q→IHT): %.2e\n"   err_solid
    @printf "  fluid link err (IHT→ΔH):%.2e\n"   err_fluid
    @printf "  end-to-end err        : %.2e\n"   err_total
    pass = err_total < 0.15
    println(pass ? "  PASS" : "  FAIL")
    println()
    return err_total
end

# ── Test 6: Spatially varying ε(r,z) — Brinkman masking of an internal ───
#
# A solid annular internal blocks the outer part of the bed: ε = ε_block ≪ ε_bulk
# for r > r_block.  The Ergun drag in the blocked cells (∝ 1/ε³) explodes,
# driving velocity to ≈0 there — Brinkman volume penalisation.  The flow is
# forced through the open core, which speeds up to satisfy continuity.
#
# Continuity is on the SUPERFICIAL velocity: the steady balance is ∇·(ρu)=0
# (ε appears only in the unsteady storage term, see plan §1) ⇒ for constant ρ,
# ∇·u = 0 and the volumetric flux Σ u_z·A is conserved.  With a uniform inlet
# u_z = U_in over the whole face, all of it funnels into the open core:
#
#   Σ u_z·A |outlet = U_in · A_total   ⇒   ⟨u_z⟩_core ≈ U_in · A_total/A_open
#                                                     = U_in / f_open
#
# Checks: (1) blocked-zone velocity ≈ 0,  (2) volumetric flux conserved,
#         (3) open-core velocity matches the area-ratio (funnelling) prediction.
#
# Note on ε_block: the plan's ε_min ≈ 1e-3 floor gives near-perfect masking
# (blocked u_z ~ 1e-10·U_in) but the 1/ε³ drag is far too stiff for the explicit
# pseudo-transient — it needs the implicit `solve_steady_nk` path (plan §4).
# Here ε_block = 0.05 already suppresses the blocked velocity to ~0.5%·U_in and
# converges cleanly with the proven Tsit5 pseudo-transient.
# ──────────────────────────────────────────────────────────────────────────

function test_porosity_masking()
    println("=" ^ 60)
    println("TEST 6: Spatially varying ε(r,z) — Brinkman masking")
    println("=" ^ 60)

    nr, nz   = 8, 20
    R, L     = 0.05, 0.5
    U_in     = 0.01         # [m/s] uniform inlet superficial velocity
    eps_v    = 0.4          # bulk bed void fraction
    eps_block = 0.05        # blocked annulus (see note above; ε_min=1e-3 needs NK)
    dp_v     = 3e-3
    μ        = 1e-3
    ρ        = 1000.0
    r_block  = 0.6 * R      # block the outer annulus r > r_block

    g  = Grid2D(nr, nz, Float64(R), Float64(L))

    # ε field: bulk in the open core, ε_block in the blocked annulus.
    eps_field = fill(eps_v, nr, nz)
    dp_field  = fill(dp_v, nr, nz)
    for j in 1:nz, i in 1:nr
        if g.r[i] > r_block
            eps_field[i,j] = eps_block
        end
    end
    open_mask = g.r .<= r_block          # length-nr Bool, per radial cell
    blk_mask  = .!open_mask
    pm = PorousMedium(eps_field, dp_field, ones(nr, nz))

    # Exact cross-sectional cell areas from the faces (for flux integrals).
    cell_area = [(g.rf[i+1]^2 - g.rf[i]^2) * π for i in 1:nr]
    A_total   = sum(cell_area)
    f_open    = sum(cell_area[open_mask]) / A_total
    uz_pred   = U_in / f_open            # funnelling prediction for the open core

    fluid  = FluidProps(ρ, μ, 4182.0, 0.6)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    β_ac   = 100.0          # pressure propagates fast (see Test 2)
    A_erg  = ergun_A(eps_v, dp_v)
    B_erg  = ergun_B(eps_v, dp_v)
    t_end  = 50 * ρ / (eps_v * (μ * A_erg + ρ * B_erg * U_in))  # 50 τ_drag (lateral redistribution)
    params = ReactorParams(g, pm, fluid, solid, (), β_ac, 0.0, Float64[];
                           uz_in=U_in, Tf_in=300.0, Ts_in=300.0)

    s  = StateLayout(nr, nz, 0)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, 300.0)
    set_field!(u0, s, F_TS, 300.0)

    @printf "  r_block = %.3f m,  open area fraction f_open = %.3f,  ε_block = %.2g\n" r_block f_open eps_block
    println("  β_ac = $(β_ac) m/s,  t_end = $(round(t_end, digits=4)) s  (50 × τ_drag)")
    @time sol = solve_steady(u0, params; t_end=t_end)
    println("  Return code: $(sol.retcode)")

    u_ss   = sol.u[end]
    uz_ss  = field_mat(u_ss, s, F_UZ)
    uz_out = uz_ss[:, nz]                 # outlet plane (fully developed)

    # Volumetric mass conservation: Σ u_z·A is the conserved flux (∇·u = 0).
    flux_in  = U_in * A_total             # uniform inlet over the whole face
    flux_out = sum(uz_out .* cell_area)
    mass_err = abs(flux_out - flux_in) / flux_in

    # Penalisation: area-weighted mean |u_z| in the blocked annulus vs U_in.
    uz_blocked    = sum(abs.(uz_out[blk_mask]) .* cell_area[blk_mask]) / sum(cell_area[blk_mask])
    blocked_ratio = uz_blocked / U_in

    # Funnelling: area-weighted mean u_z in the open core vs U_in/f_open.
    uz_core    = sum(uz_out[open_mask] .* cell_area[open_mask]) / sum(cell_area[open_mask])
    funnel_err = abs(uz_core - uz_pred) / uz_pred

    @printf "  Open-core ⟨u_z⟩      : %.4e m/s  (predicted U_in/f_open = %.4e)\n" uz_core uz_pred
    @printf "  Blocked-zone ⟨u_z⟩   : %.4e m/s  (=%.2e · U_in)\n"               uz_blocked blocked_ratio
    @printf "  Volumetric-flux error: %.2e\n" mass_err
    @printf "  Funnelling error     : %.2e\n" funnel_err
    pass = blocked_ratio < 0.05 && mass_err < 0.02 && funnel_err < 0.10
    println(pass ? "  PASS" : "  FAIL")
    println()
    return max(blocked_ratio, mass_err, funnel_err)
end

# ── Test 7: Gradient sanity check (FD vs IFT adjoint) ────────────────────
#
# Perturb the pre-exponential A_pre and compare:
#   FD gradient: (L(u*(θ+h)) − L(u*(θ))) / h   (full re-solve)
#   IFT gradient: dL/dθ = −λᵀ ∂F/∂θ            (one adjoint solve)
#
# Loss: mean outlet conversion.  Parameter: A_pre (scalar).
# Uses a small grid (nr=4, nz=10) so the dense FD Jacobian in ad.jl is fast.
# ──────────────────────────────────────────────────────────────────────────

function test_gradient_sanity()
    println("=" ^ 60)
    println("TEST 7: IFT gradient vs finite-difference (A_pre)")
    println("=" ^ 60)

    nr, nz  = 4, 10
    R, L    = 0.05, 0.5
    U_in    = 0.01
    eps_v   = 0.4
    dp_v    = 3e-3
    T0      = 500.0
    μ       = 1e-3
    ρ       = 1000.0   # liquid density — same as Tests 2&3
    M_A     = 0.028
    M_B     = 0.028
    Da      = 0.5
    A_pre   = Da * U_in / L   # 0.01 s⁻¹
    YA0     = 1.0

    rxn = Reaction(Float64(A_pre), 0.0,
                   (-1.0, 1.0),
                   0.0,
                   (M_A, M_B),
                   (true, false))

    g      = Grid2D(nr, nz, Float64(R), Float64(L))
    pm     = uniform_bed(nr, nz, Float64(eps_v), Float64(dp_v), 1.0)
    fluid  = FluidProps(ρ, μ, 1000.0, 0.03)
    solid  = SolidProps(2000.0, 800.0, 1.0)
    Ns     = 2
    params = ReactorParams(g, pm, fluid, solid, (rxn,), U_in, 0.0, fill(1e-5, Ns);
                           uz_in=U_in, Tf_in=T0, Ts_in=T0, Y_in=[YA0, 0.0])

    s = StateLayout(nr, nz, Ns)
    u0 = zero_state(s)
    set_field!(u0, s, F_UZ, U_in)
    set_field!(u0, s, F_TF, T0)
    set_field!(u0, s, F_TS, T0)
    species_mat(u0, s, 1) .= YA0

    D_erg  = μ * ergun_A(eps_v, dp_v) + ρ * ergun_B(eps_v, dp_v) * U_in
    t_flow = 50 * ρ / (eps_v * D_erg)   # 50 τ_drag: flow must reach div-free (see _run_nonisothermal)
    t_spec = 3 * L / U_in

    println("  Solving base case (two-phase operator split) ...")
    @time u_ss = solve_steady_two_phase(u0, params;
                                        β_flow=100.0, t_flow=t_flow, t_species=t_spec)

    # Loss: mean outlet YB (product B mass fraction at outlet)
    function loss_YB(uvec)
        mean(field_mat(uvec, s, 5+2)[:, nz])   # species 2 = B
    end

    # θ = [A_pre];  getter/setter for the single Reaction's A_pre
    θ_getter(p) = [p.reactions[1].A_pre]
    function θ_setter(p, θ_new)
        new_rxn = Reaction(θ_new[1], rxn.E_a, rxn.nu, rxn.dH, rxn.M, rxn.reactant_idx)
        ReactorParams(p.grid, p.porous, p.fluid, p.solid, (new_rxn,),
                      p.beta_ac, p.gravity, p.D_species, p.bcs)
    end

    println("  Computing IFT gradient ...")
    @time g_ift = ift_gradient(loss_YB, u_ss, params, θ_getter, θ_setter)

    println("  Computing FD gradient (two full re-solves) ...")
    h_fd = 1e-4 * A_pre
    θ0   = θ_getter(params)
    θ_p  = copy(θ0); θ_p[1] += h_fd
    u_p  = solve_steady_two_phase(u0, θ_setter(params, θ_p);
                                  β_flow=100.0, t_flow=t_flow, t_species=t_spec)
    g_fd = [(loss_YB(u_p) - loss_YB(u_ss)) / h_fd]

    rel_err = abs(g_ift[1] - g_fd[1]) / max(abs(g_fd[1]), 1e-12)
    @printf "  IFT gradient  (dYB/dA_pre) : %.4e\n" g_ift[1]
    @printf "  FD  gradient  (dYB/dA_pre) : %.4e\n" g_fd[1]
    @printf "  Relative error             : %.2e\n" rel_err
    pass = rel_err < 0.05
    println(pass ? "  PASS" : "  FAIL")
    println()
    return rel_err
end

# ── Entrypoint ─────────────────────────────────────────────────────────────

function main()
    println()
    println("2D Axisymmetric Packed-Bed Reactor Solver — Verification Suite")
    println()

    e1 = test_poiseuille()
    e2 = test_ergun_dp()
    e3 = test_plug_flow_reactor()
    e4 = test_two_T_heatup()
    e5a = test_nonisothermal_5a()
    e5b = test_nonisothermal_5b()
    e6 = test_porosity_masking()
    e7 = test_gradient_sanity()

    println("─" ^ 60)
    println("Summary")
    println("─" ^ 60)
    @printf "  Test 1  (Poiseuille)       L2 error      : %.2e  %s\n" e1  (e1  < 0.05 ? "PASS" : "FAIL")
    @printf "  Test 2  (Ergun ΔP)         rel error     : %.2e  %s\n" e2  (e2  < 0.10 ? "PASS" : "FAIL")
    @printf "  Test 3  (PFR conv.)        rel error     : %.2e  %s\n" e3  (e3  < 0.05 ? "PASS" : "FAIL")
    @printf "  Test 4  (Two-T heatup)     max T error   : %.2e  %s\n" e4  (e4  < 0.02 ? "PASS" : "FAIL")
    @printf "  Test 5a (Non-iso tiny dH)  energy error  : %.2e  %s\n" e5a (e5a < 0.15 ? "PASS" : "FAIL")
    @printf "  Test 5b (Non-iso full dH)  energy error  : %.2e  %s\n" e5b (e5b < 0.15 ? "PASS" : "FAIL")
    @printf "  Test 6  (ε-mask Brinkman)  max error      : %.2e  %s\n" e6  (e6  < 0.10 ? "PASS" : "FAIL")
    @printf "  Test 7  (IFT gradient)     rel error     : %.2e  %s\n" e7  (e7  < 0.05 ? "PASS" : "FAIL")
    println()
    return (; poiseuille=e1, ergun_dp=e2, plug_flow=e3, two_T_heatup=e4,
             nonisothermal_5a=e5a, nonisothermal_5b=e5b,
             porosity_masking=e6, gradient_sanity=e7)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
