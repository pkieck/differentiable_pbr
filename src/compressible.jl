"""
Compressible (low-Mach, variable-density) version of the packed-bed solver.

Difference from the incompressible solver (`rhs!` in rhs.jl)
-----------------------------------------------------------
The incompressible solver carries a constant `fluid.rho` and uses artificial
compressibility only as a pressure-velocity coupling trick: at steady state
it drives div(u) = 0 (volumetric, see Test 6).

Here the density is a genuine field set by the ideal-gas equation of state at a
fixed thermodynamic pressure p0 (low-Mach / "isobaric" approximation, standard
for reactive packed-bed flows):

      rho(r,z) = p0 * Mbar / (R * T_f),     Mbar = 1 / sum_k (Y_k / M_k)

so the gas expands as it heats or its mean molar mass drops. The dynamic
pressure p' retains its sole job - enforcing mass conservation - but the
constraint is now on the mass flux:

      dp'/dt = -beta^2 * eps * div(rho u)    =>  steady state:  div(rho u) = 0

(not div(u) = 0). Consequently the superficial mass flux rho*u is divergence-free,
and where the gas heats up the velocity must rise to keep rho*u*A constant.

Scalar (energy, species) advection uses the mass flux rho*u at the faces, with
the same uniform-field-annihilating correction the incompressible solver uses:

      div(rho u q) - q_c * div(rho u)   ->   rho u * grad(q)     (non-conservative form)

This annihilates a uniform field even when div(rho*u) != 0 (under-converged /
pseudo-transient state), exactly as the constant-rho solver subtracts q_c*div(u).

When T and composition are uniform, rho is constant and every term reduces
identically to the validated incompressible `rhs!` (verified by Test C0).
"""

include("solve.jl")   # brings in rhs.jl (helpers + structs), DifferentialEquations, NonlinearSolve
include("kinetics.jl")  # GlobalReaction / Mechanism + unified rxn_rate/rxn_negdH/rxn_mass_coeff

using LinearAlgebra: dot

# -- Ideal-gas fluid properties -------------------------------------------------

"""
Ideal-gas fluid descriptor for low-Mach compressible solver.

Mbar: default mean molar mass [kg/mol] (when Ns=0);
with species, Mbar = 1 / sum(Y_k/M_k) per cell.

p0: constant thermodynamic pressure.
The dynamic pressure solved in state vector is a gauge pressure that drives the
mass-flux divergence to zero and does not feed back into the EOS.
"""
struct IdealGas{T<:AbstractFloat}
    Mbar::T      # default mean molar mass [kg/mol]
    mu::T        # dynamic viscosity      [Pa*s]
    cp::T        # heat capacity          [J/(kg*K)]
    lambda::T    # thermal conductivity   [W/(m*K)]
    p0::T        # thermodynamic pressure [Pa]
end

"""
Top-level parameter bundle for the compressible solver.  Mirrors
`ReactorParams` but carries an `IdealGas` instead of a constant-density
`FluidProps`, plus `M_species` (molar masses for the EOS mean molar mass).
"""
struct CompressibleParams{T, PM, G<:IdealGas, S<:SolidProps, RC, BC<:BoundaryConditions}
    grid::Grid2D{T}
    porous::PM
    gas::G
    solid::S
    reactions::RC
    beta_ac::T
    gravity::T
    D_species::Vector{T}
    M_species::Vector{T}   # molar masses [kg/mol] for EOS Mbar (length Ns)
    bcs::BC
end

function CompressibleParams(grid, porous, gas::IdealGas{T}, solid, reactions,
                            beta_ac, gravity, D_species, M_species;
                            uz_in=zero(T), Tf_in=T(300), Ts_in=T(300),
                            Y_in=T[]) where T
    bcs = BoundaryConditions(T(uz_in), T(Tf_in), T(Ts_in), convert(Vector{T}, Y_in))
    CompressibleParams(grid, porous, gas, solid, reactions, T(beta_ac), T(gravity),
                       convert(Vector{T}, D_species), convert(Vector{T}, M_species), bcs)
end

function Base.copy(p::CompressibleParams)
    CompressibleParams(p.grid, p.porous, p.gas, p.solid, p.reactions,
                       p.beta_ac, p.gravity, copy(p.D_species), copy(p.M_species), p.bcs)
end

# ── Equation of state ───────────────────────────────────────────────────────


"""    ideal_gas_density(p0, Mbar, T) -> rho  [kg/m^3]

rho = p0 * Mbar / (R * T).  `R_GAS` is defined in closures.jl.
"""
@inline ideal_gas_density(p0, Mbar, Tk) = p0 * Mbar / (R_GAS * Tk)

# -- Main compressible RHS --------------------------------------------------

"""
    rhs_compressible!(du, u, p::CompressibleParams, t)

In-place RHS for the low-Mach variable-density packed-bed reactor.  State
layout is identical to the incompressible solver:
  [u_r | u_z | p' | T_f | T_s | Y_1 .. Y_Ns]
"""
function rhs_compressible!(du, u, p::CompressibleParams, t)
    # Function barrier: lift the species count to a compile-time `Val` so every
    # per-cell `ntuple(..., Val(Ns))` below is statically sized. This is the one
    # dynamic dispatch per RHS call; everything inside the kernel is type-stable
    # and allocation-free.
    _rhs_compressible_kernel!(du, u, p, t, Val(length(p.D_species)))
end

function _rhs_compressible_kernel!(du, u, p::CompressibleParams, t, ::Val{Ns}) where {Ns}
    g    = p.grid
    s    = StateLayout(g.nr, g.nz, Ns)
    pm   = p.porous
    gas  = p.gas
    sl   = p.solid
    β    = p.beta_ac
    grav = p.gravity

    nr, nz = g.nr, g.nz
    dr, dz = g.dr, g.dz

    ur = field_mat(u,  s, F_UR)
    uz = field_mat(u,  s, F_UZ)
    pp = field_mat(u,  s, F_P)
    Tf = field_mat(u,  s, F_TF)
    Ts = field_mat(u,  s, F_TS)

    dur = field_mat(du, s, F_UR)
    duz = field_mat(du, s, F_UZ)
    dpp = field_mat(du, s, F_P)
    dTf = field_mat(du, s, F_TF)
    dTs = field_mat(du, s, F_TS)

    Y_views  = ntuple(k -> field_mat(u,  s, 5+k), Val(Ns))
    dY_views = ntuple(k -> field_mat(du, s, 5+k), Val(Ns))

    fill!(du, zero(eltype(du)))

    Tdt  = eltype(u)
    mu   = gas.mu
    cpf  = gas.cp
    lamf = gas.lambda
    p0   = gas.p0
    Mbar0 = gas.Mbar
    Msp  = p.M_species
    rhos = sl.rho
    cps  = sl.cp
    lams = sl.lambda

    # -- Density field from the EOS (precomputed for face interpolation) -----
    rho_mat = Matrix{Tdt}(undef, nr, nz)
    @inbounds for j in 1:nz, i in 1:nr
        invM = zero(Tdt)
        for k in 1:Ns
            invM += Y_views[k][i,j] / Msp[k]
        end
        invM = (Ns == 0 || !(invM > 0)) ? one(Tdt) / Mbar0 : invM
        rho_mat[i,j] = p0 / (R_GAS * Tf[i,j] * invM)
    end

    # Inlet ghost density from the Dirichlet inlet state (T_f,in, Y_in)
    Tf_bc = p.bcs.Tf_in
    Ts_bc = p.bcs.Ts_in
    uz_bc = p.bcs.uz_in
    invM_in = zero(Tdt)
    for k in 1:min(Ns, length(p.bcs.Y_in))
        invM_in += p.bcs.Y_in[k] / Msp[k]
    end
    invM_in = (Ns == 0 || !(invM_in > 0)) ? one(Tdt) / Mbar0 : invM_in
    rho_in  = p0 / (R_GAS * Tf_bc * invM_in)

    for j in 1:nz, i in 1:nr
        eps  = pm.eps[i,j]
        dprt = pm.dp[i,j]
        phi  = pm.phi_cat[i,j]
        av   = pm.av[i,j]
        r_c  = g.r[i]

        ur_c = ur[i,j]
        uz_c = uz[i,j]
        Tf_c = Tf[i,j]
        Ts_c = Ts[i,j]
        pp_c = pp[i,j]
        rho_c = rho_mat[i,j]

        # -- Ghost-cell values for BCs (identical convention to rhs!) ---------
        ur_im1 = (i == 1)  ? -ur_c   : ur[i-1,j]
        ur_ip1 = (i == nr) ? -ur_c   : ur[i+1,j]
        uz_im1 = (i == 1)  ?  uz_c   : uz[i-1,j]
        uz_ip1 = (i == nr) ? -uz_c   : uz[i+1,j]
        pp_im1 = (i == 1)  ?  pp_c   : pp[i-1,j]
        pp_ip1 = (i == nr) ?  pp_c   : pp[i+1,j]
        Tf_im1 = (i == 1)  ?  Tf_c   : Tf[i-1,j]
        Tf_ip1 = (i == nr) ?  Tf_c   : Tf[i+1,j]
        Ts_im1 = (i == 1)  ?  Ts_c   : Ts[i-1,j]
        Ts_ip1 = (i == nr) ?  Ts_c   : Ts[i+1,j]

        # density ghosts mirror the temperature/composition ghosts:
        # Neumann at axis & walls, Dirichlet (= rho_in) at the inlet.
        rho_im1 = (i == 1)  ? rho_c  : rho_mat[i-1,j]
        rho_ip1 = (i == nr) ? rho_c  : rho_mat[i+1,j]

        uz_jm1 = (j == 1)  ? 2*uz_bc - uz_c   : uz[i,j-1]
        uz_jp1 = (j == nz) ? uz_c             : uz[i,j+1]
        ur_jm1 = (j == 1)  ? -ur_c            : ur[i,j-1]
        ur_jp1 = (j == nz) ? ur_c             : ur[i,j+1]

        pp_jm1 = (j == 1)  ? pp_c             : pp[i,j-1]
        pp_jp1 = (j == nz) ? -pp_c            : pp[i,j+1]

        Tf_jm1_diff = (j == 1)  ? 2*Tf_bc - Tf_c   : Tf[i,j-1]
        Tf_jm1_adv  = (j == 1)  ? Tf_bc            : Tf[i,j-1]
        Tf_jp1      = (j == nz) ? Tf_c             : Tf[i,j+1]

        Ts_jm1_diff = (j == 1)  ? 2*Ts_bc - Ts_c   : Ts[i,j-1]
        Ts_jp1      = (j == nz) ? Ts_c             : Ts[i,j+1]

        rho_jm1 = (j == 1)  ? rho_in : rho_mat[i,j-1]
        rho_jp1 = (j == nz) ? rho_c  : rho_mat[i,j+1]

        # -- Face velocities --------------------------------------------------
        ur_r_face_m = face_vel(ur_im1, ur_c)
        ur_r_face_p = face_vel(ur_c, ur_ip1)
        uz_z_face_m = face_vel(uz_jm1, uz_c)
        uz_z_face_p = face_vel(uz_c, uz_jp1)

        rf_m = g.rf[i]
        rf_p = g.rf[i+1]
        Ar   = r_c * dr

        # -- Face densities (arithmetic average across the face) --------
        rho_r_m = 0.5 * (rho_im1 + rho_c)
        rho_r_p = 0.5 * (rho_c + rho_ip1)
        rho_z_m = 0.5 * (rho_jm1 + rho_c)
        rho_z_p = 0.5 * (rho_c + rho_jp1)

        # -- Mass flux rho*u at each face [kg/(m^2*s)] -----
        G_r_m = rho_r_m * ur_r_face_m
        G_r_p = rho_r_p * ur_r_face_p
        G_z_m = rho_z_m * uz_z_face_m
        G_z_p = rho_z_p * uz_z_face_p

        # -- Continuity (artificial compressibility on the MASS flux) -------
        # div(rho*u) in axisymmetric FVM:
        div_mass = ( rf_p * G_r_p - rf_m * G_r_m ) / Ar +
                   ( G_z_p - G_z_m ) / dz

        # eps kept as a per-cell scalar (parity with incompressible solver and
        # Brinkman masking); steady state => div(rho*u) = 0 regardless of eps.
        dpp[i,j] = -β^2 * eps * div_mass

        # -- Drag & effective viscosity (local density) ----------------------
        D_drag = drag_coeff(eps, dprt, rho_c, mu, ur_c, uz_c)
        mu_eff_loc  = mu_eff(eps, mu)

        # -- r-Momentum ------------------------------------------------------
        dp_dr  = (pp_ip1 - pp_im1) / (2*dr)
        visc_r_r = ( rf_p * (ur_ip1 - ur_c) / dr
                   - rf_m * (ur_c - ur_im1) / dr ) / Ar
        visc_r_z = (ur_jp1 - 2*ur_c + ur_jm1) / dz^2
        cent_r   = ur_c / (r_c^2 + 1e-20)
        visc_r   = visc_r_r + visc_r_z - cent_r

        dur[i,j] = eps / rho_c * (-dp_dr + mu_eff_loc * visc_r - D_drag * ur_c)

        # -- z-Momentum ------------------------------------------------------
        dp_dz  = (pp_jp1 - pp_jm1) / (2*dz)
        visc_z_r = ( rf_p * (uz_ip1 - uz_c) / dr
                   - rf_m * (uz_c - uz_im1) / dr ) / Ar
        visc_z_z = (uz_jp1 - 2*uz_c + uz_jm1) / dz^2
        visc_z   = visc_z_r + visc_z_z

        duz[i,j] = eps / rho_c * (-dp_dz + mu_eff_loc * visc_z - D_drag * uz_c - rho_c * grav)

        # -- Fluid energy -------
        # Advection uses the mass flux ρu at faces; subtract T_c·div(ρu) to
        # recover the non-conservative form ρu·∇T (annihilates uniform T).
        lamf_eff_loc = lam_f_eff(eps, lamf)
        h      = h_fs(eps, dprt, rho_c, mu, cpf, lamf, ur_c, uz_c)

        adv_Tf = ( rf_p * adv_flux(G_r_p, Tf_c, Tf_ip1)
                 - rf_m * adv_flux(G_r_m, Tf_im1, Tf_c) ) / Ar +
                 ( adv_flux(G_z_p, Tf_c, Tf_jp1)
                 - adv_flux(G_z_m, Tf_jm1_adv, Tf_c) ) / dz
        adv_Tf -= Tf_c * div_mass

        diff_Tf = ( rf_p * diff_flux(lamf_eff_loc, Tf_c, Tf_ip1, dr)
                  - rf_m * diff_flux(lamf_eff_loc, Tf_im1, Tf_c, dr) ) / Ar +
                  ( diff_flux(lamf_eff_loc, Tf_c, Tf_jp1, dz)
                  - diff_flux(lamf_eff_loc, Tf_jm1_diff, Tf_c, dz) ) / dz

        interphase_Tf = h * av * (Ts_c - Tf_c)

        dTf[i,j] = ( -cpf * adv_Tf + diff_Tf + interphase_Tf ) /
                   ( eps * rho_c * cpf )

        # -- Solid energy ---
        lams_eff_loc = lam_s_eff(eps, lams)
        diff_Ts = ( rf_p * diff_flux(lams_eff_loc, Ts_c, Ts_ip1, dr)
                  - rf_m * diff_flux(lams_eff_loc, Ts_im1, Ts_c, dr) ) / Ar +
                  ( diff_flux(lams_eff_loc, Ts_c, Ts_jp1, dz)
                  - diff_flux(lams_eff_loc, Ts_jm1_diff, Ts_c, dz) ) / dz

        # Cell-local mass fractions, shared by all reaction terms. Going through
        # the rxn_rate / rxn_negdH / rxn_mass_coeff accessors (kinetics.jl) is the
        # whole coupling surface to the kinetics backend — legacy `Reaction` and
        # parsed `GlobalReaction` (CHEMKIN) both implement it.
        Yvals = ntuple(l -> Y_views[l][i,j], Val(Ns))

        Q_rxn = zero(Tdt)
        for rxn in p.reactions
            r_j = rxn_rate(rxn, Ts_c, rho_c, Yvals)
            Q_rxn += rxn_negdH(rxn) * r_j
        end

        solid_cap = max((1 - eps) * rhos * cps, 1e-20)
        dTs[i,j] = ( diff_Ts - h * av * (Ts_c - Tf_c) + phi * Q_rxn ) / solid_cap

        # -- Species transport
        for k in 1:Ns
            Y  = Y_views[k]
            dY = dY_views[k]
            Dk_eff = D_eff(eps, p.D_species[k])

            Yk_c   = Y[i,j]
            Yk_im1 = (i == 1)  ? Yk_c  : Y[i-1,j]
            Yk_ip1 = (i == nr) ? Yk_c  : Y[i+1,j]
            Yk_bc       = (k ≤ length(p.bcs.Y_in)) ? p.bcs.Y_in[k] : Yk_c
            Yk_jm1_adv  = (j == 1)  ? Yk_bc            : Y[i,j-1]
            Yk_jm1_diff = (j == 1)  ? 2*Yk_bc - Yk_c   : Y[i,j-1]
            Yk_jp1      = (j == nz) ? Yk_c             : Y[i,j+1]

            adv_Yk = ( rf_p * adv_flux(G_r_p, Yk_c, Yk_ip1)
                     - rf_m * adv_flux(G_r_m, Yk_im1, Yk_c) ) / Ar +
                     ( adv_flux(G_z_p, Yk_c, Yk_jp1)
                     - adv_flux(G_z_m, Yk_jm1_adv, Yk_c) ) / dz
            adv_Yk -= Yk_c * div_mass

            # Diffusive flux uses the face density: div(rho D grad Y)
            diff_Yk = ( rf_p * diff_flux(rho_r_p * Dk_eff, Yk_c, Yk_ip1, dr)
                      - rf_m * diff_flux(rho_r_m * Dk_eff, Yk_im1, Yk_c, dr) ) / Ar +
                      ( diff_flux(rho_z_p * Dk_eff, Yk_c, Yk_jp1, dz)
                      - diff_flux(rho_z_m * Dk_eff, Yk_jm1_diff, Yk_c, dz) ) / dz

            Sk = zero(Tdt)
            for rxn in p.reactions
                r_j = rxn_rate(rxn, Ts_c, rho_c, Yvals)
                Sk += phi * rxn_mass_coeff(rxn, k) * r_j
            end

            dY[i,j] = (-adv_Yk + diff_Yk + Sk) / (eps * rho_c)
        end

    end  # cell loop

    nothing
end

# ── Density-field utility (post-processing / diagnostics) ───────────────────

"""
    density_field(u, p::CompressibleParams) → nr×nz matrix

Evaluate the EOS density ρ(r,z) for a given state, for diagnostics and
mass-flux conservation checks.
"""
function density_field(u::AbstractVector, p::CompressibleParams)
    g  = p.grid
    Ns = length(p.D_species)
    s  = StateLayout(g.nr, g.nz, Ns)
    Tf = field_mat(u, s, F_TF)
    Y_views = ntuple(k -> field_mat(u, s, 5+k), Ns)
    T  = eltype(u)
    ρ  = Matrix{T}(undef, g.nr, g.nz)
    for j in 1:g.nz, i in 1:g.nr
        invM = zero(T)
        for k in 1:Ns
            invM += Y_views[k][i,j] / p.M_species[k]
        end
        invM = (Ns == 0 || !(invM > 0)) ? one(T) / p.gas.Mbar : invM
        ρ[i,j] = p.gas.p0 / (R_GAS * Tf[i,j] * invM)
    end
    ρ
end

# ── Solvers ─────────────────────────────────────────────────────────────────

"""
    solve_steady_compressible(u0, params; t_end, solver, kwargs...)

Fully-coupled pseudo-transient solve to steady state with `rhs_compressible!`.
Unlike the incompressible two-phase split, the velocity cannot be frozen while
the temperature evolves (the gas expands as it heats, so u responds to ρ), so
momentum/pressure/energy/species are integrated together.

Choose `t_end` ≥ the longest relaxation time and set `params.beta_ac` so the
acoustic transit L/β is shorter than the drag relaxation time (see Test 2).
"""
function solve_steady_compressible(u0::AbstractVector, params::CompressibleParams;
                                   t_end=1e3, solver=Tsit5(), kwargs...)
    prob = ODEProblem(rhs_compressible!, u0, (0.0, Float64(t_end)), params)
    solve(prob, solver; abstol=1e-6, reltol=1e-5, save_everystep=false, kwargs...)
end

# 3-arg residual for NonlinearProblem / IFT adjoint
_residual_compressible!(du, u, p) = rhs_compressible!(du, u, p, zero(eltype(u)))

"""
    solve_steady_compressible_nk(u0, params; tol, maxiters, verbose)

Newton-Krylov steady-state solve of F(u; params) = 0 with `rhs_compressible!`.
Matrix-free GMRES inner solve (no dense Jacobian).  Warm-start from a
`solve_steady_compressible` result for robustness.
"""
function solve_steady_compressible_nk(u0::AbstractVector, params::CompressibleParams;
                                      tol=1e-8, maxiters=200, verbose=false)
    prob = NLS.NonlinearProblem(_residual_compressible!, copy(u0), params)
    NLS.solve(prob, NLS.NewtonRaphson(linsolve=NLS.KrylovJL_GMRES());
              abstol=tol, reltol=tol, maxiters=maxiters, verbose=verbose)
end

"""
    solve_steady_compressible_warmstart(u0, params; β_flow, t_flow, t_end)

Two-stage robustification for stiff reacting/expanding cases:
  Stage 1 — equilibrate the *cold* flow (high β_flow, short t_flow) to seed a
            sensible velocity/pressure field;
  Stage 2 — integrate the full coupled system from there with the requested β.
Returns the final state vector.
"""
function solve_steady_compressible_warmstart(u0::AbstractVector, params::CompressibleParams;
                                             β_flow=100.0, t_flow=2.0, t_end=20.0)
    T = eltype(u0)
    p_flow = CompressibleParams(params.grid, params.porous, params.gas, params.solid,
                                params.reactions, T(β_flow), params.gravity,
                                params.D_species, params.M_species, params.bcs)
    u1 = solve_steady_compressible(u0, p_flow; t_end=t_flow).u[end]
    solve_steady_compressible(u1, params; t_end=t_end).u[end]
end

"""
    ift_gradient(loss_fn, u_star, params::CompressibleParams, θ_getter, θ_setter; h)

IFT gradient of a scalar loss over a compressible steady state.
This mirrors `src/ad.jl` but uses the compressible residual and steady solver.
"""
function ift_gradient(loss_fn, u_star::AbstractVector, params::CompressibleParams,
                      θ_getter, θ_setter;
                      h=sqrt(eps(Float64)))
    n  = length(u_star)
    θ0 = θ_getter(params)
    nθ = length(θ0)

    dLdu = similar(u_star)
    L0   = loss_fn(u_star)
    for k in 1:n
        up      = copy(u_star); up[k] += h
        dLdu[k] = (loss_fn(up) - L0) / h
    end

    J  = zeros(n, n)
    f0 = similar(u_star)
    _residual_compressible!(f0, u_star, params)
    fp = similar(u_star)
    for k in 1:n
        up = copy(u_star); up[k] += h
        _residual_compressible!(fp, up, params)
        J[:, k] .= (fp .- f0) ./ h
    end

    λ = J' \ dLdu

    grad = similar(θ0)
    for k in 1:nθ
        θp = copy(θ0); θp[k] += h
        _residual_compressible!(fp, u_star, θ_setter(params, θp))
        grad[k] = -dot(λ, (fp .- f0) ./ h)
    end
    grad
end

"""
Reference finite-difference gradient for verifying compressible IFT gradients.
"""
function sensitivity_fd(loss_fn, u0, params::CompressibleParams,
                        θ_getter, θ_setter; h=1e-5)
    θ0   = θ_getter(params)
    grad = similar(θ0)
    L0   = loss_fn(solve_steady_compressible(u0, params).u)
    for k in eachindex(θ0)
        θp      = copy(θ0); θp[k] += h
        L_plus  = loss_fn(solve_steady_compressible(u0, θ_setter(params, θp)).u)
        grad[k] = (L_plus - L0) / h
    end
    grad
end
