"""
Full PDE right-hand side for the packed-bed reactor solver.

State vector layout (field-major, each field is nr×nz, column-major):
  [u_r | u_z | p | T_f | T_s | Y_1 | ... | Y_Ns]

Equations
─────────
Continuity (artificial compressibility):
  ∂p/∂t = −β² ∇·(ε ρ u)

r-Momentum (Darcy–Brinkman–Forchheimer):
  (ρ/ε) ∂u_r/∂t = −∂p/∂r + μ_eff [∇²u_r − u_r/r²] − D·u_r

z-Momentum:
  (ρ/ε) ∂u_z/∂t = −∂p/∂z + μ_eff ∇²u_z − D·u_z − ρ g

Fluid energy:
  ε ρ c_pf ∂T_f/∂t = −ρ c_pf (u·∇T_f) + ∇·(λ_f,eff ∇T_f) + h_fs a_v (T_s−T_f)

Solid energy:
  (1−ε) ρ_s c_ps ∂T_s/∂t = ∇·(λ_s,eff ∇T_s) − h_fs a_v (T_s−T_f)
                           + φ_cat Σ_j (−ΔH_j) r_j

Species k:
  ε ρ ∂Y_k/∂t = −ρ (u·∇Y_k) + ∇·(ρ D_k,eff ∇Y_k) + φ_cat M_k Σ_j ν_kj r_j

Axisymmetric FVM divergence formula (per plan, §2):
  div(F)[i,j] = (rf[i+1] F_r[i+1] − rf[i] F_r[i]) / (r[i] dr)
              + (F_z[j+1] − F_z[j])                 / dz
"""

include("grid.jl")
include("fields.jl")
include("porosity.jl")
include("closures.jl")
include("fluxes.jl")
include("params.jl")

# Boundary condition helpers─────

# Ghost-cell value for Dirichlet BC: returns prescribed value.
@inline ghost_dirichlet(q_bc, q_interior) = 2*q_bc - q_interior

# Ghost-cell value for Neumann BC (zero flux): mirror the interior.
@inline ghost_neumann(q_interior) = q_interior

# Main RHS─────────────

"""
    rhs!(du, u, p, t)

In-place right-hand side compatible with OrdinaryDiffEq.jl.
`p` is a `ReactorParams` struct; `t` is time (unused at steady state).
"""
function rhs!(du, u, p::ReactorParams, t)
    g   = p.grid
    s   = StateLayout(g.nr, g.nz, length(p.D_species))
    pm  = p.porous
    fl  = p.fluid
    sl  = p.solid
    beta   = p.beta_ac
    grav = p.gravity

    nr, nz = g.nr, g.nz
    dr, dz = g.dr, g.dz

    # Views into state
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

    Ns = s.Ns
    # Pre-fetch species views (avoid repeated indexing)
    Y_views  = ntuple(k -> field_mat(u,  s, 5+k), Ns)
    dY_views = ntuple(k -> field_mat(du, s, 5+k), Ns)

    fill!(du, zero(eltype(du)))

    rho  = fl.rho
    mu   = fl.mu
    cpf  = fl.cp
    lamf = fl.lambda
    rhos = sl.rho
    cps  = sl.cp
    lams = sl.lambda

    for j in 1:nz, i in 1:nr
        eps   = pm.eps[i,j]
        dprt  = pm.dp[i,j]
        phi   = pm.phi_cat[i,j]
        av    = pm.av[i,j]
        r_c   = g.r[i]

        ur_c  = ur[i,j]
        uz_c  = uz[i,j]
        Tf_c  = Tf[i,j]
        Ts_c  = Ts[i,j]
        pp_c  = pp[i,j]

        # ── Ghost-cell values for BCs ────────────────────────────────────
        # r direction
        # Axis (i=1, left face): symmetry → u_r=0, Neumann for scalars
        ur_im1 = (i == 1)  ? -ur_c   : ur[i-1,j]   # ghost: u_r=0 at axis
        ur_ip1 = (i == nr) ? -ur_c   : ur[i+1,j]   # ghost: no-slip wall u_r=0
        uz_im1 = (i == 1)  ?  uz_c   : uz[i-1,j]   # ghost: symmetry duz/dr=0
        uz_ip1 = (i == nr) ? -uz_c   : uz[i+1,j]   # ghost: no-slip wall uz=0
        pp_im1 = (i == 1)  ?  pp_c   : pp[i-1,j]   # ghost: dp/dr=0 (Neumann)
        pp_ip1 = (i == nr) ?  pp_c   : pp[i+1,j]   # ghost: dp/dr=0 at wall
        Tf_im1 = (i == 1)  ?  Tf_c   : Tf[i-1,j]
        Tf_ip1 = (i == nr) ?  Tf_c   : Tf[i+1,j]   # insulated wall (Neumann)
        Ts_im1 = (i == 1)  ?  Ts_c   : Ts[i-1,j]
        Ts_ip1 = (i == nr) ?  Ts_c   : Ts[i+1,j]

        # z direction
        # Inlet (j=1): Dirichlet BCs for uz, ur, Tf, Ts, Y; Neumann for p.
        #
        # Two separate ghost-cell values for each Dirichlet scalar:
        #   _diff: standard ghost for DIFFUSIVE flux  → 2·BC - interior
        #          (ensures face value = BC in central-difference stencil)
        #   _adv:  upwind ghost for ADVECTIVE flux   → BC directly
        #          (ensures upwind scheme injects fluid at the physical inlet T,
        #           not at 2·BC - interior which is wrong when interior > BC)
        #
        # For momentum (uz, ur) the same ghost is used for both; for scalars
        # (T, Y) we split. This is the correct FVM treatment of Dirichlet inlet
        # BCs with first-order upwind advection.
        uz_bc  = p.bcs.uz_in
        Tf_bc  = p.bcs.Tf_in
        Ts_bc  = p.bcs.Ts_in

        # velocity ghosts (same for advection and diffusion — momentum uses Dirichlet on velocity)
        uz_jm1 = (j == 1)  ? 2*uz_bc - uz_c   : uz[i,j-1]
        uz_jp1 = (j == nz) ? uz_c              : uz[i,j+1]   # Neumann at outlet
        ur_jm1 = (j == 1)  ? -ur_c             : ur[i,j-1]   # ur=0 at inlet face
        ur_jp1 = (j == nz) ? ur_c              : ur[i,j+1]

        # pressure ghosts (Neumann at inlet, Dirichlet p=0 at outlet)
        pp_jm1 = (j == 1)  ? pp_c              : pp[i,j-1]
        pp_jp1 = (j == nz) ? -pp_c             : pp[i,j+1]

        # Tf: separate ghosts for diffusion (central) and advection (upwind-correct)
        Tf_jm1_diff = (j == 1)  ? 2*Tf_bc - Tf_c   : Tf[i,j-1]
        Tf_jm1_adv  = (j == 1)  ? Tf_bc             : Tf[i,j-1]
        Tf_jp1      = (j == nz) ? Tf_c              : Tf[i,j+1]

        # Ts: same split
        Ts_jm1_diff = (j == 1)  ? 2*Ts_bc - Ts_c   : Ts[i,j-1]
        Ts_jm1_adv  = (j == 1)  ? Ts_bc             : Ts[i,j-1]
        Ts_jp1      = (j == nz) ? Ts_c              : Ts[i,j+1]

        # Convenience aliases for code that uses only one ghost (diffusion-dominant contexts)
        Tf_jm1 = Tf_jm1_diff
        Ts_jm1 = Ts_jm1_diff

        # ── Face velocities ──────────────────────────────────────────────
        # r-faces: at rf[i] and rf[i+1]
        ur_r_face_m = face_vel(ur_im1, ur_c)   # at rf[i]
        ur_r_face_p = face_vel(ur_c, ur_ip1)   # at rf[i+1]
        uz_r_face_m = face_vel(uz_im1, uz_c)
        uz_r_face_p = face_vel(uz_c, uz_ip1)

        # z-faces: at zf[j] and zf[j+1]
        ur_z_face_m = face_vel(ur_jm1, ur_c)
        ur_z_face_p = face_vel(ur_c, ur_jp1)
        uz_z_face_m = face_vel(uz_jm1, uz_c)
        uz_z_face_p = face_vel(uz_c, uz_jp1)

        rf_m = g.rf[i]      # left r-face coordinate (= 0 at axis)
        rf_p = g.rf[i+1]    # right r-face coordinate
        Ar   = r_c * dr     # = (rf_p² - rf_m²)/2, exact for uniform grid

        # Discrete divergence of the advecting velocity, using the SAME face
        # velocities the scalar advection uses below.  Subtracting q_c·div_u
        # from the conservative flux ∇·(u q) yields the non-conservative form
        # u·∇q, which annihilates a uniform field even when div_u ≠ 0 (e.g. an
        # under-converged / frozen velocity field).  Without this, a spurious
        # source −ρc_p·T·div_u scales with T(~400 K) and corrupts the energy
        # balance even at tiny continuity residuals.
        div_u_adv = ( rf_p * ur_r_face_p - rf_m * ur_r_face_m ) / Ar +
                    ( uz_z_face_p - uz_z_face_m ) / dz

        # ── Continuity (artificial compressibility) ──────────────────────
        # div(ε ρ u) in axisymmetric FVM:
        #   = (rf_p * ε*ρ*ur_face_p - rf_m * ε*ρ*ur_face_m) / (r_c * dr)
        #   + (ε*ρ*uz_face_p - ε*ρ*uz_face_m) / dz
        div_u = ( rf_p * eps * rho * ur_r_face_p
                - rf_m * eps * rho * ur_r_face_m ) / Ar +
                ( eps * rho * uz_z_face_p
                - eps * rho * uz_z_face_m ) / dz

        dpp[i,j] = -beta^2 * div_u

        # ── Drag coefficient ─────────────────────────────────────────────
        D_drag = drag_coeff(eps, dprt, rho, mu, ur_c, uz_c)
        mu_eff_val  = mu_eff(eps, mu)

        # ── r-Momentum ───────────────────────────────────────────────────
        # Pressure gradient ∂p/∂r (central difference)
        dp_dr  = (pp_ip1 - pp_im1) / (2*dr)

        # Brinkman viscous term: ∇²u_r (in cylindrical) − u_r/r²
        # ∇²u_r in FVM: (rf_p*(ur_ip1-ur_c)/dr - rf_m*(ur_c-ur_im1)/dr) / Ar
        #             + (ur_jp1 - 2*ur_c + ur_jm1) / dz²
        visc_r_r = ( rf_p * (ur_ip1 - ur_c) / dr
                   - rf_m * (ur_c - ur_im1) / dr ) / Ar
        visc_r_z = (ur_jp1 - 2*ur_c + ur_jm1) / dz^2
        cent_r   = ur_c / (r_c^2 + 1e-20)   # 1/r² term; regularised at axis
        visc_r   = visc_r_r + visc_r_z - cent_r

        dur[i,j] = eps / rho * (-dp_dr + mu_eff_val * visc_r - D_drag * ur_c)

        # ── z-Momentum ───────────────────────────────────────────────────
        dp_dz  = (pp_jp1 - pp_jm1) / (2*dz)

        visc_z_r = ( rf_p * (uz_ip1 - uz_c) / dr
                   - rf_m * (uz_c - uz_im1) / dr ) / Ar
        visc_z_z = (uz_jp1 - 2*uz_c + uz_jm1) / dz^2
        visc_z   = visc_z_r + visc_z_z

        dur_grav = zero(grav)   # gravity acts on z only
        duz[i,j] = eps / rho * (-dp_dz + mu_eff_val * visc_z - D_drag * uz_c - rho * grav)

        # ── Fluid energy ─────────────────────────────────────────────────
        λf_eff = lam_f_eff(eps, lamf)
        h      = h_fs(eps, dprt, rho, mu, cpf, lamf, ur_c, uz_c)

        # Advective flux: u·∇Tf  (use _adv ghost at inlet for correct upwind)
        adv_Tf = ( rf_p * adv_flux(ur_r_face_p, Tf_c, Tf_ip1)
                 - rf_m * adv_flux(ur_r_face_m, Tf_im1, Tf_c) ) / Ar +
                 ( adv_flux(uz_z_face_p, Tf_c, Tf_jp1)
                 - adv_flux(uz_z_face_m, Tf_jm1_adv, Tf_c) ) / dz
        adv_Tf -= Tf_c * div_u_adv   # → non-conservative u·∇Tf

        # Diffusive flux: ∇·(λ_eff ∇T)  (use _diff ghost at inlet)
        diff_Tf = ( rf_p * diff_flux(λf_eff, Tf_c, Tf_ip1, dr)
                  - rf_m * diff_flux(λf_eff, Tf_im1, Tf_c, dr) ) / Ar +
                  ( diff_flux(λf_eff, Tf_c, Tf_jp1, dz)
                  - diff_flux(λf_eff, Tf_jm1_diff, Tf_c, dz) ) / dz

        interphase_Tf = h * av * (Ts_c - Tf_c)

        dTf[i,j] = ( -rho * cpf * adv_Tf + diff_Tf + interphase_Tf ) /
                   ( eps * rho * cpf )

        # ── Solid energy ─────────────────────────────────────────────────
        λs_eff = lam_s_eff(eps, lams)

        diff_Ts = ( rf_p * diff_flux(λs_eff, Ts_c, Ts_ip1, dr)
                  - rf_m * diff_flux(λs_eff, Ts_im1, Ts_c, dr) ) / Ar +
                  ( diff_flux(λs_eff, Ts_c, Ts_jp1, dz)
                  - diff_flux(λs_eff, Ts_jm1_diff, Ts_c, dz) ) / dz

        # Chemistry heat source: φ_cat * Σ_j (-ΔH_j) r_j
        Q_rxn = zero(eltype(u))
        for rxn in p.reactions
            T_rxn = Ts_c
            # Build concentration vector from Y views
            C_args = ntuple(k -> begin
                Y_k = Y_views[k][i,j]
                # Non-reactants contribute a factor of 1 (not 0) so they don't
                # zero out the product in arrhenius_rate.
                rxn.reactant_idx[k] ? rho * Y_k / rxn.M[k] : one(Y_k)
            end, Ns)
            r_j = arrhenius_rate(rxn.A_pre, rxn.E_a, T_rxn, C_args...)
            Q_rxn += (-rxn.dH) * r_j
        end

        # Floor prevents 0/0 when ε→1 (fluid-only cells, no solid phase).
        # Numerator also → 0 in that limit (λ_s,eff and a_v both vanish).
        solid_cap = max((1 - eps) * rhos * cps, 1e-20)
        dTs[i,j] = ( diff_Ts - h * av * (Ts_c - Tf_c) + phi * Q_rxn ) / solid_cap

        # ── Species transport ─────────────────────────────────────────────
        for k in 1:Ns
            Y  = Y_views[k]
            dY = dY_views[k]
            Dk_eff = D_eff(eps, p.D_species[k])

            Yk_c   = Y[i,j]
            Yk_im1 = (i == 1)  ? Yk_c  : Y[i-1,j]
            Yk_ip1 = (i == nr) ? Yk_c  : Y[i+1,j]
            # Inlet: separate advection (BC value) and diffusion (Dirichlet ghost) ghosts.
            # Outlet: Neumann (zero flux across boundary).
            Yk_bc       = (k ≤ length(p.bcs.Y_in)) ? p.bcs.Y_in[k] : Yk_c
            Yk_jm1_adv  = (j == 1)  ? Yk_bc            : Y[i,j-1]
            Yk_jm1_diff = (j == 1)  ? 2*Yk_bc - Yk_c   : Y[i,j-1]
            Yk_jp1      = (j == nz) ? Yk_c              : Y[i,j+1]

            adv_Yk = ( rf_p * adv_flux(ur_r_face_p, Yk_c, Yk_ip1)
                     - rf_m * adv_flux(ur_r_face_m, Yk_im1, Yk_c) ) / Ar +
                     ( adv_flux(uz_z_face_p, Yk_c, Yk_jp1)
                     - adv_flux(uz_z_face_m, Yk_jm1_adv, Yk_c) ) / dz
            adv_Yk -= Yk_c * div_u_adv   # → non-conservative u·∇Yk

            diff_Yk = ( rf_p * diff_flux(rho * Dk_eff, Yk_c, Yk_ip1, dr)
                      - rf_m * diff_flux(rho * Dk_eff, Yk_im1, Yk_c, dr) ) / Ar +
                      ( diff_flux(rho * Dk_eff, Yk_c, Yk_jp1, dz)
                      - diff_flux(rho * Dk_eff, Yk_jm1_diff, Yk_c, dz) ) / dz

            # Species source from reactions
            Sk = zero(eltype(u))
            for rxn in p.reactions
                C_args = ntuple(l -> begin
                    Yl = Y_views[l][i,j]
                    rxn.reactant_idx[l] ? rho * Yl / rxn.M[l] : one(Yl)
                end, Ns)
                r_j = arrhenius_rate(rxn.A_pre, rxn.E_a, Ts_c, C_args...)
                Sk += phi * rxn.M[k] * rxn.nu[k] * r_j
            end

            dY[i,j] = (-rho * adv_Yk + diff_Yk + Sk) / (eps * rho)
        end

    end  # cell loop

    nothing
end
