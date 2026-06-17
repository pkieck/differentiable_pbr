"""
Physical closures for packed-bed solver.

All functions are pure, type-stable, and smooth for AD gradients.
|u| is regularized as sqrt(u_r^2 + u_z^2 + SMOOTH_VEL^2) to avoid kinks.
"""

const SMOOTH_VEL = 1e-10   # regularisation for velocity magnitude [m/s]

# Ergun / Forchheimer drag

"Ergun A coefficient (viscous drag) [1/m2]"
@inline ergun_A(eps, dp) = 150 * (1 - eps)^2 / (eps^3 * dp^2)

"Ergun B coefficient (inertial drag) [1/m]"
@inline ergun_B(eps, dp) = 1.75 * (1 - eps) / (eps^3 * dp)

"Darcy permeability K [m2]"
@inline darcy_K(eps, dp)  = eps^3 * dp^2 / (150 * (1 - eps + 1e-14)^2)

"""Drag coefficient: D = mu*A + rho*B*|u| [Pa*s/m2]."""
@inline function drag_coeff(eps, dp, rho, mu, ur, uz)
    umag = sqrt(ur^2 + uz^2 + SMOOTH_VEL^2)
    mu * ergun_A(eps, dp) + rho * ergun_B(eps, dp) * umag
end

# Fluid-solid heat transfer

"""Fluid-solid heat-transfer coefficient h_fs [W/(m2*K)].
Wakao-Funazkri correlation: Nu_p = 2 + 1.1 Re_p^0.6 Pr^(1/3)
"""
@inline function h_fs(eps, dp, rho, mu, cp_f, lam_f, ur, uz)
    umag  = sqrt(ur^2 + uz^2 + SMOOTH_VEL^2)
    Re_p  = rho * umag * dp / (mu + 1e-30)
    Pr    = mu * cp_f / (lam_f + 1e-30)  # Prandtl number
    Nu    = 2 + 1.1 * Re_p^0.6 * Pr^(1/3)
    Nu * lam_f / (dp + 1e-30)
end

# Effective transport properties

"Effective fluid thermal conductivity [W/(m*K)]."
@inline lam_f_eff(eps, lam_f) = eps * lam_f

"Effective solid thermal conductivity [W/(m*K)]."
@inline lam_s_eff(eps, lam_s) = (1 - eps) * lam_s

"Effective species diffusivity [m2/s] with tortuosity ~ 1.5."
@inline D_eff(eps, D_k) = eps * D_k / 1.5

# Effective viscosity for Brinkman term

"""Effective (Brinkman) viscosity mu_eff [Pa*s].
Common choice: mu_eff = mu/eps (Nield & Bejan).
"""
@inline mu_eff(eps, mu) = mu / (eps + 1e-14)

# Simple Arrhenius kinetics

"""Arrhenius rate [mol/(m3*s)]: r = A_pre * exp(-E_a/(R_gas*T)) * prod(C_k^nu_k)
C_k = rho * Y_k / M_k [mol/m3].
"""
const R_GAS = 8.314  # J/(mol·K)

@inline function arrhenius_rate(A_pre, E_a, T, C_reactants...)
    k = A_pre * exp(-E_a / (R_GAS * T))
    r = k
    for C in C_reactants
        r *= max(C, zero(C))
    end
    r
end
