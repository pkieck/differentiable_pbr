"""
Face-flux computations for the FVM discretisation.

All limiters and flux functions are smooth everywhere for AD gradients.

Van Albada limiter: phi(r) = (r^2 + r) / (r^2 + 1)
Gives second-order accurate MUSCL reconstruction without minmod kinks.

Smooth upwind flux avoids non-differentiable terms at u=0.
"""

const FLUX_EPS = 1e-20   # limiter denominator guard

# Slope limiter

"Van Albada limiter: smooth TVD, r = upwind/downwind slope ratio."
@inline van_albada(r) = (r^2 + r) / (r^2 + 1 + FLUX_EPS)

"""
MUSCL left- and right-state reconstruction at the face between cells L and R,
using the cell to the left of L (LL) and the cell to the right of R (RR).
Returns (q_L_face, q_R_face).
"""
@inline function muscl_reconstruct(q_LL, q_L, q_R, q_RR)
    dL = q_R  - q_L
    dR = q_RR - q_R
    d0 = q_L  - q_LL
    # slope ratio for left cell
    r_L = dL / (d0 + FLUX_EPS)
    phi_L = van_albada(r_L)
    # slope ratio for right cell
    r_R = dL / (dR + FLUX_EPS)
    phi_R = van_albada(r_R)
    q_L_face = q_L + 0.5 * phi_L * d0
    q_R_face = q_R - 0.5 * phi_R * dR
    q_L_face, q_R_face
end

# Advective flux

"Smooth upwind advective flux for scalar q with normal velocity u."
@inline function adv_flux(u, q_L, q_R)
    abs_u = sqrt(u^2 + SMOOTH_VEL^2)
    0.5 * u * (q_L + q_R) - 0.5 * abs_u * (q_R - q_L)
end

# Diffusive flux

"Central-difference diffusive flux -D * (q_R - q_L) / dx."
@inline diff_flux(D, q_L, q_R, dx) = D * (q_R - q_L) / dx

# Combined scalar transport flux

"Total scalar flux (advective + diffusive) at a face."

@inline function scalar_face_flux(u, D, q_L, q_R, dx)
    adv_flux(u, q_L, q_R) - diff_flux(D, q_L, q_R, dx)
end

# Face velocity interpolation

"Linear average of adjacent cell velocities."
@inline face_vel(u_a, u_b) = 0.5 * (u_a + u_b)
