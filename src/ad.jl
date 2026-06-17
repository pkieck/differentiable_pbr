"""
AD / sensitivity layer - Implicit Function Theorem adjoint.

At steady state u* satisfies F(u*; theta) = 0.
IFT gradient of scalar loss L(u*) w.r.t. parameters theta:
  dL/dtheta = -lambda^T * dF/dtheta(u*; theta),
  where lambda solves [dF/du]^T lambda = dL/du

Steps:
1. dL/du: finite differences on loss
2. dF/du: finite-difference Jacobian
3. Adjoint: lambda = (J^T) \\ (dL/du)
4. dF/dtheta: finite differences on F
5. Gradient: grad[k] = -lambda * dF/dtheta_k

Cost for N <= ~500: O(N^2) for Jacobian + O(N) for rest.
"""

include("solve.jl")

using LinearAlgebra: dot

# Jacobian build

"""
build_jacobian_fd(u, params; h) -> n x n matrix
Forward finite-difference Jacobian of rhs! at (u, params).
Column k = (F(u + h*e_k) - F(u)) / h. Cost: n+1 rhs! calls.
"""
function build_jacobian_fd(u::AbstractVector, params::ReactorParams;
                            h=sqrt(eps(Float64)))
    n  = length(u)
    J  = zeros(n, n)
    f0 = similar(u)
    _residual!(f0, u, params)
    fp = similar(u)
    for k in 1:n
        up = copy(u); up[k] += h
        _residual!(fp, up, params)
        J[:, k] .= (fp .- f0) ./ h
    end
    J
end

# ── IFT gradient ─────────────────────────────────────────────────────────────

"""
    ift_gradient(loss_fn, u_star, params, θ_getter, θ_setter; h, krylov_tol)

IFT gradient of `loss_fn(u*)` w.r.t. the parameter vector returned by
`θ_getter(params)`.

Arguments
─────────
  loss_fn   : u -> scalar
  u_star    : converged steady state (from solve_steady_nk or solve_steady)
  params    : ReactorParams at the operating point
  θ_getter  : params -> AbstractVector{Float64}
  θ_setter  : (params, θ) -> ReactorParams   (returns updated params)
  h         : finite-difference step size

Returns a gradient vector of the same length as θ_getter(params).

Adjoint solve
─────────────
Builds J = ∂F/∂u with n+1 rhs! calls, then solves Jᵀ λ = ∂L/∂u with Julia's
backslash.  For n ≤ ~500 this is fast.  For larger n, the path is: replace
build_jacobian_fd with a sparse FD Jacobian (SparseConnectivityTracer), and
replace the direct solve with GMRES using Enzyme vjp for mat-vec products.
"""
function ift_gradient(loss_fn, u_star::AbstractVector, params::ReactorParams,
                      θ_getter, θ_setter;
                      h=sqrt(eps(Float64)))
    n  = length(u_star)
    θ0 = θ_getter(params)
    nθ = length(θ0)

    # 1. dL/du via finite differences─────────
    dLdu = similar(u_star)
    L0   = loss_fn(u_star)
    for k in 1:n
        up      = copy(u_star); up[k] += h
        dLdu[k] = (loss_fn(up) - L0) / h
    end

    # 2. Build Jacobian J = dF/du─────
    J = build_jacobian_fd(u_star, params; h=h)

    # 3. Adjoint solve: J^T lambda = dL/du─────────────
    λ = J' \ dLdu

    # 4-5. dF/dtheta via FD, then contract with lambda──────────

    f0 = similar(u_star)
    _residual!(f0, u_star, params)

    grad = similar(θ0)
    for k in 1:nθ
        θp = copy(θ0); θp[k] += h
        fp = similar(u_star)
        _residual!(fp, u_star, θ_setter(params, θp))
        grad[k] = -dot(λ, (fp .- f0) ./ h)
    end
    grad
end

# Reference FD gradient

"""Reference finite-difference gradient for verifying ift_gradient.
Re-solves the full forward problem for each theta component.
"""
function sensitivity_fd(loss_fn, u0, params::ReactorParams,
                        θ_getter, θ_setter; h=1e-5)
    θ0   = θ_getter(params)
    grad = similar(θ0)
    L0   = loss_fn(solve_steady(u0, params).u)
    for k in eachindex(θ0)
        θp      = copy(θ0); θp[k] += h
        L_plus  = loss_fn(solve_steady(u0, θ_setter(params, θp)).u)
        grad[k] = (L_plus - L0) / h
    end
    grad
end
