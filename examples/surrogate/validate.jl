"""
validate.jl — Evaluate the trained surrogate against the full solver.

Three checks:
  (A) Accuracy   — per-field relative L2 error on the held-out test set, plus
                   parity of two scalar QoIs (outlet conversion, hot-spot T).
  (B) Gradients  — the differentiability payoff: ∂QoI/∂θ from the surrogate
                   (AD through the network) vs from the solver (ForwardDiff
                   through run_case).  An accurate surrogate *gradient* is what
                   makes the surrogate usable for inverse design.
  (C) Inverse design — one round: use the surrogate's gradient to hit a target
                   outlet conversion, then verify the chosen θ with the solver.

Also writes examples/figures/surrogate_compare.png (true vs predicted fields for
one representative test case).

Run:  julia --project examples/surrogate/validate.jl
"""

include("wallloss_problem.jl")     # run_case, WallLossSetup, OUT_FIELD_NAMES
include("model.jl")
include("../plotting.jl")

using JLD2, Serialization, ForwardDiff, Printf, Statistics
using Lux, Random
using Plots
using LinearAlgebra: dot, norm

# ── load trained surrogate ───────────────────────────────────────────────────
const CKPT_PATH = get(ENV, "WALLLOSS_CKPT", joinpath(@__DIR__, "data", "surrogate.jld2"))
const CKPT = jldopen(CKPT_PATH, "r")
const PS   = CKPT["ps"];  const ST = CKPT["st"];  const CFG = CKPT["cfg"]
const NRM  = CKPT["norm"]; const FIELD_NAMES = CKPT["field_names"]
const MODEL = build_surrogate(; CFG...)
close(CKPT)

const LO = Float32.(NRM.lo); const HI = Float32.(NRM.hi)
const MU = NRM.mu;           const SIG = NRM.sigma

"Surrogate prediction θ (length-3 phys) → (nr,nz,6) denormalised fields. Differentiable."
function predict(θ)
    T   = eltype(θ)
    θn  = T(2) .* (θ .- T.(LO)) ./ T.(HI .- LO) .- T(1)
    Yn  = first(MODEL(reshape(θn, :, 1), PS, ST))          # (nr,nz,6,1)
    nf  = size(Yn, 3)
    μ = reshape(T.(MU), 1, 1, nf); σ = reshape(T.(SIG), 1, 1, nf)
    return Yn[:, :, :, 1] .* σ .+ μ
end

# scalar QoIs from a (nr,nz,6) field stack
conversion(Y) = 1 - mean(@view Y[:, end, 5])               # 1 − ⟨Y_A⟩ at outlet
hotspot(Y)    = maximum(@view Y[:, :, 3])                   # peak fluid T

# ── (A) accuracy on the test set ─────────────────────────────────────────────
function accuracy(DATA)
    Θ = Float32.(DATA.theta_test); Ytrue = DATA.Y_test
    nf = size(Ytrue, 3); Nte = size(Θ, 2)
    Ypred = similar(Ytrue)
    for i in 1:Nte
        Ypred[:, :, :, i] = predict(Θ[:, i])
    end
    println("── (A) test-set accuracy ($(Nte) cases) ──")
    for f in 1:nf
        relL2 = sqrt(sum(abs2, Ypred[:,:,f,:] .- Ytrue[:,:,f,:]) /
                     sum(abs2, Ytrue[:,:,f,:] .- mean(Ytrue[:,:,f,:])))
        mae   = mean(abs, Ypred[:,:,f,:] .- Ytrue[:,:,f,:])
        @printf "  %-4s  rel-L2 = %.3e   MAE = %.3e\n" FIELD_NAMES[f] relL2 mae
    end
    Xp = [conversion(Ypred[:,:,:,i]) for i in 1:Nte]
    Xt = [conversion(Ytrue[:,:,:,i]) for i in 1:Nte]
    Hp = [hotspot(Ypred[:,:,:,i])    for i in 1:Nte]
    Ht = [hotspot(Ytrue[:,:,:,i])    for i in 1:Nte]
    @printf "  outlet conversion X : MAE = %.4f  (range %.2f–%.2f)\n" mean(abs.(Xp.-Xt)) minimum(Xt) maximum(Xt)
    @printf "  hot-spot T_f [K]    : MAE = %.2f K (range %.0f–%.0f)\n" mean(abs.(Hp.-Ht)) minimum(Ht) maximum(Ht)
    return Ypred, Ytrue, Θ
end

# ── (B) gradient check: surrogate vs solver ──────────────────────────────────
function gradient_check(setup)
    println("\n── (B) ∂QoI/∂θ: surrogate (AD) vs solver (ForwardDiff) ──")
    pts = ([380.0, 0.009, 2.5], [420.0, 0.014, 4.0], [360.0, 0.016, 6.0])
    for θ0 in pts
        gs_surr = ForwardDiff.gradient(θ -> conversion(predict(θ)), θ0)
        gs_solv = ForwardDiff.gradient(θ -> conversion(run_case(θ, setup)), θ0)
        @printf "  θ=(T=%.0f,U=%.3f,m=%.1f)\n" θ0[1] θ0[2] θ0[3]
        @printf "    ∂X/∂T_in  surrogate=%+.3e  solver=%+.3e\n" gs_surr[1] gs_solv[1]
        @printf "    ∂X/∂U     surrogate=%+.3e  solver=%+.3e\n" gs_surr[2] gs_solv[2]
        @printf "    ∂X/∂m     surrogate=%+.3e  solver=%+.3e\n" gs_surr[3] gs_solv[3]
        cossim = dot(gs_surr, gs_solv) / (norm(gs_surr) * norm(gs_solv))
        @printf "    cosine-similarity of gradient = %.3f\n" cossim
    end
end

# ── (C) one inverse-design round using the surrogate gradient ────────────────
function inverse_design(setup; X_target = 0.30, θ0 = [400.0, 0.012, 3.0])
    println("\n── (C) inverse design: hit X_target = $(X_target) via surrogate gradient ──")
    θ = copy(θ0); lr = [3e3, 3e-4, 0.2]                  # per-input step scales
    for it in 1:60
        g = ForwardDiff.gradient(t -> (conversion(predict(t)) - X_target)^2, θ)
        θ .-= lr .* g
        θ .= clamp.(θ, LO, HI)
    end
    X_surr = conversion(predict(θ))
    X_solv = conversion(run_case(θ, setup))
    @printf "  found θ = (T_in=%.1f K, U=%.4f m/s, m=%.2f)\n" θ[1] θ[2] θ[3]
    @printf "  surrogate X = %.3f   solver X = %.3f   (target %.3f)\n" X_surr X_solv X_target
end

# ── comparison figure ────────────────────────────────────────────────────────
function compare_figure(Ypred, Ytrue, Θ, setup; idx = 1)
    g = Grid2D(setup.nr, setup.nz, setup.R, setup.L)
    θ = Θ[:, idx]
    panels = []
    for f in (3, 5, 6)                                     # T_f, Y_A, Y_B
        yt = Ytrue[:, :, f, idx]; yp = Ypred[:, :, f, idx]
        push!(panels, heatmap(g.z, g.r, yt, title="$(FIELD_NAMES[f]) true", c=:viridis))
        push!(panels, heatmap(g.z, g.r, yp, title="$(FIELD_NAMES[f]) pred", c=:viridis))
        push!(panels, heatmap(g.z, g.r, yp .- yt, title="$(FIELD_NAMES[f]) err", c=:balance))
    end
    fig = plot(panels...; layout=(3, 3), size=(1100, 700),
               plot_title=@sprintf("test θ: T_in=%.0f K, U=%.4f, m=%.2f", θ[1], θ[2], θ[3]))
    save_fig(fig, "surrogate_compare.png")
    println("\nwrote examples/figures/surrogate_compare.png")
end

function main()
    DATA  = deserialize(joinpath(@__DIR__, "data", "wallloss_dataset.jls"))
    setup = DATA.setup
    Ypred, Ytrue, Θ = accuracy(DATA)
    gradient_check(setup)
    inverse_design(setup)
    compare_figure(Ypred, Ytrue, Θ, setup; idx = 1)
end

main()
