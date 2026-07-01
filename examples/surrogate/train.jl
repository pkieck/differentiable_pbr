"""
train.jl — Train the DiT/Perceiver-IO surrogate with a Sobolev (field +
input-sensitivity) loss.

Loss = MSE(field) + λ · MSE(∂field/∂input)   [all in normalised space]

The input-sensitivity term is the payoff of the differentiable solver: the
dataset carries exact ∂(fields)/∂θ from ForwardDiff, and we train the surrogate's
own input-gradient to match it.  The surrogate's ∂ŷ/∂θ is obtained by a central
difference in (normalised) input space — 2·n_in extra forward passes — so the
whole loss differentiates with plain first-order Zygote (no nested AD).

Run:  julia --project examples/surrogate/train.jl [epochs]
Writes examples/surrogate/data/surrogate.jld2  (params, states, config, norm stats).
"""

include("wallloss_problem.jl")   # defines WallLossSetup (needed to deserialize)
include("model.jl")

using Serialization, JLD2, Printf
using Lux, Optimisers, Zygote, Random
using Statistics: mean, std

# ── load dataset ─────────────────────────────────────────────────────────────
const DATA_PATH = get(ENV, "WALLLOSS_DATA", joinpath(@__DIR__, "data", "wallloss_dataset.jls"))
const DATA = deserialize(DATA_PATH)

# ── normalisation ────────────────────────────────────────────────────────────
# inputs  : box [lo,hi] → [-1,1]
# outputs : per-field standardisation (mean/std over the training fields)
struct Norm
    lo::Vector{Float32}; hi::Vector{Float32}      # input box
    μ::Vector{Float32};  σ::Vector{Float32}       # per-field output stats
end

input_to_norm(N::Norm, θ)  = Float32.(2 .* (θ .- N.lo) ./ (N.hi .- N.lo) .- 1)
# d(θ)/d(θ_norm) = (hi-lo)/2  — used to map sensitivity targets into norm space
input_jac(N::Norm) = Float32.((N.hi .- N.lo) ./ 2)

function build_norm(DATA)
    nf = length(DATA.field_names)
    μ = Float32[mean(DATA.Y_train[:, :, f, :]) for f in 1:nf]
    σ = Float32[std(DATA.Y_train[:, :, f, :]) + 1f-8 for f in 1:nf]
    Norm(Float32.(DATA.lo), Float32.(DATA.hi), μ, σ)
end

"Pack a dataset split into normalised (θn, Yn, dYn) Float32 arrays."
function pack(DATA, N::Norm, Θ, Y, dY)
    nr, nz, nf, B = size(Y)
    θn  = input_to_norm(N, Θ)                              # (n_in, B)
    μ = reshape(N.μ, 1, 1, nf, 1); σ = reshape(N.σ, 1, 1, nf, 1)
    Yn  = Float32.((Y .- μ) ./ σ)                          # (nr,nz,nf,B)
    ij  = reshape(input_jac(N), 1, 1, 1, :, 1)             # (1,1,1,n_in,1)
    σ5  = reshape(N.σ, 1, 1, nf, 1, 1)
    dYn = Float32.(dY .* ij ./ σ5)                         # ∂Yn/∂θn (nr,nz,nf,n_in,B)
    return θn, Yn, dYn
end

# ── Sobolev loss ─────────────────────────────────────────────────────────────
function sobolev_loss(model, ps, st, θn, Yn, dYn; h::Float32, λ::Float32)
    n_in = size(θn, 1)
    ŷ = first(model(θn, ps, st))
    Lf = mean(abs2, ŷ .- Yn)
    Ls = zero(Lf)
    for k in 1:n_in
        # unit perturbation in input k, built by broadcasting (no mutation → Zygote-safe)
        ek = reshape(h .* Float32.((1:n_in) .== k), n_in, 1)
        yp = first(model(θn .+ ek, ps, st))
        ym = first(model(θn .- ek, ps, st))
        Jk = (yp .- ym) ./ (2h)
        Ls += mean(abs2, Jk .- @view dYn[:, :, :, k, :])
    end
    return Lf + λ * Ls / n_in, Lf, Ls / n_in
end

# ── training loop ────────────────────────────────────────────────────────────
cosine_lr(ep, epochs, lr0; warmup = 10) =
    ep ≤ warmup ? lr0 * ep / warmup :
    lr0 * 0.5f0 * (1 + cos(Float32(π) * (ep - warmup) / (epochs - warmup)))

function main()
    epochs   = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 250
    batch    = 64
    h        = 0.02f0
    λ        = 0.25f0
    lr0      = 6f-4
    rng      = Random.MersenneTwister(42)

    N = build_norm(DATA)
    θn_tr, Yn_tr, dYn_tr = pack(DATA, N, DATA.theta_train, DATA.Y_train, DATA.dY_train)
    θn_te, Yn_te, dYn_te = pack(DATA, N, DATA.theta_test,  DATA.Y_test,  DATA.dY_test)
    Ntr = size(θn_tr, 2)
    @info "data" Ntr Nte=size(θn_te, 2) input=size(θn_tr, 1) fields=size(Yn_tr, 3)

    cfg   = (; nr=DATA.setup.nr, nz=DATA.setup.nz, n_in=size(θn_tr, 1),
              n_out=length(DATA.field_names), d_model=48, nheads=4,
              patch=(2, 4), n_proc=3, n_dec=2)
    model = build_surrogate(; cfg...)
    ps, st = Lux.setup(rng, model)
    opt    = Optimisers.setup(Optimisers.AdamW(lr0, (0.9f0, 0.999f0), 1f-4), ps)
    @info "model" params=Lux.parameterlength(model) cfg

    evalloss(θn, Yn, dYn) = sobolev_loss(model, ps, st, θn, Yn, dYn; h, λ)
    ckpt() = joinpath(@__DIR__, "data", "surrogate.jld2")
    save_ckpt(ep, Lte) = jldsave(ckpt(); ps, st, cfg,
        norm=(lo=N.lo, hi=N.hi, mu=N.μ, sigma=N.σ),
        field_names=DATA.field_names, input_names=DATA.input_names, epoch=ep, test_loss=Lte)

    best = Inf
    for ep in 1:epochs
        Optimisers.adjust!(opt, cosine_lr(ep, epochs, lr0))
        idx = Random.randperm(rng, Ntr)
        for s in 1:batch:Ntr
            b = idx[s:min(s + batch - 1, Ntr)]
            θb = θn_tr[:, b]; Yb = Yn_tr[:, :, :, b]; dYb = dYn_tr[:, :, :, :, b]
            (_, gs) = Zygote.withgradient(p -> first(sobolev_loss(model, p, st, θb, Yb, dYb; h, λ)), ps)
            opt, ps = Optimisers.update(opt, ps, gs[1])
        end
        if ep % 10 == 0 || ep == 1 || ep == epochs
            Ltr, = evalloss(θn_tr, Yn_tr, dYn_tr)
            Lte, Lf_te, Ls_te = evalloss(θn_te, Yn_te, dYn_te)
            @printf("epoch %4d  lr=%.2e  train=%.4e  test=%.4e  (field=%.4e sob=%.4e)\n",
                    ep, cosine_lr(ep, epochs, lr0), Ltr, Lte, Lf_te, Ls_te)
            flush(stdout)
            if Lte < best; best = Lte; save_ckpt(ep, Lte); end
        end
    end
    @printf("done — best test loss = %.4e\n", best); flush(stdout)
end

main()
