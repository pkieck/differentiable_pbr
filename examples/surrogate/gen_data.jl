"""
gen_data.jl — Generate the surrogate training dataset from the differentiable
wall-heatloss reactor.

For each sampled input θ = [T_in, U, m] we store BOTH:
  • the steady 6-field solution      Y  ∈ ℝ^(nr×nz×6)
  • its input-sensitivity Jacobian   dY ∈ ℝ^(nr×nz×6×3)   (∂fields/∂θ via ForwardDiff)

The Jacobian is the "Sobolev" target: training the surrogate to match dY as well
as Y exploits the solver's differentiability for far better data efficiency and —
crucially for inverse design — accurate surrogate gradients.

Inputs are drawn by Latin-Hypercube sampling over a physical box:
    T_in ∈ [350, 450] K
    U    ∈ [0.006, 0.018] m/s
    m    ∈ [1.5, 8.0]   (inlet profile shape: 1.5 peaked → 2 parabolic → 8 plug)

Run from the project root (threads strongly recommended):
    julia --project -t auto examples/surrogate/gen_data.jl [N_train] [N_test]

Writes examples/surrogate/data/wallloss_dataset.jls (Julia Serialization).
"""

include("wallloss_problem.jl")
using ForwardDiff
const DiffResults = ForwardDiff.DiffResults
using Random, Serialization
using Base.Threads: @threads, nthreads, threadid

# ── sampling box ─────────────────────────────────────────────────────────────
const LO = [350.0, 0.006, 1.5]
const HI = [450.0, 0.018, 8.0]
const INPUT_NAMES = ["T_in", "U", "m"]

"Latin-Hypercube sample: returns a (3, N) matrix in the physical box."
function lhs_sample(N::Int, rng::AbstractRNG)
    d = length(LO)
    U01 = zeros(d, N)
    for k in 1:d
        perm = randperm(rng, N)
        for i in 1:N
            U01[k, i] = (perm[i] - rand(rng)) / N      # stratified within each bin
        end
    end
    return LO .+ (HI .- LO) .* U01
end

"Solve + sensitivities for one θ in a single AD pass. Y (nr,nz,6), dY (nr,nz,6,3)."
function sample_one(θ::Vector{Float64}, setup::WallLossSetup)
    nr, nz = setup.nr, setup.nz
    res = DiffResults.JacobianResult(zeros(nr * nz * NFIELDS_OUT), θ)
    ForwardDiff.jacobian!(res, t -> vec(run_case(t, setup)), θ)
    Y  = reshape(copy(DiffResults.value(res)),    nr, nz, NFIELDS_OUT)
    dY = reshape(copy(DiffResults.jacobian(res)), nr, nz, NFIELDS_OUT, length(θ))
    return Y, dY
end

function generate(N::Int, setup::WallLossSetup, rng::AbstractRNG)
    Θ  = lhs_sample(N, rng)
    nr, nz, nf, nin = setup.nr, setup.nz, NFIELDS_OUT, length(LO)
    Y  = Array{Float64}(undef, nr, nz, nf, N)
    dY = Array{Float64}(undef, nr, nz, nf, nin, N)
    done = Threads.Atomic{Int}(0)
    @threads for i in 1:N
        Yi, dYi = sample_one(Θ[:, i], setup)
        Y[:, :, :, i]    .= Yi
        dY[:, :, :, :, i] .= dYi
        n = Threads.atomic_add!(done, 1) + 1
        if n % 16 == 0 || n == N
            @info "generated $n / $N samples"
        end
    end
    return Θ, Y, dY
end

function main()
    N_train = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 256
    N_test  = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 64
    setup   = WallLossSetup()
    g       = Grid2D(setup.nr, setup.nz, setup.R, setup.L)

    @info "Generating dataset" N_train N_test nthreads() grid=(setup.nr, setup.nz)
    t0 = time()
    Θtr, Ytr, dYtr = generate(N_train, setup, MersenneTwister(1))
    Θte, Yte, dYte = generate(N_test,  setup, MersenneTwister(999))
    @info "done" elapsed_min=round((time()-t0)/60, digits=2)

    data = (; setup, input_names=INPUT_NAMES, field_names=OUT_FIELD_NAMES,
              lo=LO, hi=HI, r=g.r, z=g.z, R=g.R, L=g.L,
              theta_train=Θtr, Y_train=Ytr, dY_train=dYtr,
              theta_test=Θte,  Y_test=Yte,  dY_test=dYte)

    outdir = joinpath(@__DIR__, "data")
    mkpath(outdir)
    path = joinpath(outdir, "wallloss_dataset.jls")
    serialize(path, data)
    @info "wrote dataset" path size_train=size(Ytr) size_test=size(Yte)
end

main()
