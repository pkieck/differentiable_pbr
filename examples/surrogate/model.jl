"""
model.jl — DiT/Perceiver-IO surrogate for the wall-heatloss reactor (Lux.jl).

Maps the 3 scalar inputs θ = [T_in, U, m] (normalised) to the 6 steady fields on
the (nr, nz) grid.  Architecture (as requested):

  ┌ encoder ──────────────────────────────────────────────────────────────┐
  │ build a (nr, nz, Cin) "image": per-cell coordinates [r,z] + θ broadcast │
  │ → ViT-style patch-embed Conv(patch) → grid of patch tokens + pos-embed  │
  └────────────────────────────────────────────────────────────────────────┘
  ┌ conditioning ─────────────────────────────────────────────────────────┐
  │ c = MLP(θ)  → drives adaLN-Zero (DiT) modulation in every block         │
  └────────────────────────────────────────────────────────────────────────┘
  ┌ processor ── N_proc DiT blocks (self-attention + MLP, adaLN-modulated) ─┐
  ┌ decoder ──── N_dec Perceiver-IO blocks: full-res query tokens cross-    │
  │              attend the latent (adaLN-modulated) → adaLN head → 6 chans │
  └────────────────────────────────────────────────────────────────────────┘

Conditioning enters twice (input channels of the patch-embed AND adaLN), which
is standard for conditional ViT/DiT.  adaLN-Zero: the modulation projection is
zero-initialised so each block starts at identity (stable training).

All tensors are feature-first, batch-last: tokens are (d_model, seq, batch); the
field output is reshaped to (nr, nz, 6, batch).
"""

using Lux, Random
import NNlib
using NNlib: swish, gelu

# ── small array helpers (Zygote-friendly: pure broadcasting / reshape) ───────
"insert a singleton sequence axis: (d, B) → (d, 1, B)"
unsq(v) = reshape(v, size(v, 1), 1, size(v, 2))
"split a (k*d, B) modulation vector into k pieces of (d, B)"
chunk(x, d, k) = ntuple(i -> x[(i-1)*d+1:i*d, :], k)
"adaLN affine modulation of token array h (d,L,B) by per-sample shift/scale (d,B)"
modulate(h, shift, scale) = h .* (1 .+ unsq(scale)) .+ unsq(shift)

# ── attention primitives ─────────────────────────────────────────────────────
function SelfAttention(d::Int, nheads::Int)
    nh = nheads
    Lux.@compact(qkv = Dense(d => 3d), proj = Dense(d => d)) do x
        dd = size(x, 1)
        h = qkv(x)
        q = h[1:dd, :, :]; k = h[dd+1:2dd, :, :]; v = h[2dd+1:3dd, :, :]
        y, _ = NNlib.dot_product_attention(q, k, v; nheads = nh)
        @return proj(y)
    end
end

function CrossAttention(d::Int, nheads::Int)
    nh = nheads
    Lux.@compact(wq = Dense(d => d), wkv = Dense(d => 2d), proj = Dense(d => d)) do inputs
        q_in, ctx = inputs
        dd = size(q_in, 1)
        q = wq(q_in)
        kv = wkv(ctx); k = kv[1:dd, :, :]; v = kv[dd+1:2dd, :, :]
        y, _ = NNlib.dot_product_attention(q, k, v; nheads = nh)
        @return proj(y)
    end
end

mlp_block(d, ratio) = Chain(Dense(d => ratio * d, gelu), Dense(ratio * d => d))

# ── DiT block (self-attention processor) — input/output carry (x, c) ─────────
function DiTBlock(d::Int, nheads::Int; ratio::Int = 4)
    dd = d
    Lux.@compact(n1 = LayerNorm((d,); affine = false), attn = SelfAttention(d, nheads),
                 n2 = LayerNorm((d,); affine = false), mlp = mlp_block(d, ratio),
                 mod = Dense(d => 6d; init_weight = zeros32, init_bias = zeros32)) do inputs
        x, c = inputs
        sh1, sc1, g1, sh2, sc2, g2 = chunk(mod(swish.(c)), dd, 6)
        x = x .+ unsq(g1) .* attn(modulate(n1(x), sh1, sc1))
        x = x .+ unsq(g2) .* mlp(modulate(n2(x), sh2, sc2))
        @return (x, c)
    end
end

# ── Perceiver-IO decoder block — queries cross-attend latent; carry (q,lat,c) ─
function DecoderBlock(d::Int, nheads::Int; ratio::Int = 4)
    dd = d
    Lux.@compact(nq = LayerNorm((d,); affine = false), cross = CrossAttention(d, nheads),
                 n2 = LayerNorm((d,); affine = false), mlp = mlp_block(d, ratio),
                 mod = Dense(d => 6d; init_weight = zeros32, init_bias = zeros32)) do inputs
        q, lat, c = inputs
        sh1, sc1, g1, sh2, sc2, g2 = chunk(mod(swish.(c)), dd, 6)
        q = q .+ unsq(g1) .* cross((modulate(nq(q), sh1, sc1), lat))
        q = q .+ unsq(g2) .* mlp(modulate(n2(q), sh2, sc2))
        @return (q, lat, c)
    end
end

"""
    build_surrogate(; nr, nz, n_in, n_out, d_model, nheads, patch, n_proc, n_dec)

Returns a Lux layer mapping θ::(n_in, B) → fields::(nr, nz, n_out, B).
Per-cell coordinates are baked in as fixed (non-trainable) constants.
"""
function build_surrogate(; nr = 8, nz = 40, n_in = 3, n_out = 6,
                         d_model = 64, nheads = 4, patch = (2, 4),
                         n_proc = 4, n_dec = 2, ratio = 4)
    pr, pz = patch
    @assert nr % pr == 0 && nz % pz == 0 "patch must tile the grid"
    ntok  = (nr ÷ pr) * (nz ÷ pz)
    ncell = nr * nz
    Cin   = 2 + n_in                         # [r, z] + broadcast θ

    # fixed normalised coordinate image (nr, nz, 2, 1) — captured as a constant
    rcoord = Float32[(i - 0.5f0) / nr for i in 1:nr, j in 1:nz]
    zcoord = Float32[(j - 0.5f0) / nz for i in 1:nr, j in 1:nz]
    coords = reshape(cat(rcoord, zcoord; dims = 3), nr, nz, 2, 1)  # const

    proc = Chain([DiTBlock(d_model, nheads; ratio) for _ in 1:n_proc]...)
    dec  = Chain([DecoderBlock(d_model, nheads; ratio) for _ in 1:n_dec]...)
    d = d_model                              # captured constants (not parameters):
    _nr, _nz, _nout = nr, nz, n_out          #   coords, d, _nr, _nz, _nout

    Lux.@compact(patch_embed = Conv(patch, Cin => d_model; stride = patch),
                 cond = Chain(Dense(n_in => d_model, swish), Dense(d_model => d_model, swish)),
                 pos_tok = randn32(Random.default_rng(), d_model, ntok) .* 0.02f0,
                 q_pos   = randn32(Random.default_rng(), d_model, ncell) .* 0.02f0,
                 processor = proc, decoder = dec,
                 nf = LayerNorm((d_model,); affine = false),
                 modf = Dense(d_model => 2d_model; init_weight = zeros32, init_bias = zeros32),
                 head = Dense(d_model => n_out)) do θ
        B = size(θ, 2)
        nr, nz, n_out = _nr, _nz, _nout
        seqconst(v) = reshape(v, size(v, 1), size(v, 2), 1)        # (d,L) → (d,L,1)
        # ── build input image (nr, nz, Cin, B): coords (const) + θ broadcast ──
        coordsB = repeat(eltype(θ).(coords), 1, 1, 1, B)            # (nr,nz,2,B)
        θimg = repeat(reshape(θ, 1, 1, size(θ, 1), B), nr, nz, 1, 1) # (nr,nz,n_in,B)
        img  = cat(coordsB, θimg; dims = 3)                         # (nr,nz,Cin,B)

        # ── ViT patch embedding → tokens (d, ntok, B) ──
        pe = patch_embed(img)                                      # (nr/pr, nz/pz, d, B)
        wc, hc = size(pe, 1), size(pe, 2)
        tok = reshape(permutedims(pe, (3, 1, 2, 4)), size(pe, 3), wc * hc, B)
        tok = tok .+ seqconst(pos_tok)                             # + positional

        c = cond(θ)                                                # (d, B)
        lat, _ = processor((tok, c))                               # (d, ntok, B)

        # ── full-res query tokens conditioned on c, cross-attend the latent ──
        q = seqconst(q_pos) .+ reshape(c, size(c, 1), 1, B)        # (d, ncell, B)
        qo, _, _ = decoder((q, lat, c))                            # (d, ncell, B)

        shf, scf = chunk(modf(swish.(c)), d, 2)
        y = head(modulate(nf(qo), shf, scf))                       # (n_out, ncell, B)
        # (n_out, ncell, B) → (nr, nz, n_out, B); ncell laid out row-major (i fast)
        y = permutedims(reshape(y, n_out, nr, nz, B), (2, 3, 1, 4))
        @return y
    end
end
