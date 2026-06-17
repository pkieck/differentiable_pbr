"""
Gaseous kinetics for packed-bed reactor.

Modified-Arrhenius rate: k(T) = A * T^beta * exp(-Ea/RT)
Independent forward reaction orders per species (CHEMKIN FORD).

The solver uses three accessors: rxn_rate, rxn_negdH, rxn_mass_coeff.
This design allows swapping kinetics backends without changing the PDE code.
"""

# Global reaction with modified-Arrhenius kinetics

"""
One global reaction: r = A * T^beta * exp(-Ea/RT) * prod(C_k^order_k)

Fields (SI units):
  A     : pre-exponential factor
  beta  : temperature exponent
  Ea    : activation energy [J/mol]
  nu    : stoichiometric coefficients (products +, reactants -)
  order : forward reaction order per species (0 = not involved)
  dH    : heat of reaction [J/mol] (negative = exothermic)
  M     : molar masses [kg/mol]
"""
struct GlobalReaction{T<:AbstractFloat, N}
    A::T
    beta::T
    Ea::T
    nu::NTuple{N,T}
    order::NTuple{N,T}
    dH::T
    M::NTuple{N,T}
end

# Unified accessors for PDE coupling

"""rxn_rate(rxn, Ts, rho, Y) -> r [mol/(m^3·s)]
Volumetric rate at solid temperature Ts, fluid density rho.
"""
@inline function rxn_rate(rxn::GlobalReaction{T,N}, Ts, ρ, Y::NTuple{N}) where {T,N}
    k = rxn.A * Ts^rxn.beta * exp(-rxn.Ea / (R_GAS * Ts))
    r = k
    @inbounds for l in 1:N
        o = rxn.order[l]
        if o != zero(o)                       # data branch (not on state) → AD-safe
            C = max(ρ * Y[l] / rxn.M[l], zero(Ts))   # mol/m³, clamped ≥ 0
            r *= C^o
        end
    end
    r
end

"Heat released by rxn per unit advancement, -dH [J/mol] (positive = exothermic)."
@inline rxn_negdH(rxn::GlobalReaction)      = -rxn.dH

"Mass source coefficient of species k: M_k * nu_k [kg/mol]."
@inline rxn_mass_coeff(rxn::GlobalReaction, k) = rxn.M[k] * rxn.nu[k]

# Legacy Reaction (params.jl) - old inline math for backward compatibility.
@inline function rxn_rate(rxn::Reaction{T,N}, Ts, rho, Y::NTuple{N}) where {T,N}
    C_args = ntuple(l -> rxn.reactant_idx[l] ? rho * Y[l] / rxn.M[l] : one(Y[l]), N)
    arrhenius_rate(rxn.A_pre, rxn.E_a, Ts, C_args...)
end
@inline rxn_negdH(rxn::Reaction)        = -rxn.dH
@inline rxn_mass_coeff(rxn::Reaction, k) = rxn.M[k] * rxn.nu[k]

# Mechanism bundle (output of CHEMKIN reader)

"""
A parsed mechanism: species, molar masses, and a tuple of GlobalReaction.
Tuple (not vector) keeps the per-cell reaction loop type-stable.

species[k] <-> M[k] <-> state species index k
"""
struct Mechanism{RT<:Tuple, T<:AbstractFloat}
    species::Vector{String}
    M::Vector{T}          # molar mass [kg/mol], length Ns
    reactions::RT         # tuple of GlobalReaction
end

nspecies(m::Mechanism)  = length(m.species)
nreactions(m::Mechanism) = length(m.reactions)

"Index of species `name` in the mechanism (errors if absent)."
function species_index(m::Mechanism, name::AbstractString)
    i = findfirst(==(uppercase(name)), uppercase.(m.species))
    i === nothing && error("species '$name' not in mechanism $(m.species)")
    i
end

"""
    inlet_mass_fractions(m, moles) → Vector

Convert a `name => mole_fraction` (or mole-count) spec into a normalised
mass-fraction vector in the mechanism's species order.  Convenience for
building `Y_in` boundary conditions from a feed composition.
"""
function inlet_mass_fractions(m::Mechanism{RT,T}, moles::AbstractDict) where {RT,T}
    x = zeros(T, nspecies(m))
    for (name, n) in moles
        x[species_index(m, name)] = T(n)
    end
    w = x .* m.M
    s = sum(w)
    s > 0 || error("inlet composition is empty")
    w ./ s
end

function Base.show(io::IO, m::Mechanism)
    print(io, "Mechanism(", nspecies(m), " species ", m.species,
          ", ", nreactions(m), " reactions)")
end

# ── NASA-7 thermodynamics (for heats of reaction & molar masses) ────────────

"""
A NASA-7 polynomial entry (two temperature ranges).  Used only at *parse time*
to derive each reaction's heat of reaction ΔH and each species' molar mass; the
PDE solver itself carries a single constant `cp` (low-Mach) and the precomputed
scalar `dH`, so the polynomials never enter the hot loop.
"""
struct Nasa7{T<:AbstractFloat}
    Tlow::T
    Tmid::T
    Thigh::T
    low::NTuple{7,T}      # coefficients valid on [Tlow, Tmid]
    high::NTuple{7,T}     # coefficients valid on [Tmid, Thigh]
    Mw::T                 # molar mass [kg/mol] from element composition
end

"""
    mechanism_params(mech, grid, porous, gas, solid, beta_ac, gravity;
                     D, uz_in, Tf_in, Ts_in, feed) → CompressibleParams

Build a `CompressibleParams` (compressible.jl) for the compressible solver from a
parsed `Mechanism`, wiring the species ordering through molar masses, diffusion
coefficients, and the inlet mass fractions (`feed` is a `name => mole_fraction`
dict).  `D` is a scalar diffusivity applied to all species (or a vector).
"""
function mechanism_params(mech::Mechanism, grid, porous, gas, solid,
                          beta_ac, gravity;
                          D=1e-5, uz_in=0.0, Tf_in=300.0, Ts_in=300.0,
                          feed::AbstractDict=Dict{String,Float64}())
    Ns  = nspecies(mech)
    Dsp = D isa AbstractVector ? collect(float.(D)) : fill(float(D), Ns)
    Y_in = inlet_mass_fractions(mech, feed)
    CompressibleParams(grid, porous, gas, solid, mech.reactions,
                       float(beta_ac), float(gravity), Dsp, copy(mech.M);
                       uz_in=uz_in, Tf_in=Tf_in, Ts_in=Ts_in, Y_in=Y_in)
end

"Dimensionless enthalpy H/(R·T) from a NASA-7 set at temperature `T`."
@inline function _h_RT(a::NTuple{7}, T)
    a[1] + a[2]/2*T + a[3]/3*T^2 + a[4]/4*T^3 + a[5]/5*T^4 + a[6]/T
end

"Molar enthalpy h(T) [J/mol] of a NASA-7 species."
function enthalpy(n::Nasa7, T)
    a = T < n.Tmid ? n.low : n.high
    R_GAS * T * _h_RT(a, T)
end
