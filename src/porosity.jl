"""
Porous-medium descriptor. All fields are (nr x nz) matrices.
Porosity patterns (partial beds, inert zones, catalyst loading) are
encoded as scalar fields rather than geometry.

Setting eps to eps_min in solid zones drives the Ergun drag to infinity.
phi_cat controls catalyst activity independently (allows inert packing).
"""

const EPS_MIN = 1e-3   # floor on void fraction to keep 1/eps finite

struct PorousMedium{T<:AbstractFloat, M<:AbstractMatrix{T}}
    eps::M       # void fraction eps(r,z), clamped >= EPS_MIN
    dp::M        # particle diameter d_p(r,z) [m]
    phi_cat::M   # catalyst activity mask phi_cat(r,z) in [0,1]
    av::M        # interfacial area a_v(r,z) = 6(1-eps)/d_p [m2/m3]
end

function PorousMedium(eps::M, dp::M, phi_cat::M) where {T, M<:AbstractMatrix{T}}
    eps_c = max.(eps, T(EPS_MIN))
    av    = @. 6 * (1 - eps_c) / dp
    PorousMedium{T,M}(eps_c, dp, phi_cat, av)
end

# Convenience: uniform properties over the whole grid
function uniform_bed(nr::Int, nz::Int, eps::T, dp::T,
                     phi_cat::T=one(T)) where {T<:AbstractFloat}
    PorousMedium(fill(eps, nr, nz), fill(dp, nr, nz), fill(phi_cat, nr, nz))
end

# Open tube (no packing): eps=1, very large dp, no catalyst
function open_tube(nr::Int, nz::Int, T::Type{<:AbstractFloat}=Float64)
    PorousMedium(fill(one(T), nr, nz), fill(T(1.0), nr, nz), fill(zero(T), nr, nz))
end
