"""
State vector layout for the packed-bed solver.

Field order (1-indexed):
  1  : u_r  (radial velocity)
  2  : u_z  (axial velocity)
  3  : p    (pressure)
  4  : T_f  (fluid temperature)
  5  : T_s  (solid temperature)
  6..(5+Ns): Y_k (species mass fractions)

All fields are stored as column-major (nr x nz) matrices flattened into
one contiguous vector of length nr*nz*(5+Ns).  The layout is field-major:
  field f lives at indices (f-1)*nr*nz+1 : f*nr*nz.
"""

const F_UR = 1
const F_UZ = 2
const F_P  = 3
const F_TF = 4
const F_TS = 5

struct StateLayout
    nr::Int
    nz::Int
    Ns::Int   # number of species
end

Base.length(s::StateLayout) = s.nr * s.nz * (5 + s.Ns)
nfields(s::StateLayout)     = 5 + s.Ns

# Return a reshaped view of field f (1-indexed) inside flat vector u.
@inline function field_mat(u::AbstractVector, s::StateLayout, f::Int)
    n = s.nr * s.nz
    reshape(view(u, (f-1)*n+1 : f*n), s.nr, s.nz)
end

@inline species_mat(u, s, k) = field_mat(u, s, 5 + k)

# Allocate a zero state vector of type T.
zero_state(s::StateLayout, T::Type{<:AbstractFloat}=Float64) = zeros(T, length(s))

# Set a constant value for field f across all cells.
function set_field!(u::AbstractVector, s::StateLayout, f::Int, val)
    q = field_mat(u, s, f)
    q .= val
end
