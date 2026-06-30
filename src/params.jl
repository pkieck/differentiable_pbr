"""
Physical property and parameter structs.
All plain, immutable, concrete types for Enzyme compatibility.
"""

struct FluidProps{T<:AbstractFloat}
    rho::T       # density [kg/m3]
    mu::T        # dynamic viscosity [Pa*s]
    cp::T        # heat capacity [J/(kg*K)]
    lambda::T    # thermal conductivity [W/(m*K)]
end

struct SolidProps{T<:AbstractFloat}
    rho::T       # bulk density [kg/m3]
    cp::T        # heat capacity [J/(kg*K)]
    lambda::T    # thermal conductivity [W/(m*K)]
end

"""
Boundary conditions applied as ghost-cell Dirichlet/Neumann values.

Conventions:
- Inlet (z=0): Dirichlet on uz, T_f, T_s, Y_k; Neumann on p.
- Outlet (z=L): Dirichlet on p=0 (gauge); Neumann on velocity.
- Axis (r=0): Symmetry ur=0, Neumann on others.
- Wall (r=R): No-slip ur=uz=0; Neumann on T, Y.
"""
struct BoundaryConditions{T<:Real, U}
    uz_in::U           # inlet axial velocity [m/s]: scalar OR length-nr radial profile
    Tf_in::T           # inlet fluid temperature [K]
    Ts_in::T           # inlet solid temperature [K]
    Y_in::Vector{T}    # inlet mass fractions (length Ns)
end

function BoundaryConditions(uz_in::U, Tf_in::Real=300.0, Ts_in::Real=300.0,
                            Y_in::AbstractVector=Float64[]) where {U}
    T = promote_type(eltype(float.(uz_in)), typeof(float(Tf_in)),
                     typeof(float(Ts_in)), eltype([float.(Y_in); 0.0]))
    BoundaryConditions{T, U}(uz_in, T(Tf_in), T(Ts_in), convert(Vector{T}, Y_in))
end

# When uz_in is a length-nr radial profile use the value at row i; when it is a
# scalar return it unchanged.  Lets a velocity *profile* be imposed at the inlet
# without changing the scalar fast path (bc_at(::Number) is identity → existing
# verified cases stay bit-for-bit identical).
@inline bc_at(x::Number, i::Integer)        = x
@inline bc_at(x::AbstractVector, i::Integer) = @inbounds x[i]

"""
A single Arrhenius reaction:
  r = A_pre * exp(-E_a/(R_gas*T)) * prod(C_k^nu_k for reactants k)
  [mol/(m3_bed*s)]

Fields:
  A_pre       : pre-exponential [mol/(m3*s) per (mol/m3)^order]
  E_a         : activation energy [J/mol]
  nu          : stoichiometric coefficients (negative = consumed)
  dH          : heat of reaction [J/mol] (negative = exothermic)
  M           : molar masses [kg/mol]
  reactant_idx: true for species that appear as concentration factors
"""
struct Reaction{T<:AbstractFloat, N}
    A_pre::T
    E_a::T
    nu::NTuple{N,T}
    dH::T
    M::NTuple{N,T}
    reactant_idx::NTuple{N,Bool}
end

"""
Top-level parameter bundle passed to rhs!.
"""
struct ReactorParams{T, PM, F<:FluidProps, S<:SolidProps, RC, BC<:BoundaryConditions}
    grid::Grid2D{T}
    porous::PM
    fluid::F
    solid::S
    reactions::RC
    beta_ac::T
    gravity::T
    D_species::Vector{T}
    bcs::BC
end

# Convenience constructor with default BCs
function ReactorParams(grid, porous, fluid::FluidProps{T}, solid, reactions,
                       beta_ac, gravity, D_species;
                       uz_in=zero(T), Tf_in=T(300), Ts_in=T(300),
                       Y_in=T[]) where T
    # uz_in may be a scalar OR a length-nr radial profile vector; coerce element
    # type but keep its container shape.
    uz_bc = uz_in isa AbstractVector ? convert(Vector, float.(uz_in)) : float(uz_in)
    bcs = BoundaryConditions(uz_bc, Tf_in, Ts_in, convert(Vector, float.(Y_in)))
    ReactorParams(grid, porous, fluid, solid, reactions,
                  T(beta_ac), T(gravity), D_species, bcs)
end

# NonlinearSolve's Jacobian operator needs to copy params when building the JVP.
# ReactorParams is logically immutable during a solve; copy only the mutable Vector.
function Base.copy(p::ReactorParams)
    ReactorParams(p.grid, p.porous, p.fluid, p.solid, p.reactions,
                  p.beta_ac, p.gravity, copy(p.D_species), p.bcs)
end
