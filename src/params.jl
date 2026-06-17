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
struct BoundaryConditions{T<:AbstractFloat}
    uz_in::T           # inlet axial velocity  [m/s]
    Tf_in::T           # inlet fluid temperature [K]
    Ts_in::T           # inlet solid temperature [K]
    Y_in::Vector{T}    # inlet mass fractions (length Ns)
end

BoundaryConditions(uz_in::T, Tf_in::T=T(300), Ts_in::T=T(300),
                   Y_in::Vector{T}=T[]) where T =
    BoundaryConditions{T}(uz_in, Tf_in, Ts_in, Y_in)

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
    bcs = BoundaryConditions(T(uz_in), T(Tf_in), T(Ts_in), convert(Vector{T}, Y_in))
    ReactorParams(grid, porous, fluid, solid, reactions,
                  T(beta_ac), T(gravity), D_species, bcs)
end

# NonlinearSolve's Jacobian operator needs to copy params when building the JVP.
# ReactorParams is logically immutable during a solve; copy only the mutable Vector.
function Base.copy(p::ReactorParams)
    ReactorParams(p.grid, p.porous, p.fluid, p.solid, p.reactions,
                  p.beta_ac, p.gravity, copy(p.D_species), p.bcs)
end
