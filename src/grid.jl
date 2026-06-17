"""
Uniform 2D axisymmetric (r,z) finite-volume grid.

Cells are indexed (i,j), i=1..nr in r, j=1..nz in z.
Face arrays use 1-based indexing: rf[1]=0 (axis), rf[nr+1]=R (wall).

Axisymmetric FVM divergence formula for a cell-averaged scalar div(F):
  div(F)[i,j] = (rf[i+1]*F_r[i+1,j] - rf[i]*F_r[i,j]) / (r[i]*dr)
              + (F_z[i,j+1] - F_z[i,j])                 / dz
where F_r[i,j] is the radial flux at the left face of cell i (r=rf[i]),
and F_z[i,j] is the axial flux at the bottom face of cell j (z=zf[j]).
At the axis rf[1]=0 so the axis term vanishes automatically.
"""
struct Grid2D{T<:AbstractFloat}
    nr::Int
    nz::Int
    R::T
    L::T
    dr::T
    dz::T
    r::Vector{T}   # cell-centre r, length nr
    z::Vector{T}   # cell-centre z, length nz
    rf::Vector{T}  # r face coords 0..R, length nr+1
    zf::Vector{T}  # z face coords 0..L, length nz+1
end

function Grid2D(nr::Int, nz::Int, R::T, L::T) where {T<:AbstractFloat}
    dr = R / nr
    dz = L / nz
    r  = T[(i - T(0.5)) * dr for i in 1:nr]
    z  = T[(j - T(0.5)) * dz for j in 1:nz]
    rf = T[i * dr for i in 0:nr]
    zf = T[j * dz for j in 0:nz]
    Grid2D{T}(nr, nz, R, L, dr, dz, r, z, rf, zf)
end
