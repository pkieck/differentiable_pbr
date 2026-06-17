using Test
include("../src/grid.jl")

@testset "Grid2D" begin

    g = Grid2D(10, 20, 1.0, 2.0)

    @testset "dimensions" begin
        @test g.nr == 10
        @test g.nz == 20
        @test g.dr ≈ 0.1
        @test g.dz ≈ 0.1
        @test length(g.r)  == 10
        @test length(g.z)  == 20
        @test length(g.rf) == 11   # nr+1 faces
        @test length(g.zf) == 21
    end

    @testset "cell centres" begin
        @test g.r[1] ≈ 0.05    # first cell: dr/2
        @test g.r[end] ≈ 0.95  # last cell: R - dr/2
        @test g.z[1] ≈ 0.05
    end

    @testset "face coordinates" begin
        @test g.rf[1] ≈ 0.0    # axis
        @test g.rf[end] ≈ 1.0  # wall
        @test g.zf[1] ≈ 0.0
        @test g.zf[end] ≈ 2.0
    end

    @testset "axisymmetric divergence identity" begin
        # For a constant flux F_r = F_z = 1, div(F) in FVM should give
        # (rf[i+1] - rf[i]) / (r[i]*dr) + 0 = dr / (r[i]*dr) = 1/r[i]
        # ... actually for a uniform F_r=c and F_z=c:
        # div = (rf[i+1]*c - rf[i]*c)/(r[i]*dr) + (c-c)/dz = c*dr/(r[i]*dr) = c/r[i]
        i = 5
        rf_m = g.rf[i]; rf_p = g.rf[i+1]; r_c = g.r[i]; dr = g.dr
        c = 2.0
        div_r = (rf_p * c - rf_m * c) / (r_c * dr)
        @test div_r ≈ c / r_c  rtol=1e-12
    end

    @testset "axis face is zero" begin
        # rf[1] = 0 so the axis flux contribution vanishes automatically
        @test g.rf[1] == 0.0
    end

end
