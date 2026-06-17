using Test
include("../src/porosity.jl")

@testset "PorousMedium" begin
    @testset "uniform_bed basic" begin
        pm = uniform_bed(4, 6, 0.4, 3e-3)
        @test size(pm.eps)     == (4, 6)
        @test size(pm.dp)      == (4, 6)
        @test size(pm.phi_cat) == (4, 6)
        @test size(pm.av)      == (4, 6)
        @test all(pm.eps     .≈ 0.4)
        @test all(pm.dp      .≈ 3e-3)
        @test all(pm.phi_cat .≈ 1.0)
        # a_v = 6*(1-ε)/d_p
        expected_av = 6 * (1 - 0.4) / 3e-3
        @test all(pm.av .≈ expected_av)
    end

    @testset "eps floor" begin
        # eps below EPS_MIN should be clamped
        pm = uniform_bed(2, 2, 0.0, 1e-3)
        @test all(pm.eps .≥ EPS_MIN)
    end

    @testset "open_tube" begin
        pm = open_tube(3, 5)
        @test all(pm.eps     .≈ 1.0)
        @test all(pm.phi_cat .≈ 0.0)
        # a_v = 6*(1-1)/dp = 0
        @test all(pm.av .≈ 0.0)
    end
end
