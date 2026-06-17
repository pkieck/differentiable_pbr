using Test
include("../src/fluxes.jl")

@testset "Fluxes" begin

    @testset "van_albada limiter" begin
        # φ(1) = 1 (perfectly smooth gradient)
        @test van_albada(1.0) ≈ 1.0
        # φ(0) = 0 (local extremum — no reconstruction)
        @test van_albada(0.0) ≈ 0.0
        # φ(-1) ≈ 0 (negative slope ratio — compressive, clamp to ≥ 0)
        @test van_albada(-1.0) ≈ 0.0  rtol=1e-14
        # TVD Sweby region: 0 ≤ φ(r) ≤ min(2r, 2) for r ≥ 0.
        # Van Albada CAN exceed 1 for r > 1 but stays well below 2.
        for r in [0.1, 0.5, 2.0, 10.0]
            v = van_albada(r)
            @test v ≥ 0
            @test v ≤ min(2*r + 1e-10, 2.0 + 1e-10)   # TVD upper bound
        end
    end

    @testset "adv_flux upwinding" begin
        # Positive u: flux should carry left value
        F_pos = adv_flux(1.0, 3.0, 5.0)
        @test F_pos ≈ 3.0  rtol=1e-4

        # Negative u: flux should carry right value
        F_neg = adv_flux(-1.0, 3.0, 5.0)
        @test F_neg ≈ -5.0  rtol=1e-4

        # u=0: flux is 0
        F_zero = adv_flux(0.0, 3.0, 5.0)
        @test abs(F_zero) < 1e-9
    end

    @testset "adv_flux smooth at u=0" begin
        # Should be finite and continuous at u=0 (no hard branch)
        F1 = adv_flux( 1e-12, 2.0, 4.0)
        F2 = adv_flux(-1e-12, 2.0, 4.0)
        @test isfinite(F1)
        @test isfinite(F2)
        # F(+ε) + F(-ε) should be O(SMOOTH_VEL) ≈ 2e-10 — just check smallness
        @test abs(F1 + F2) < 1e-8
    end

    @testset "diff_flux" begin
        # Central difference: D*(qR - qL)/dx
        @test diff_flux(0.1, 0.0, 1.0, 0.5) ≈ 0.2
        @test diff_flux(0.0, 0.0, 1.0, 0.5) ≈ 0.0
    end

    @testset "scalar_face_flux conservation" begin
        # Uniform field: advective flux is u*q (exact), diffusive is zero
        u = 2.0; q = 3.0; D = 0.1; dx = 0.01
        F = scalar_face_flux(u, D, q, q, dx)
        @test F ≈ u * q  rtol=1e-6
    end

    @testset "muscl_reconstruct reduces to first-order at extrema" begin
        # At a local extremum (flat slope on one side), phi→0 → first order
        q_L, q_R = muscl_reconstruct(1.0, 1.0, 2.0, 2.0)
        # Both sides flat: reconstructed values should equal cell centres
        @test q_L ≈ 1.0  rtol=1e-8
        @test q_R ≈ 2.0  rtol=1e-8
    end

end
