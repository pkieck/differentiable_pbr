using Test
include("../src/closures.jl")

@testset "Closures" begin

    @testset "Ergun coefficients" begin
        eps = 0.4; dp = 3e-3
        A = ergun_A(eps, dp)
        B = ergun_B(eps, dp)
        K = darcy_K(eps, dp)
        # A = 150*(0.6)²/(0.4³*9e-6) ≈ 1.054e7  m⁻²
        @test A ≈ 150 * 0.36 / (0.064 * 9e-6)  rtol=1e-10
        # B = 1.75*0.6/(0.4³*3e-3) ≈ 5469  m⁻¹
        @test B ≈ 1.75 * 0.6 / (0.064 * 3e-3)  rtol=1e-10
        # K = 1/A
        @test K ≈ 1/A  rtol=1e-10
    end

    @testset "drag_coeff is smooth at u=0" begin
        # Should not return NaN or Inf at zero velocity
        D = drag_coeff(0.4, 3e-3, 1.0, 1e-3, 0.0, 0.0)
        @test isfinite(D)
        @test D > 0
    end

    @testset "drag_coeff scales linearly with mu at small u" begin
        eps=0.4; dp=3e-3; rho=1.0
        mu1 = 1e-3; mu2 = 2e-3
        # At u=0 the inertial term B*rho*|u| → 0, so D ≈ mu*A
        D1 = drag_coeff(eps, dp, rho, mu1, 0.0, 0.0)
        D2 = drag_coeff(eps, dp, rho, mu2, 0.0, 0.0)
        @test D2/D1 ≈ mu2/mu1  rtol=1e-6
    end

    @testset "h_fs Wakao–Funazkri" begin
        # At very low Re, Nu→2, so h_fs → 2*lam_f/dp
        eps=0.4; dp=1e-3; rho=1.0; mu=1e-5; cp=1000.0; lam=0.03
        # use tiny velocity
        h = h_fs(eps, dp, rho, mu, cp, lam, 1e-10, 0.0)
        @test h ≈ 2 * lam / dp  rtol=0.02   # small Re → Nu≈2
    end

    @testset "effective properties" begin
        eps = 0.4
        @test lam_f_eff(eps, 0.03) ≈ 0.4 * 0.03
        @test lam_s_eff(eps, 1.0)  ≈ 0.6 * 1.0
        @test D_eff(eps, 1e-5)     ≈ 0.4 * 1e-5 / 1.5
    end

    @testset "Arrhenius rate" begin
        # At E_a=0, rate = A_pre * C (first order)
        r = arrhenius_rate(1e6, 0.0, 500.0, 1.0)
        @test r ≈ 1e6

        # Zero concentration clamps to zero (reactant absent)
        r0 = arrhenius_rate(1e6, 1e4, 500.0, 0.0)
        @test r0 == 0.0

        # Non-reactant passed as 1.0 (doesn't kill the rate)
        # This is how rhs! passes non-reactant species
        r1 = arrhenius_rate(2.0, 0.0, 500.0, 5.0, 1.0)
        @test r1 ≈ 10.0   # 2.0 * 5.0 * 1.0 = 10.0
    end

end
