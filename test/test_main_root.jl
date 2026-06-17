module RootVerificationSuite
include("../main.jl")
end

using Test

@testset "root verification driver" begin
    results = RootVerificationSuite.main()
    @test results.poiseuille < 0.05
    @test results.ergun_dp < 0.10
    @test results.plug_flow < 0.05
    @test results.two_T_heatup < 0.02
    @test results.nonisothermal_5a < 0.15
    @test results.nonisothermal_5b < 0.15
    @test results.porosity_masking < 0.10
    @test results.gradient_sanity < 0.05
end