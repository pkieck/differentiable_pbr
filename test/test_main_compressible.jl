module CompressibleVerificationSuite
include("../main_compressible.jl")
end

using Test

@testset "compressible verification driver" begin
    results = CompressibleVerificationSuite.main()
    @test results.reduction < 1e-10
    @test results.isothermal_poiseuille < 0.05
    @test results.thermal_expansion < 0.05
end