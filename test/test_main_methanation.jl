module MethanationVerificationSuite
include("../main_methanation.jl")
end

using Test

@testset "methanation verification driver" begin
    result = MethanationVerificationSuite.run_methanation()
    @test result.allpass
    @test result.elem_err < 0.02
    @test result.mass_dev < 0.02
    @test result.en_err < 0.10
    @test result.X_CO > 0.05
    @test result.Tf_out > 595.0
end