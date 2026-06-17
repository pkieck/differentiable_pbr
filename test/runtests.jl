"""
Top-level test runner.  Run from the project root with:
    julia --project test/runtests.jl
or via:
    julia --project -e "using Pkg; Pkg.test()"
"""

using Test

@testset "2dflow solver" begin
    include("test_grid.jl")
    include("test_fields.jl")
    include("test_porosity.jl")
    include("test_closures.jl")
    include("test_fluxes.jl")
    include("test_rhs.jl")
    include("test_chemkin.jl")
    include("test_main_root.jl")
    include("test_main_compressible.jl")
    include("test_main_methanation.jl")
end
