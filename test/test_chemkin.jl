using Test
# kinetics.jl needs R_GAS / arrhenius_rate (closures) and the legacy Reaction
# struct (params, which needs the grid type) defined first.
include("../src/grid.jl")
include("../src/fields.jl")
include("../src/closures.jl")
include("../src/params.jl")
include("../src/chemkin.jl")

@testset "CHEMKIN reader + kinetics" begin
    mech = read_chemkin(joinpath(@__DIR__, "..", "data", "methanation.inp"))

    @testset "species & molar masses" begin
        @test mech.species == ["CO", "CO2", "H2", "CH4", "H2O", "N2"]
        # molar masses from element composition in the THERMO cards [kg/mol]
        Mref = Dict("CO"=>0.02801, "CO2"=>0.044009, "H2"=>0.002016,
                    "CH4"=>0.016043, "H2O"=>0.018015, "N2"=>0.028014)
        for (k, s) in enumerate(mech.species)
            @test mech.M[k] ≈ Mref[s]  rtol=1e-3
        end
    end

    @testset "two reactions parsed (scope-safe loop)" begin
        @test nreactions(mech) == 2          # regression: closure must not clobber loop index
    end

    @testset "heats of reaction from NASA-7 thermo" begin
        # CO  + 3H2 -> CH4 + H2O   : ΔH298 ≈ -206 kJ/mol
        # CO2 + 4H2 -> CH4 + 2H2O  : ΔH298 ≈ -165 kJ/mol
        @test mech.reactions[1].dH/1e3 ≈ -206  atol=3
        @test mech.reactions[2].dH/1e3 ≈ -165  atol=3
    end

    @testset "stoichiometry, orders & unit conversion" begin
        r1 = mech.reactions[1]
        iCO, iH2, iCH4, iH2O = (species_index(mech, n) for n in ("CO","H2","CH4","H2O"))
        @test r1.nu[iCO]  == -1 && r1.nu[iH2] == -3
        @test r1.nu[iCH4] ==  1 && r1.nu[iH2O] == 1
        @test r1.order[iCO] == 1.0 && r1.order[iH2] == 0.0   # FORD overrides
        # JOULES/MOLE: Ea passes through unchanged
        @test r1.Ea ≈ 8.0e4
        # MOLES (cm-mol-s) with overall order 1 ⇒ A unchanged (1/s)
        @test r1.A ≈ 8.0e6  rtol=1e-12
    end

    @testset "rate evaluation is finite & smooth" begin
        Ns = nspecies(mech)
        Y = inlet_mass_fractions(mech, Dict("CO"=>0.04, "H2"=>0.12, "N2"=>0.84))
        Yt = ntuple(k -> Y[k], Ns)
        r = rxn_rate(mech.reactions[1], 600.0, 2.5, Yt)
        @test isfinite(r) && r > 0
        # zero CO ⇒ zero forward rate (order 1 in CO)
        Y0 = ntuple(k -> mech.species[k] == "CO" ? 0.0 : Yt[k], Ns)
        @test rxn_rate(mech.reactions[1], 600.0, 2.5, Y0) == 0.0
    end
end
