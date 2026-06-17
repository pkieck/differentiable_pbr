using Test
include("../src/fields.jl")

@testset "StateLayout" begin

    @testset "length" begin
        s = StateLayout(10, 20, 3)
        @test length(s) == 10 * 20 * (5 + 3)
    end

    @testset "field_mat indexing" begin
        s  = StateLayout(4, 6, 2)
        u  = collect(Float64, 1:length(s))
        q  = field_mat(u, s, 1)
        @test size(q) == (4, 6)
        # First element of field 1 maps to u[1]
        @test q[1,1] == 1.0
        # First element of field 2 maps to u[4*6+1]
        q2 = field_mat(u, s, 2)
        @test q2[1,1] == Float64(4*6 + 1)
    end

    @testset "species_mat" begin
        s = StateLayout(3, 5, 2)
        u = zeros(length(s))
        Y1 = species_mat(u, s, 1)
        Y2 = species_mat(u, s, 2)
        @test size(Y1) == (3, 5)
        @test size(Y2) == (3, 5)
        # Y1 starts after fields 1..5
        Y1[1,1] = 42.0
        @test u[5*3*5 + 1] == 42.0
    end

    @testset "set_field!" begin
        s = StateLayout(4, 5, 0)
        u = zeros(length(s))
        set_field!(u, s, F_TF, 300.0)
        q = field_mat(u, s, F_TF)
        @test all(q .== 300.0)
        # Other fields untouched
        @test all(field_mat(u, s, F_UR) .== 0.0)
    end

    @testset "zero_state" begin
        s = StateLayout(5, 7, 1)
        u = zero_state(s)
        @test length(u) == length(s)
        @test all(u .== 0.0)
    end

end
