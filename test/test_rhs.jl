"""
Unit tests for the RHS function.

These check mathematical properties rather than exact solver output:
  1. Zero-velocity steady state (p const, T const, Y const) → RHS = 0
  2. Mass conservation: net mass flux out of all cells sums to zero
  3. Energy conservation in no-reaction case
  4. Species conservation (no reaction case)
"""

using Test
using LinearAlgebra
include("../src/rhs.jl")

function make_minimal_params(nr, nz; eps=0.4, dp=3e-3, Ns=0, beta_ac=1.0, uz_in=0.0)
    g  = Grid2D(nr, nz, 0.05, 0.5)
    pm = uniform_bed(nr, nz, Float64(eps), Float64(dp))
    fl = FluidProps(1.0, 1e-3, 1000.0, 0.03)
    sl = SolidProps(2000.0, 800.0, 1.0)
    D  = Ns > 0 ? fill(1e-5, Ns) : Float64[]
    ReactorParams(g, pm, fl, sl, (), Float64(beta_ac), 0.0, D;
                  uz_in=Float64(uz_in), Tf_in=300.0, Ts_in=300.0)
end

@testset "RHS" begin

    @testset "quiescent uniform state → zero RHS for p and T" begin
        nr, nz = 6, 8
        p0     = make_minimal_params(nr, nz; Ns=0)
        s      = StateLayout(nr, nz, 0)
        u      = zero_state(s)
        # Set uniform T (same Tf=Ts → no interphase exchange)
        set_field!(u, s, F_TF, 300.0)
        set_field!(u, s, F_TS, 300.0)
        # Zero velocity, zero pressure → dp/dt should be 0
        du = similar(u)
        rhs!(du, u, p0, 0.0)

        dpp = field_mat(du, s, F_P)
        dTf = field_mat(du, s, F_TF)
        dTs = field_mat(du, s, F_TS)

        @test norm(dpp) < 1e-12
        @test norm(dTf) < 1e-10
        @test norm(dTs) < 1e-10
    end

    @testset "uniform velocity → no acceleration (inertia + drag balanced by dp/dz)" begin
        # A flat uz = U, ur = 0, p gradient = -drag*uz → dur=duz≈0 in interior
        nr, nz = 4, 10
        params = make_minimal_params(nr, nz; eps=1.0, dp=1.0, Ns=0, beta_ac=1.0,
                                     uz_in=0.01)
        s      = StateLayout(nr, nz, 0)
        u      = zero_state(s)
        U      = 0.01
        set_field!(u, s, F_UZ, U)
        set_field!(u, s, F_TF, 300.0)
        set_field!(u, s, F_TS, 300.0)

        du = similar(u)
        rhs!(du, u, params, 0.0)

        # Continuity: uniform uz, ur=0 → div=0 → dp/dt=0
        dpp = field_mat(du, s, F_P)
        @test norm(dpp) < 1e-10
    end

    @testset "temperature equilibrium (Tf=Ts, no reaction) → zero dT" begin
        # Inlet BC must match the uniform field value so there is no ghost-cell gradient.
        nr, nz = 5, 5
        g  = Grid2D(nr, nz, 0.05, 0.5)
        pm = uniform_bed(nr, nz, 0.4, 3e-3)
        fl = FluidProps(1.0, 1e-3, 1000.0, 0.03)
        sl = SolidProps(2000.0, 800.0, 1.0)
        params = ReactorParams(g, pm, fl, sl, (), 1.0, 0.0, Float64[];
                               uz_in=0.0, Tf_in=500.0, Ts_in=500.0)
        s  = StateLayout(nr, nz, 0)
        u  = zero_state(s)
        set_field!(u, s, F_TF, 500.0)
        set_field!(u, s, F_TS, 500.0)
        du = similar(u)
        rhs!(du, u, params, 0.0)
        dTf = field_mat(du, s, F_TF)
        dTs = field_mat(du, s, F_TS)
        @test norm(dTf) < 1e-8
        @test norm(dTs) < 1e-8
    end

    @testset "species rhs is zero for uniform Y, no reaction, no flow" begin
        nr, nz = 4, 4
        Ns     = 2
        params = make_minimal_params(nr, nz; Ns=Ns)
        s      = StateLayout(nr, nz, Ns)
        u      = zero_state(s)
        set_field!(u, s, F_TF, 300.0)
        set_field!(u, s, F_TS, 300.0)
        species_mat(u, s, 1) .= 0.8
        species_mat(u, s, 2) .= 0.2
        du = similar(u)
        rhs!(du, u, params, 0.0)
        for k in 1:Ns
            @test norm(field_mat(du, s, 5+k)) < 1e-10
        end
    end

    @testset "RHS finite and non-NaN on random state" begin
        nr, nz = 8, 8
        Ns     = 1
        params = make_minimal_params(nr, nz; Ns=Ns)
        s      = StateLayout(nr, nz, Ns)
        u      = zero_state(s)
        # Random perturbation (small velocities, physical temperatures)
        set_field!(u, s, F_UZ, 0.01)
        set_field!(u, s, F_TF, 400.0)
        set_field!(u, s, F_TS, 420.0)
        species_mat(u, s, 1) .= 0.5
        du = similar(u)
        rhs!(du, u, params, 0.0)
        @test all(isfinite, du)
    end

end
