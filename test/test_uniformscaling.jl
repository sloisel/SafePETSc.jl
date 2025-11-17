#!/usr/bin/env julia
# Test UniformScaling operations (A ± I)

using MPI
using SafePETSc
using LinearAlgebra
using SparseArrays
using Test

SafePETSc.Init()

@testset "UniformScaling operations" begin
    # Create a simple sparse matrix (deterministic, same on all ranks)
    n = 10
    # Create a tridiagonal matrix (deterministic)
    A_sparse = spdiagm(
        -1 => fill(-1.0, n-1),
        0 => fill(6.0, n),
        1 => fill(-1.0, n-1)
    )

    # Convert to PETSc
    A_petsc = Mat_uniform(A_sparse; Prefix=SafePETSc.MPIAIJ)

    # Test A + I
    @testset "A + I" begin
        B_petsc = A_petsc + I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) + I

        println(io0(), "A_sparse diagonal: ", diag(Matrix(A_sparse))[1:5])
        println(io0(), "B_expected diagonal: ", diag(B_expected)[1:5])
        println(io0(), "B_native diagonal: ", diag(B_native)[1:5])
        println(io0(), "Difference norm: ", norm(B_native - B_expected))

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test I + A
    @testset "I + A" begin
        B_petsc = I + A_petsc
        B_native = Matrix(B_petsc)
        B_expected = I + Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test A - I
    @testset "A - I" begin
        B_petsc = A_petsc - I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - I

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test I - A
    @testset "I - A" begin
        B_petsc = I - A_petsc
        B_native = Matrix(B_petsc)
        B_expected = I - Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test A + 2*I (scaled UniformScaling)
    @testset "A + 2I" begin
        B_petsc = A_petsc + 2I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) + 2I

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test A - 3*I (scaled UniformScaling)
    @testset "A - 3I" begin
        B_petsc = A_petsc - 3I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - 3I

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test scalar operations (A + α and A - α)
    @testset "Scalar addition A + α" begin
        α = 2.5
        B_petsc = A_petsc + α
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) + α*I

        @test norm(B_native - B_expected) < 1e-10
    end

    @testset "Scalar subtraction A - α" begin
        α = 1.5
        B_petsc = A_petsc - α
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - α*I

        @test norm(B_native - B_expected) < 1e-10
    end

    @testset "Scalar subtraction α - A" begin
        α = 10.0
        B_petsc = α - A_petsc
        B_native = Matrix(B_petsc)
        B_expected = α*I - Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test with MPIDENSE
    @testset "Dense matrices with UniformScaling" begin
        # Create deterministic dense matrix
        A_dense = Float64[i+j for i in 1:5, j in 1:5]
        A_dense = A_dense + A_dense'  # Symmetric
        A_petsc_dense = Mat_uniform(A_dense; Prefix=SafePETSc.MPIDENSE)

        B_petsc = A_petsc_dense - I
        B_native = Matrix(B_petsc)
        B_expected = A_dense - I

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test that non-square matrices fail appropriately
    # TODO: Re-enable after fixing MPI abort handling in tests
    # @testset "Non-square matrix error" begin
    #     # Create deterministic non-square matrix
    #     A_nonsquare = Float64[i+j for i in 1:5, j in 1:3]
    #     A_petsc_nonsquare = Mat_uniform(A_nonsquare; Prefix=SafePETSc.MPIDENSE)
    #
    #     @test_throws ErrorException A_petsc_nonsquare + I
    #     @test_throws ErrorException A_petsc_nonsquare - I
    #     @test_throws ErrorException A_petsc_nonsquare + 2.0
    #     @test_throws ErrorException A_petsc_nonsquare - 2.0
    # end
end

println(io0(), "UniformScaling tests passed!")
