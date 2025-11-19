#!/usr/bin/env julia
# Test UniformScaling operations (A ± I)

using Test
using MPI
using SafePETSc
using LinearAlgebra
using LinearAlgebra: opnorm
using SparseArrays

SafePETSc.Init()
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "UniformScaling operations" begin
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
    begin
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
    begin
        B_petsc = I + A_petsc
        B_native = Matrix(B_petsc)
        B_expected = I + Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test A - I
    begin
        B_petsc = A_petsc - I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - I

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test I - A
    begin
        B_petsc = I - A_petsc
        B_native = Matrix(B_petsc)
        B_expected = I - Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
        @test size(B_petsc) == size(A_petsc)
    end

    # Test A + 2*I (scaled UniformScaling)
    begin
        B_petsc = A_petsc + 2I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) + 2I

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test A - 3*I (scaled UniformScaling)
    begin
        B_petsc = A_petsc - 3I
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - 3I

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test scalar operations (A + α and A - α)
    begin
        α = 2.5
        B_petsc = A_petsc + α
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) + α*I

        @test norm(B_native - B_expected) < 1e-10
    end

    begin
        α = 1.5
        B_petsc = A_petsc - α
        B_native = Matrix(B_petsc)
        B_expected = Matrix(A_sparse) - α*I

        @test norm(B_native - B_expected) < 1e-10
    end

    begin
        α = 10.0
        B_petsc = α - A_petsc
        B_native = Matrix(B_petsc)
        B_expected = α*I - Matrix(A_sparse)

        @test norm(B_native - B_expected) < 1e-10
    end

    # Test with MPIDENSE
    begin
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
    # begin
    #     # Create deterministic non-square matrix
    #     A_nonsquare = Float64[i+j for i in 1:5, j in 1:3]
    #     A_petsc_nonsquare = Mat_uniform(A_nonsquare; Prefix=SafePETSc.MPIDENSE)
    #
    #     @test_throws ErrorException A_petsc_nonsquare + I
    #     @test_throws ErrorException A_petsc_nonsquare - I
    #     @test_throws ErrorException A_petsc_nonsquare + 2.0
    #     @test_throws ErrorException A_petsc_nonsquare - 2.0
    # end
end  # End of UniformScaling operations testset

ts2 = @testset MPITestHarness.QuietTestSet "Matrix norm operations" begin
    # Create a deterministic sparse matrix
    n = 10
    A_sparse = spdiagm(
        -1 => fill(-1.0, n-1),
        0 => fill(4.0, n),
        1 => fill(-1.0, n-1)
    )
    A_petsc = Mat_uniform(A_sparse; Prefix=SafePETSc.MPIAIJ)

    # Test Frobenius norm (default, p=2)
    begin
        norm_petsc = norm(A_petsc)
        norm_native = norm(Matrix(A_sparse))

        @test abs(norm_petsc - norm_native) < 1e-10
    end

    # Test 1-norm (induced operator norm)
    begin
        norm_petsc = opnorm(A_petsc, 1)
        norm_native = opnorm(Matrix(A_sparse), 1)

        println(io0(), "opnorm 1: PETSc=$(norm_petsc), Native=$(norm_native), Diff=$(abs(norm_petsc - norm_native))")
        @test abs(norm_petsc - norm_native) < 1e-10
    end

    # Test Inf-norm (induced operator norm)
    begin
        norm_petsc = opnorm(A_petsc, Inf)
        norm_native = opnorm(Matrix(A_sparse), Inf)

        println(io0(), "opnorm Inf: PETSc=$(norm_petsc), Native=$(norm_native), Diff=$(abs(norm_petsc - norm_native))")
        @test abs(norm_petsc - norm_native) < 1e-10
    end

    # Test with dense matrix
    begin
        A_dense = Float64[i+j for i in 1:5, j in 1:5]
        A_petsc_dense = Mat_uniform(A_dense; Prefix=SafePETSc.MPIDENSE)

        norm_petsc = norm(A_petsc_dense)
        norm_native = norm(A_dense)

        @test abs(norm_petsc - norm_native) < 1e-10
    end

    # Test that unsupported norms fail
    begin
        @test_throws ErrorException norm(A_petsc, 3)
        @test_throws ErrorException norm(A_petsc, 0.5)
    end
end  # Matrix norm operations testset

# Aggregate results across ranks
local_counts = [ts.counts[:pass], ts.counts[:fail], ts.counts[:error], ts.counts[:broken], ts.counts[:skip]]
local_counts .+= [ts2.counts[:pass], ts2.counts[:fail], ts2.counts[:error], ts2.counts[:broken], ts2.counts[:skip]]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: UniformScaling and norm tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
