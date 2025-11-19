#!/usr/bin/env julia

# Test map_rows and eachrow for sparse (MPIAIJ) matrices

using Test
using MPI
using SafePETSc
using SafePETSc: MPIAIJ, MPIDENSE
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

MPI.Initialized() || MPI.Init()
SafePETSc.Initialized() || SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset MPITestHarness.QuietTestSet "map_rows sparse tests" begin

@testset "eachrow for sparse matrices" begin
    # Create a sparse matrix (same on all ranks)
    n = 20
    using Random
    Random.seed!(42)  # Same seed on all ranks
    A_native = sprand(n, n, 0.3)  # 30% density
    A_native = A_native + I  # Make sure diagonal is populated

    # Convert to PETSc MPIAIJ
    A_petsc = Mat_uniform(A_native; Prefix=MPIAIJ)

    # Test eachrow iteration
    row_lo = A_petsc.obj.row_partition[rank+1]
    row_hi = A_petsc.obj.row_partition[rank+2] - 1

    rows_collected = []
    for row in eachrow(A_petsc)
        push!(rows_collected, row)
    end

    # Check we got the right number of rows
    expected_nrows = row_hi - row_lo + 1
    @test length(rows_collected) == expected_nrows

    # Check each row matches the native data
    for (i, row) in enumerate(rows_collected)
        global_row = row_lo + i - 1
        expected_row = Vector(A_native[global_row, :])
        @test row ≈ expected_row atol=1e-14
    end

    if rank == 0
        println("✓ eachrow for sparse matrices works correctly")
    end
end

@testset "map_rows for sparse matrices" begin
    # Create a sparse matrix (same on all ranks)
    using Random
    n = 20
    Random.seed!(43)
    A_native = sprand(n, n, 0.3)
    A_native = A_native + I

    A_petsc = Mat_uniform(A_native; Prefix=MPIAIJ)

    # Test 1: sum of each row (returns Vec)
    row_sums_petsc = map_rows(sum, A_petsc)
    row_sums_native = vec(sum(A_native, dims=2))

    @test Vector(row_sums_petsc) ≈ row_sums_native atol=1e-14

    if rank == 0
        println("✓ map_rows sum on sparse matrix works")
    end

    # Test 2: create matrix from rows (returns Mat)
    stats_petsc = map_rows(x -> [sum(x), maximum(x)]', A_petsc)

    # Compute expected result
    stats_native = zeros(n, 2)
    for i in 1:n
        row = Vector(A_native[i, :])
        stats_native[i, 1] = sum(row)
        stats_native[i, 2] = maximum(row)
    end

    @test Matrix(stats_petsc) ≈ stats_native atol=1e-14

    if rank == 0
        println("✓ map_rows with matrix output on sparse matrix works")
    end

    # Test 3: map_rows with multiple inputs (sparse + sparse)
    Random.seed!(44)
    B_native = sprand(n, n, 0.3) + I
    B_petsc = Mat_uniform(B_native; Prefix=MPIAIJ)

    combined_petsc = map_rows((x, y) -> [sum(x), sum(y)]', A_petsc, B_petsc)

    combined_native = zeros(n, 2)
    for i in 1:n
        combined_native[i, 1] = sum(Vector(A_native[i, :]))
        combined_native[i, 2] = sum(Vector(B_native[i, :]))
    end

    @test Matrix(combined_petsc) ≈ combined_native atol=1e-14

    if rank == 0
        println("✓ map_rows with multiple sparse matrices works")
    end
end

@testset "map_rows mixed dense and sparse" begin
    # Create both dense and sparse matrices (same on all ranks)
    using Random
    n = 20
    Random.seed!(45)
    A_sparse_native = sprand(n, n, 0.3) + I
    Random.seed!(46)
    A_dense_native = randn(n, n)

    A_sparse_petsc = Mat_uniform(A_sparse_native; Prefix=MPIAIJ)
    A_dense_petsc = Mat_uniform(A_dense_native; Prefix=MPIDENSE)

    # map_rows should work with mixed inputs
    mixed_petsc = map_rows((x, y) -> [sum(x), sum(y)]', A_sparse_petsc, A_dense_petsc)

    mixed_native = zeros(n, 2)
    for i in 1:n
        mixed_native[i, 1] = sum(Vector(A_sparse_native[i, :]))
        mixed_native[i, 2] = sum(A_dense_native[i, :])
    end

    @test Matrix(mixed_petsc) ≈ mixed_native atol=1e-14

    if rank == 0
        println("✓ map_rows with mixed sparse/dense matrices works")
    end
end

@testset "eachrow edge cases" begin
    # Test with very sparse matrix
    n = 10
    A_native = spzeros(n, n)
    A_native[1, 1] = 1.0
    A_native[n, n] = 2.0

    A_petsc = Mat_uniform(A_native; Prefix=MPIAIJ)

    row_lo = A_petsc.obj.row_partition[rank+1]
    row_hi = A_petsc.obj.row_partition[rank+2] - 1

    for (i, row) in enumerate(eachrow(A_petsc))
        global_row = row_lo + i - 1
        expected = Vector(A_native[global_row, :])
        @test row ≈ expected atol=1e-14
    end

    if rank == 0
        println("✓ eachrow handles very sparse matrices correctly")
    end
end

end  # End of parent testset

MPI.Barrier(comm)

# Aggregate counts and print summary on root
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: map_rows sparse (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

if rank == 0
    println("\n" * "="^70)
    println("All map_rows sparse matrix tests passed!")
    println("="^70)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
