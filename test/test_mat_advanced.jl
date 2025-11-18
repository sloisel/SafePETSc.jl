using Test
using MPI
using SafePETSc
SafePETSc.Init()
using LinearAlgebra
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset MPITestHarness.QuietTestSet "Advanced matrix operations" begin
    println(io0(), "[DEBUG] Advanced matrix operations test starting")

    # Test 1: Materialize adjoint as a new matrix: Mat(A')
    println(io0(), "[DEBUG] Test 1: Materialize A' as new matrix")
    A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0])
    B = Mat(A')  # This should create a materialized transpose
    @test size(B) == (3, 2)
    B_mat = Matrix(B)
    @test B_mat ≈ [1.0 4.0; 2.0 5.0; 3.0 6.0]
    println(io0(), "[DEBUG] Test 1 passed")

    # Test 2: Adjoint matrix times vector: A' * x
    println(io0(), "[DEBUG] Test 2: Adjoint matrix times vector A' * x")
    M = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0])  # 2x3 matrix
    v = Vec_uniform([10.0, 20.0])  # 2-element vector
    # A' is 3x2, v is 2x1, so result should be 3x1
    result = M' * v
    @test length(result) == 3
    # Expected: [1 4; 2 5; 3 6] * [10; 20] = [1*10+4*20; 2*10+5*20; 3*10+6*20] = [90; 120; 150]
    @test Vector(result) ≈ [90.0, 120.0, 150.0]
    # Test with square matrix
    A_sq = Mat_uniform([1.0 2.0; 3.0 4.0])
    x_sq = Vec_uniform([5.0, 6.0])
    res_sq = A_sq' * x_sq
    # [1 3; 2 4] * [5; 6] = [1*5+3*6; 2*5+4*6] = [23; 34]
    @test Vector(res_sq) ≈ [23.0, 34.0]
    println(io0(), "[DEBUG] Test 2 passed")

    println(io0(), "[DEBUG] All advanced matrix operations tests completed")
end

# Aggregate per-rank counts and print a single summary on root
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
    println("Test Summary: Advanced matrix operations (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)
