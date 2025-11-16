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
