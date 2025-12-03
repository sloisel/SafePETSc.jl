using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Vec min/max test starting")
    flush(stdout)
end

ts = @testset QuietTestSet "Vec min/max tests" begin

if rank == 0
    println("[DEBUG] Test 1: Basic maximum and minimum")
    flush(stdout)
end

# Test 1: Basic maximum and minimum
v = Vec_uniform([1.0, 4.0, 2.0, 3.0])
@test sum(v) ≈ 10.0
@test maximum(v) ≈ 4.0
@test minimum(v) ≈ 1.0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Test 2: Negative values")
    flush(stdout)
end

# Test 2: Negative values
v2 = Vec_uniform([-5.0, 2.0, -1.0, 3.0])
@test maximum(v2) ≈ 3.0
@test minimum(v2) ≈ -5.0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Test 3: All same values")
    flush(stdout)
end

# Test 3: All same values
v3 = Vec_uniform([7.0, 7.0, 7.0, 7.0])
@test maximum(v3) ≈ 7.0
@test minimum(v3) ≈ 7.0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Test 4: Single element")
    flush(stdout)
end

# Test 4: Single element
v4 = Vec_uniform([42.0])
@test maximum(v4) ≈ 42.0
@test minimum(v4) ≈ 42.0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Test 5: Larger vector")
    flush(stdout)
end

# Test 5: Larger vector distributed across ranks
v5 = Vec_uniform(collect(1.0:16.0))
@test maximum(v5) ≈ 16.0
@test minimum(v5) ≈ 1.0
@test sum(v5) ≈ 136.0  # 16*17/2

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

end  # End of QuietTestSet

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
    println("Test Summary: Vec min/max tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec min/max test file completed successfully")
    flush(stdout)
end
