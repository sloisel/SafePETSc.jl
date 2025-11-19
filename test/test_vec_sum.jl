using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Vec_sum test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Vec_sum tests" begin

if rank == 0
    println("[DEBUG] Vec_sum Test 1 starting")
    flush(stdout)
end

# Test 1: Basic summing - each rank contributes to different indices
v = sparsevec(Int[], Float64[], 16)  # Start with empty sparse vector
if rank == 0
    v[1] = 1.0
    v[2] = 2.0
elseif rank == 1
    v[5] = 10.0
elseif rank == 2
    v[9] = 100.0
elseif rank == 3
    v[13] = 1000.0
end

dr = SafePETSc.Vec_sum(v)
@test dr isa SafeMPI.DRef
obj = dr.obj
@test obj.v isa PETSc.Vec
@test length(obj.row_partition) == nranks + 1

# Check the result
local_view = PETSc.unsafe_localarray(obj.v; read=true, write=false)
try
    lo = obj.row_partition[rank+1]
    hi = obj.row_partition[rank+2] - 1

    # Verify summed values in each rank's partition
    if rank == 0  # Owns indices 1-4
        @test local_view[1-lo+1] == 1.0  # Index 1
        @test local_view[2-lo+1] == 2.0  # Index 2
        @test local_view[3-lo+1] == 0.0  # Index 3 (no contribution)
        @test local_view[4-lo+1] == 0.0  # Index 4 (no contribution)
    elseif rank == 1  # Owns indices 5-8
        @test local_view[5-lo+1] == 10.0  # Index 5
        @test local_view[6-lo+1] == 0.0
        @test local_view[7-lo+1] == 0.0
        @test local_view[8-lo+1] == 0.0
    elseif rank == 2  # Owns indices 9-12
        @test local_view[9-lo+1] == 100.0  # Index 9
        @test local_view[10-lo+1] == 0.0
        @test local_view[11-lo+1] == 0.0
        @test local_view[12-lo+1] == 0.0
    elseif rank == 3  # Owns indices 13-16
        @test local_view[13-lo+1] == 1000.0  # Index 13
        @test local_view[14-lo+1] == 0.0
        @test local_view[15-lo+1] == 0.0
        @test local_view[16-lo+1] == 0.0
    end
finally
    Base.finalize(local_view)
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec_sum Test 1 completed, starting Test 2")
    flush(stdout)
end

# Test 2: Overlapping contributions - multiple ranks contribute to same indices
v2 = sparsevec(Int[], Float64[], 16)
# All ranks contribute to index 8
v2[8] = Float64(rank + 1)  # rank 0 adds 1, rank 1 adds 2, rank 2 adds 3, rank 3 adds 4

dr2 = SafePETSc.Vec_sum(v2)
local_view2 = PETSc.unsafe_localarray(dr2.obj.v; read=true, write=false)
try
    lo = dr2.obj.row_partition[rank+1]
    hi = dr2.obj.row_partition[rank+2] - 1

    # Index 8 should be owned by rank 1 (indices 5-8)
    if rank == 1
        # Sum should be 1 + 2 + 3 + 4 = 10
        @test local_view2[8-lo+1] ≈ 10.0
    end
finally
    Base.finalize(local_view2)
end

if rank == 0
    println("[DEBUG] Test 2 cleanup: about to call check_and_destroy!()")
    flush(stdout)
end

SafeMPI.check_and_destroy!()

if rank == 0
    println("[DEBUG] Test 2 cleanup: check_and_destroy!() done, about to barrier")
    flush(stdout)
end

MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Test 2 cleanup: barrier done, starting Test 3")
    flush(stdout)
end

# Test 3: own_rank_only=true with valid partitioning
if rank == 0
    println("[DEBUG] Test 3: creating sparse vector")
    flush(stdout)
end

v3 = sparsevec(Int[], Float64[], 16)
# Each rank only sets indices in its own partition
lo = SafePETSc.default_row_partition(16, nranks)[rank+1]
hi = SafePETSc.default_row_partition(16, nranks)[rank+2] - 1

if rank == 0
    println("[DEBUG] Test 3: lo=$lo, hi=$hi, about to check condition")
    flush(stdout)
end

if lo <= hi
    if rank == 0
        println("[DEBUG] Test 3: setting v3[$lo]")
        flush(stdout)
    end
    v3[lo] = Float64(rank * 10)
end

if rank == 0
    println("[DEBUG] Test 3: about to call Vec_sum with own_rank_only=true")
    flush(stdout)
end

dr3 = SafePETSc.Vec_sum(v3; own_rank_only=true)
@test dr3 isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Custom row partition
if nranks == 4
    custom_partition = [1, 5, 9, 13, 17]
    v4 = sparsevec(Int[], Float64[], 16)

    # Each rank contributes to its own partition
    partition_lo = custom_partition[rank+1]
    partition_hi = custom_partition[rank+2] - 1

    if partition_lo <= partition_hi
        v4[partition_lo] = Float64(100 * rank)
    end

    dr4 = SafePETSc.Vec_sum(v4; row_partition=custom_partition, own_rank_only=true)
    @test dr4.obj.row_partition == custom_partition

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

# Test 5: Empty sparse vector (no nonzeros)
v5 = sparsevec(Int[], Float64[], 16)
dr5 = SafePETSc.Vec_sum(v5)
@test dr5 isa SafeMPI.DRef

# Verify all zeros
local_view5 = PETSc.unsafe_localarray(dr5.obj.v; read=true, write=false)
try
    @test all(local_view5[:] .== 0.0)
finally
    Base.finalize(local_view5)
end

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
    println("Test Summary: Vec_sum tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec_sum test file completed successfully")
    flush(stdout)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
