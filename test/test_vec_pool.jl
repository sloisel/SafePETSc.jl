using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Vec pooling tests" begin

# Test 1: Basic pooling - create, destroy, create again
# Clear pool to start fresh
SafePETSc.clear_vec_pool!()

v = ones(16)
dr1 = SafePETSc.Vec_uniform(v; prefix="test_")
ptr1 = dr1.obj.v.ptr  # Store pointer to underlying PETSc vec

# Destroy the vector (should go to pool)
dr1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Check pool stats
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    # Pool should contain one vector of size 16 with prefix "test_"
    @test haskey(stats, (16, "test_", Float64))
    @test stats[(16, "test_", Float64)] == 1
end

# Create another vector with same size and prefix - should reuse from pool
dr2 = SafePETSc.Vec_uniform(v; prefix="test_")
ptr2 = dr2.obj.v.ptr

# Should have the same pointer (reused from pool)
@test ptr1 == ptr2

# Pool should now be empty
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test !haskey(stats, (16, "test_", Float64)) || stats[(16, "test_", Float64)] == 0
end

# Cleanup
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Different prefix should not reuse
SafePETSc.clear_vec_pool!()

dr3 = SafePETSc.Vec_uniform(v; prefix="prefix1_")
ptr3 = dr3.obj.v.ptr

dr3 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Create with different prefix - should NOT reuse
dr4 = SafePETSc.Vec_uniform(v; prefix="prefix2_")
ptr4 = dr4.obj.v.ptr

@test ptr3 != ptr4

# Both vectors should now be in pool (if we destroy dr4)
dr4 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "prefix1_", Float64))
    @test haskey(stats, (16, "prefix2_", Float64))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Different partition should not reuse
SafePETSc.clear_vec_pool!()

partition1 = SafePETSc.default_row_partition(16, nranks)
dr5 = SafePETSc.Vec_uniform(v; row_partition=partition1, prefix="")
ptr5 = dr5.obj.v.ptr

dr5 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Create with different partition (if we have 4 ranks)
if nranks == 4
    partition2 = [1, 4, 8, 12, 17]  # Different from default
    if partition2 != partition1
        dr6 = SafePETSc.Vec_uniform(v; row_partition=partition2, prefix="")
        ptr6 = dr6.obj.v.ptr

        @test ptr5 != ptr6  # Should NOT reuse due to different partition

        SafeMPI.check_and_destroy!()
        MPI.Barrier(comm)
    end
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Disable pooling
SafePETSc.clear_vec_pool!()
SafePETSc.ENABLE_VEC_POOL[] = false

dr7 = SafePETSc.Vec_uniform(v; prefix="test_")
ptr7 = dr7.obj.v.ptr

dr7 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should be empty because pooling was disabled
stats = SafePETSc.get_vec_pool_stats()
@test isempty(stats)

# Re-enable pooling
SafePETSc.ENABLE_VEC_POOL[] = true

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Clear pool destroys vectors
SafePETSc.clear_vec_pool!()

dr8 = SafePETSc.Vec_uniform(v; prefix="clear_test_")
dr8 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Verify vector is in pool
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "clear_test_", Float64))
end

# Clear pool - should destroy vectors
SafePETSc.clear_vec_pool!()

# Pool should now be empty
stats = SafePETSc.get_vec_pool_stats()
@test isempty(stats)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: Vector contents are zeroed when returned to pool
SafePETSc.clear_vec_pool!()

# Create vector with non-zero values
v_nonzero = collect(1.0:16.0)
dr9 = SafePETSc.Vec_uniform(v_nonzero; prefix="zero_test_")

dr9 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Create new vector with zeros - should reuse from pool
v_zeros = zeros(16)
dr10 = SafePETSc.Vec_uniform(v_zeros; prefix="zero_test_")

# Check that values are zeros (were cleared by pool)
local_view = PETSc.unsafe_localarray(dr10.obj.v; read=true, write=false)
try
    @test all(local_view[:] .== 0.0)
finally
    Base.finalize(local_view)
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
    println("Test Summary: Vec pooling tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

# Note: We don't call MPI.Finalize() here because Julia's MPI.jl
# automatically finalizes MPI at exit via atexit hook
