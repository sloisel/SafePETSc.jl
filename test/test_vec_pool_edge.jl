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

if rank == 0
    println("[DEBUG] Vec pooling edge cases test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Vec pooling edge case tests" begin

# Test 1: Pool stats behavior
if rank == 0
    println("[DEBUG] Test 1: Pool stats with empty and populated pool")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()

# Empty pool
stats = SafePETSc.get_vec_pool_stats()
@test isempty(stats)

# Add a vector to pool
v = ones(16)
dr1 = SafePETSc.Vec_uniform(v)
dr1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should have one entry (default Prefix is MPIAIJ)
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "MPIAIJ_", Float64))
    @test stats[(16, "MPIAIJ_", Float64)] == 1
end

# Create another vector - should reuse from pool
dr2 = SafePETSc.Vec_uniform(v)
@test dr2 isa SafeMPI.DRef

# Pool should now be empty
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test !haskey(stats, (16, "MPIAIJ_", Float64)) || stats[(16, "MPIAIJ_", Float64)] == 0
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Different prefixes don't share pool entries
if rank == 0
    println("[DEBUG] Test 2: Different prefixes use separate pool entries")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()

# Create vector with MPIAIJ prefix
dr_a = SafePETSc.Vec_uniform(v; Prefix=SafePETSc.MPIAIJ)
dr_a = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Create vector with MPIDENSE prefix
dr_b = SafePETSc.Vec_uniform(v; Prefix=SafePETSc.MPIDENSE)
dr_b = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Both should be in pool with different prefixes
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "MPIAIJ_", Float64))
    @test haskey(stats, (16, "MPIDENSE_", Float64))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Mixed sizes in pool
if rank == 0
    println("[DEBUG] Test 3: Multiple vector sizes in pool")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()

v8 = ones(8)
v16 = ones(16)
v32 = ones(32)

# Add vectors of different sizes to pool
dr8 = SafePETSc.Vec_uniform(v8)
dr16 = SafePETSc.Vec_uniform(v16)
dr32 = SafePETSc.Vec_uniform(v32)

dr8 = nothing
dr16 = nothing
dr32 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Check all three sizes are in pool
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (8, "MPIAIJ_", Float64))
    @test haskey(stats, (16, "MPIAIJ_", Float64))
    @test haskey(stats, (32, "MPIAIJ_", Float64))
end

# Create size 16 - should only consume the 16-sized vector
dr16_new = SafePETSc.Vec_uniform(v16)
@test dr16_new isa SafeMPI.DRef

# Other sizes should still be in pool
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (8, "MPIAIJ_", Float64))
    @test haskey(stats, (32, "MPIAIJ_", Float64))
    @test !haskey(stats, (16, "MPIAIJ_", Float64)) || stats[(16, "MPIAIJ_", Float64)] == 0
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Toggle ENABLE_VEC_POOL during execution
if rank == 0
    println("[DEBUG] Test 4: Toggle pooling on/off during execution")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()
SafePETSc.ENABLE_VEC_POOL[] = true

# Create with pooling enabled
dr1 = SafePETSc.Vec_uniform(v)
dr1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Should be in pool
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "MPIAIJ_", Float64))
end

# Disable pooling
SafePETSc.ENABLE_VEC_POOL[] = false

# Create new vector - should NOT consume from pool when pooling is disabled
dr2 = SafePETSc.Vec_uniform(v)
@test dr2 isa SafeMPI.DRef

dr2 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should still have the first vector
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test haskey(stats, (16, "MPIAIJ_", Float64))
    @test stats[(16, "MPIAIJ_", Float64)] == 1
end

# Re-enable pooling
SafePETSc.ENABLE_VEC_POOL[] = true

# Now should consume from pool again
dr3 = SafePETSc.Vec_uniform(v)
@test dr3 isa SafeMPI.DRef

# Pool should be empty
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    @test !haskey(stats, (16, "MPIAIJ_", Float64)) || stats[(16, "MPIAIJ_", Float64)] == 0
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Clear pool removes all entries
if rank == 0
    println("[DEBUG] Test 5: Clear pool removes all entries")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()

# Add vectors with different prefixes and sizes
v1 = ones(8)
v2 = ones(16)

dr_a = SafePETSc.Vec_uniform(v1)
dr_b = SafePETSc.Vec_uniform(v2)
dr_c = SafePETSc.Vec_uniform(v1)

dr_a = nothing
dr_b = nothing
dr_c = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should have 3 entries
stats = SafePETSc.get_vec_pool_stats()
if rank == 0
    total_entries = sum(values(stats))
    @test total_entries == 3
end

# Clear pool
SafePETSc.clear_vec_pool!()

# Pool should be completely empty
stats = SafePETSc.get_vec_pool_stats()
@test isempty(stats)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: Vectors zeroed before pooling
if rank == 0
    println("[DEBUG] Test 6: Vectors zeroed when returned to pool")
    flush(stdout)
end

SafePETSc.clear_vec_pool!()

# Create vector with specific values
v_init = collect(1.0:16.0)
dr_init = SafePETSc.Vec_uniform(v_init)

# Verify initial values are correct
nr = MPI.Comm_rank(MPI.COMM_WORLD)
partition = SafePETSc.default_row_partition(16, nranks)
lo = partition[nr+1]
hi = partition[nr+2] - 1
expected_local = v_init[lo:hi]

local_view = PETSc.unsafe_localarray(dr_init.obj.v; read=true, write=false)
try
    @test all(isapprox.(local_view[:], expected_local, atol=1e-10))
finally
    Base.finalize(local_view)
end

# Release to pool (should zero the vector)
dr_init = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Create new vector (should reuse and be zeroed)
v_zeros = zeros(16)
dr_reused = SafePETSc.Vec_uniform(v_zeros)

# Should be all zeros
local_view2 = PETSc.unsafe_localarray(dr_reused.obj.v; read=true, write=false)
try
    @test all(local_view2[:] .== 0.0)
finally
    Base.finalize(local_view2)
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: Direct destruction when pooling is disabled
if rank == 0
    println("[DEBUG] Test 7: Direct PETSc destruction when pooling disabled")
    flush(stdout)
end

# Disable pooling
old_pool_setting = SafePETSc.ENABLE_VEC_POOL[]
SafePETSc.ENABLE_VEC_POOL[] = false

# Create vector - should be destroyed directly when released, not pooled
v_direct = ones(16)
dr_direct = SafePETSc.Vec_uniform(v_direct)

# Release the vector (should destroy directly via _destroy_petsc_vec!)
dr_direct = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should remain empty
stats = SafePETSc.get_vec_pool_stats()
@test isempty(stats)

# Re-enable pooling
SafePETSc.ENABLE_VEC_POOL[] = old_pool_setting

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: Broadcast error path for unsupported types
if rank == 0
    println("[DEBUG] Test 8: Broadcast error for unsupported types")
    flush(stdout)
end

v_test = ones(16)
dr_test = SafePETSc.Vec_uniform(v_test)

# Try to broadcast with a Matrix (unsupported type) - should error
unsupported_array = [1.0 2.0; 3.0 4.0]
@test_throws DimensionMismatch dr_test .+ unsupported_array

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All vec pooling edge case tests completed")
    flush(stdout)
end

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
    println("Test Summary: Vec pooling edge case tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec pooling edge case test file completed successfully")
    flush(stdout)
end

