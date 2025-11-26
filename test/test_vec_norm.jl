using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Vec norm test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Vec norm tests" begin

if rank == 0
    println("[DEBUG] Vec norm Test 1 starting - 2-norm (default)")
    flush(stdout)
end

# Test 1: 2-norm (Euclidean norm) - default
v1_native = [3.0, 4.0, 0.0, 0.0]
v1 = SafePETSc.Vec_uniform(v1_native)

# Compute norm using SafePETSc
norm_petsc = norm(v1)
# Compute norm using native Julia
norm_native = norm(v1_native)

@test norm_petsc ≈ norm_native atol=1e-12
@test norm_petsc ≈ 5.0 atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 1 completed, starting Test 2")
    flush(stdout)
end

# Test 2: Explicit 2-norm
v2_native = [1.0, 2.0, 3.0, 4.0]
v2 = SafePETSc.Vec_uniform(v2_native)

norm_petsc_2 = norm(v2, 2)
norm_native_2 = norm(v2_native, 2)

@test norm_petsc_2 ≈ norm_native_2 atol=1e-12
@test norm_petsc_2 ≈ sqrt(1.0 + 4.0 + 9.0 + 16.0) atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 2 completed, starting Test 3")
    flush(stdout)
end

# Test 3: 1-norm (sum of absolute values)
v3_native = [1.0, -2.0, 3.0, -4.0]
v3 = SafePETSc.Vec_uniform(v3_native)

norm_petsc_1 = norm(v3, 1)
norm_native_1 = norm(v3_native, 1)

@test norm_petsc_1 ≈ norm_native_1 atol=1e-12
@test norm_petsc_1 ≈ 10.0 atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 3 completed, starting Test 4")
    flush(stdout)
end

# Test 4: Infinity norm (max absolute value)
v4_native = [1.0, -5.0, 3.0, 2.0]
v4 = SafePETSc.Vec_uniform(v4_native)

norm_petsc_inf = norm(v4, Inf)
norm_native_inf = norm(v4_native, Inf)

@test norm_petsc_inf ≈ norm_native_inf atol=1e-12
@test norm_petsc_inf ≈ 5.0 atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 4 completed, starting Test 5")
    flush(stdout)
end

# Test 5: Zero vector norms
v5_native = [0.0, 0.0, 0.0, 0.0]
v5 = SafePETSc.Vec_uniform(v5_native)

@test norm(v5) ≈ 0.0 atol=1e-12
@test norm(v5, 1) ≈ 0.0 atol=1e-12
@test norm(v5, Inf) ≈ 0.0 atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 5 completed, starting Test 6")
    flush(stdout)
end

# Test 6: Larger vector with different partition sizes
v6_native = Float64[i for i in 1:16]
v6 = SafePETSc.Vec_uniform(v6_native)

norm_petsc_large = norm(v6)
norm_native_large = norm(v6_native)

@test norm_petsc_large ≈ norm_native_large atol=1e-10

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 6 completed, starting Test 7")
    flush(stdout)
end

# Test 7: All three norms on the same larger vector
v7_native = [1.5, -2.5, 3.5, -4.5, 5.5, -6.5, 7.5, -8.5]
v7 = SafePETSc.Vec_uniform(v7_native)

norm1 = norm(v7, 1)
norm2 = norm(v7, 2)
norminf = norm(v7, Inf)

norm1_native = norm(v7_native, 1)
norm2_native = norm(v7_native, 2)
norminf_native = norm(v7_native, Inf)

@test norm1 ≈ norm1_native atol=1e-10
@test norm2 ≈ norm2_native atol=1e-10
@test norminf ≈ norminf_native atol=1e-10

# Verify norm ordering: ||v||_inf <= ||v||_2 <= ||v||_1
@test norminf <= norm2 + 1e-10
@test norm2 <= norm1 + 1e-10

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 7 completed, starting Test 8")
    flush(stdout)
end

# Test 8: Custom row partition
if nranks == 4
    custom_partition = [1, 5, 9, 13, 17]
    v8_native = Float64[i for i in 1:16]
    v8 = SafePETSc.Vec_uniform(v8_native; row_partition=custom_partition)

    norm_custom = norm(v8)
    norm_native_custom = norm(v8_native)

    @test norm_custom ≈ norm_native_custom atol=1e-10

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

if rank == 0
    println("[DEBUG] Vec norm Test 8 completed, starting Test 9")
    flush(stdout)
end

# Test 9: Unit vectors
for i in 1:4
    v9_native = zeros(4)
    v9_native[i] = 1.0
    v9 = SafePETSc.Vec_uniform(v9_native)

    @test norm(v9) ≈ 1.0 atol=1e-12
    @test norm(v9, 1) ≈ 1.0 atol=1e-12
    @test norm(v9, Inf) ≈ 1.0 atol=1e-12

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

if rank == 0
    println("[DEBUG] Vec norm Test 9 completed, starting Test 10")
    flush(stdout)
end

# Test 10: Additional norm test
v10_native = [2.0, -3.0, 4.0, -5.0]
v10 = SafePETSc.Vec_uniform(v10_native)

norm_dense = norm(v10)
norm_native_dense = norm(v10_native)

@test norm_dense ≈ norm_native_dense atol=1e-12

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm Test 10 completed")
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
    println("Test Summary: Vec norm tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec norm test file completed successfully")
    flush(stdout)
end

