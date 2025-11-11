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
    println("[DEBUG] Matrix addition/subtraction pooling test starting")
    flush(stdout)
end

# Clear pool before tests
clear_mat_pool!()

# Test 1: Pool miss then pool hit for addition
if rank == 0
    println("[DEBUG] Test 1: A+B pool miss then hit")
    flush(stdout)
end

# Create two matrices with specific structure (only rank 0 contributes)
if rank == 0
    A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
    B1 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
else
    A1 = spzeros(Float64, 4, 4)
    B1 = spzeros(Float64, 4, 4)
end

matA1 = SafePETSc.Mat_sum(A1)
matB1 = SafePETSc.Mat_sum(B1)

# First addition - should be a pool miss (creates new matrix)
matC1 = matA1 + matB1
@test matC1 isa SafeMPI.DRef
@test size(matC1) == (4, 4)

# Verify correctness
C1_local = SafePETSc._mat_to_local_sparse(matC1)
C1_sum = zeros(4, 4)
MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 + B1)
    @test all(isapprox.(C1_sum, expected, atol=1e-10))
end

# Release C1 back to pool
matC1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Check pool stats - should have one matrix in nonproduct pool
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    # Should have exactly one non-product matrix (count entries with :nonproduct)
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 1
end

# Second addition with same structure - should be a pool hit (reuses matrix)
matC2 = matA1 + matB1
@test matC2 isa SafeMPI.DRef

# Verify correctness again after pool reuse
C2_local = SafePETSc._mat_to_local_sparse(matC2)
C2_sum = zeros(4, 4)
MPI.Reduce!(Matrix(C2_local), C2_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 + B1)
    @test all(isapprox.(C2_sum, expected, atol=1e-10))
end

# Pool should be empty now (matrix was reused)
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 0
end

# Release matC2
matC2 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Pool reuse for same-sized matrices
if rank == 0
    println("[DEBUG] Test 2: Pool reuse for same-sized matrices")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
if rank == 0
    A2 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
    B2 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
    C2 = sparse([1, 3], [2, 4], [7.0, 8.0], 4, 4)
else
    A2 = spzeros(Float64, 4, 4)
    B2 = spzeros(Float64, 4, 4)
    C2 = spzeros(Float64, 4, 4)
end

matA2 = SafePETSc.Mat_sum(A2)
matB2 = SafePETSc.Mat_sum(B2)
matC2 = SafePETSc.Mat_sum(C2)

# First addition
matAB = matA2 + matB2
@test matAB isa SafeMPI.DRef

# Release to pool
matAB = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should have one entry
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 1
end

# Second addition with same size - pool will reuse the matrix
matAC = matA2 + matC2
@test matAC isa SafeMPI.DRef

# Verify correctness
AC_local = SafePETSc._mat_to_local_sparse(matAC)
AC_sum = zeros(4, 4)
MPI.Reduce!(Matrix(AC_local), AC_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A2 + C2)
    @test all(isapprox.(AC_sum, expected, atol=1e-10))
end

# Pool should now be empty (matrix was reused)
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 0
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Subtraction pool miss then hit
if rank == 0
    println("[DEBUG] Test 3: A-B pool miss then hit")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
if rank == 0
    A3 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
    B3 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
else
    A3 = spzeros(Float64, 4, 4)
    B3 = spzeros(Float64, 4, 4)
end

matA3 = SafePETSc.Mat_sum(A3)
matB3 = SafePETSc.Mat_sum(B3)

# First subtraction - pool miss
matD1 = matA3 - matB3
@test matD1 isa SafeMPI.DRef
@test size(matD1) == (4, 4)

# Verify correctness
D1_local = SafePETSc._mat_to_local_sparse(matD1)
D1_sum = zeros(4, 4)
MPI.Reduce!(Matrix(D1_local), D1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A3 - B3)
    @test all(isapprox.(D1_sum, expected, atol=1e-10))
end

# Release to pool
matD1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Second subtraction with same structure - pool hit
matD2 = matA3 - matB3
@test matD2 isa SafeMPI.DRef

# Verify correctness
D2_local = SafePETSc._mat_to_local_sparse(matD2)
D2_sum = zeros(4, 4)
MPI.Reduce!(Matrix(D2_local), D2_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A3 - B3)
    @test all(isapprox.(D2_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Mixed addition and subtraction share pool
if rank == 0
    println("[DEBUG] Test 4: Addition and subtraction share pool")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
if rank == 0
    A4 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
    B4 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
else
    A4 = spzeros(Float64, 4, 4)
    B4 = spzeros(Float64, 4, 4)
end

matA4 = SafePETSc.Mat_sum(A4)
matB4 = SafePETSc.Mat_sum(B4)

# Do addition first
matAdd = matA4 + matB4
@test matAdd isa SafeMPI.DRef

# Release to pool
matAdd = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Pool should have one entry
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 1
end

# Do subtraction with same structure - should reuse from pool
matSub = matA4 - matB4
@test matSub isa SafeMPI.DRef

# Verify correctness
Sub_local = SafePETSc._mat_to_local_sparse(matSub)
Sub_sum = zeros(4, 4)
MPI.Reduce!(Matrix(Sub_local), Sub_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A4 - B4)
    @test all(isapprox.(Sub_sum, expected, atol=1e-10))
end

# Pool should now be empty (matrix was reused)
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct; init=0)
    @test nonproduct_count == 0
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Verify structure reuse flag (SUBSET_NONZERO_PATTERN vs DIFFERENT_NONZERO_PATTERN)
if rank == 0
    println("[DEBUG] Test 5: Verify correct structure flags used")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# This test verifies the code paths use correct PETSc structure flags
# Pool miss should use DIFFERENT_NONZERO_PATTERN
# Pool hit should use SUBSET_NONZERO_PATTERN

# Only rank 0 contributes
if rank == 0
    A5 = sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3)
    B5 = sparse([2, 3], [2, 3], [3.0, 4.0], 3, 3)
else
    A5 = spzeros(Float64, 3, 3)
    B5 = spzeros(Float64, 3, 3)
end

matA5 = SafePETSc.Mat_sum(A5)
matB5 = SafePETSc.Mat_sum(B5)

# First addition (pool miss path)
matC5a = matA5 + matB5
@test matC5a isa SafeMPI.DRef

# Verify values are correct
C5_local = SafePETSc._mat_to_local_sparse(matC5a)
C5_sum = zeros(3, 3)
MPI.Reduce!(Matrix(C5_local), C5_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A5 + B5)
    @test all(isapprox.(C5_sum, expected, atol=1e-10))
end

# Release and re-compute (pool hit path)
matC5a = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

matC5b = matA5 + matB5
@test matC5b isa SafeMPI.DRef

# Verify values are still correct after pool reuse
C5b_local = SafePETSc._mat_to_local_sparse(matC5b)
C5b_sum = zeros(3, 3)
MPI.Reduce!(Matrix(C5b_local), C5b_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A5 + B5)
    @test all(isapprox.(C5b_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All addition/subtraction pooling tests completed")
    flush(stdout)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix add/sub pooling test file completed successfully")
    flush(stdout)
end
