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
    println("[DEBUG] Matrix product pooling test starting")
    flush(stdout)
end

# Test 1: A*B pool miss then pool hit (MATPRODUCT_AB)
if rank == 0
    println("[DEBUG] Test 1: A*B pool miss then hit")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
# Using non-square matrices: 3×5 * 5×4 = 3×4 to expose row/col bugs
if rank == 0
    A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 5)  # 3×5 matrix
    B1 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 5, 4)  # 5×4 matrix
else
    A1 = spzeros(Float64, 3, 5)
    B1 = spzeros(Float64, 5, 4)
end

matA1 = SafePETSc.Mat_sum(A1)
matB1 = SafePETSc.Mat_sum(B1)

# First product - should be a pool miss (creates new matrix)
matC1 = matA1 * matB1
@test matC1 isa SafeMPI.DRef
@test size(matC1) == (3, 4)  # Result is 3×4
@test matC1.obj.product_type == SafePETSc.MATPRODUCT_AB

# Verify correctness
C1_local = SafePETSc._mat_to_local_sparse(matC1)
C1_sum = zeros(3, 4)
MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 * B1)
    @test all(isapprox.(C1_sum, expected, atol=1e-10))
end

# Release C1 back to pool
matC1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Check pool stats - should have one matrix in product pool
stats = SafePETSc.get_mat_pool_stats()
if rank == 0
    # Count product pool entries (keys ending with :product)
    product_count = sum(v for (k, v) in stats if length(k) >= 3 && k[end] == :product; init=0)
    @test product_count >= 1
end

# Second product with same matrices - should be a pool hit (reuses matrix)
matC2 = matA1 * matB1
@test matC2 isa SafeMPI.DRef

# Verify correctness again
C2_local = SafePETSc._mat_to_local_sparse(matC2)
C2_sum = zeros(3, 4)  # Updated to match 3×4 result
MPI.Reduce!(Matrix(C2_local), C2_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 * B1)
    @test all(isapprox.(C2_sum, expected, atol=1e-10))
end

# Release matC2
matC2 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: A'*B pool miss then hit (MATPRODUCT_AtB)
if rank == 0
    println("[DEBUG] Test 2: A'*B pool miss then hit")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
if rank == 0
    A2 = sparse([1, 2, 3], [1, 1, 1], [1.0, 2.0, 3.0], 4, 3)  # 4x3 matrix
    B2 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 4, 4)  # 4x4 matrix
else
    A2 = spzeros(Float64, 4, 3)
    B2 = spzeros(Float64, 4, 4)
end

matA2 = SafePETSc.Mat_sum(A2)
matB2 = SafePETSc.Mat_sum(B2)

# First transpose product - pool miss
matC2a = matA2' * matB2
@test matC2a isa SafeMPI.DRef
@test size(matC2a) == (3, 4)
@test matC2a.obj.product_type == SafePETSc.MATPRODUCT_AtB

# Release to pool
matC2a = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Second transpose product - pool hit
matC2b = matA2' * matB2
@test matC2b isa SafeMPI.DRef
@test size(matC2b) == (3, 4)

# Release matC2b
matC2b = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] End of Test 2")
    flush(stdout)
end

# Test 3: A*B' pool miss then hit (MATPRODUCT_ABt)
if rank == 0
    println("[DEBUG] Test 3: A*B' pool miss then hit")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
# Using non-square matrices: 4×6 * (3×6)' = 4×3 to expose row/col bugs
if rank == 0
    A3 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 6)  # 4×6 matrix
    B3 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 3, 6)  # 3×6 matrix
else
    A3 = spzeros(Float64, 4, 6)
    B3 = spzeros(Float64, 3, 6)
end

matA3 = SafePETSc.Mat_sum(A3)
matB3 = SafePETSc.Mat_sum(B3)

# First product - pool miss
matC3a = matA3 * matB3'
@test matC3a isa SafeMPI.DRef
@test size(matC3a) == (4, 3)  # Result is 4×3
@test matC3a.obj.product_type == SafePETSc.MATPRODUCT_ABt

# Release to pool
matC3a = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Second product - pool hit
matC3b = matA3 * matB3'
@test matC3b isa SafeMPI.DRef
@test size(matC3b) == (4, 3)  # Result is 4×3

# Release matC3b
matC3b = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Verify correctness of pooled products
if rank == 0
    println("[DEBUG] Test 4: Verify pooled products compute correct values")
    flush(stdout)
end

SafePETSc.clear_mat_pool!()

# Only rank 0 contributes
# Using non-square matrices: 2×5 * 5×3 = 2×3 to expose row/col bugs
if rank == 0
    A4 = sparse([1, 2], [1, 2], [2.0, 3.0], 2, 5)  # 2×5 matrix
    B4 = sparse([1, 2], [1, 2], [4.0, 5.0], 5, 3)  # 5×3 matrix
else
    A4 = spzeros(Float64, 2, 5)
    B4 = spzeros(Float64, 5, 3)
end

matA4 = SafePETSc.Mat_sum(A4)
matB4 = SafePETSc.Mat_sum(B4)

# First product
matC4a = matA4 * matB4
C4a_local = SafePETSc._mat_to_local_sparse(matC4a)
C4a_sum = zeros(2, 3)  # Result is 2×3
MPI.Reduce!(Matrix(C4a_local), C4a_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A4 * B4)
    @test all(isapprox.(C4a_sum, expected, atol=1e-10))
end

# Release and re-compute
matC4a = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Second product (should reuse from pool)
matC4b = matA4 * matB4
C4b_local = SafePETSc._mat_to_local_sparse(matC4b)
C4b_sum = zeros(2, 3)  # Result is 2×3
MPI.Reduce!(Matrix(C4b_local), C4b_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A4 * B4)
    @test all(isapprox.(C4b_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All product pooling tests completed")
    flush(stdout)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix product pooling test file completed successfully")
    flush(stdout)
end
