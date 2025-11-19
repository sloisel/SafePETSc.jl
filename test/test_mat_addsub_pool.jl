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
    println("[DEBUG] Matrix addition/subtraction with non-square matrices test starting")
    flush(stdout)
end

# Test 1: Addition with 5×3 non-square matrices
if rank == 0
    println("[DEBUG] Test 1: A+B with 5×3 matrices")
    flush(stdout)
end

# Create two non-square matrices with specific structure (only rank 0 contributes)
# Using 5×3 matrices to expose row/col bugs
if rank == 0
    A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 5, 3)  # 5×3 matrix
    B1 = sparse([1, 2, 4], [1, 2, 3], [4.0, 5.0, 6.0], 5, 3)  # 5×3 matrix
else
    A1 = spzeros(Float64, 5, 3)
    B1 = spzeros(Float64, 5, 3)
end

matA1 = SafePETSc.Mat_sum(A1)
matB1 = SafePETSc.Mat_sum(B1)

# Test addition
matC1 = matA1 + matB1
@test matC1 isa SafeMPI.DRef
@test size(matC1) == (5, 3)  # Result is 5×3

# Verify correctness
C1_local = SafePETSc._mat_to_local_sparse(matC1)
C1_sum = zeros(5, 3)
MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 + B1)
    @test all(isapprox.(C1_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Addition with 6×4 non-square matrices
if rank == 0
    println("[DEBUG] Test 2: Multiple additions with 6×4 matrices")
    flush(stdout)
end

# Only rank 0 contributes (using 6×4 non-square matrices)
if rank == 0
    A2 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 6, 4)  # 6×4 matrix
    B2 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 6, 4)  # 6×4 matrix
    C2 = sparse([1, 3], [2, 4], [7.0, 8.0], 6, 4)  # 6×4 matrix
else
    A2 = spzeros(Float64, 6, 4)
    B2 = spzeros(Float64, 6, 4)
    C2 = spzeros(Float64, 6, 4)
end

matA2 = SafePETSc.Mat_sum(A2)
matB2 = SafePETSc.Mat_sum(B2)
matC2 = SafePETSc.Mat_sum(C2)

# Test A+B
matAB = matA2 + matB2
@test matAB isa SafeMPI.DRef

# Test A+C
matAC = matA2 + matC2
@test matAC isa SafeMPI.DRef

# Verify correctness of A+C
AC_local = SafePETSc._mat_to_local_sparse(matAC)
AC_sum = zeros(6, 4)
MPI.Reduce!(Matrix(AC_local), AC_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A2 + C2)
    @test all(isapprox.(AC_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Subtraction with 3×7 non-square matrices
if rank == 0
    println("[DEBUG] Test 3: A-B with 3×7 matrices")
    flush(stdout)
end

# Only rank 0 contributes (using 3×7 non-square matrices)
if rank == 0
    A3 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 7)  # 3×7 matrix
    B3 = sparse([1, 2, 3], [1, 2, 4], [4.0, 5.0, 6.0], 3, 7)  # 3×7 matrix
else
    A3 = spzeros(Float64, 3, 7)
    B3 = spzeros(Float64, 3, 7)
end

matA3 = SafePETSc.Mat_sum(A3)
matB3 = SafePETSc.Mat_sum(B3)

# Test subtraction
matD1 = matA3 - matB3
@test matD1 isa SafeMPI.DRef
@test size(matD1) == (3, 7)  # Result is 3×7

# Verify correctness
D1_local = SafePETSc._mat_to_local_sparse(matD1)
D1_sum = zeros(3, 7)
MPI.Reduce!(Matrix(D1_local), D1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A3 - B3)
    @test all(isapprox.(D1_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Mixed addition and subtraction with 4×8 matrices
if rank == 0
    println("[DEBUG] Test 4: Addition and subtraction with 4×8 matrices")
    flush(stdout)
end

# Only rank 0 contributes (using 4×8 non-square matrices)
if rank == 0
    A4 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 8)  # 4×8 matrix
    B4 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 8)  # 4×8 matrix
else
    A4 = spzeros(Float64, 4, 8)
    B4 = spzeros(Float64, 4, 8)
end

matA4 = SafePETSc.Mat_sum(A4)
matB4 = SafePETSc.Mat_sum(B4)

# Test addition
matAdd = matA4 + matB4
@test matAdd isa SafeMPI.DRef

# Test subtraction
matSub = matA4 - matB4
@test matSub isa SafeMPI.DRef

# Verify correctness of subtraction
Sub_local = SafePETSc._mat_to_local_sparse(matSub)
Sub_sum = zeros(4, 8)
MPI.Reduce!(Matrix(Sub_local), Sub_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A4 - B4)
    @test all(isapprox.(Sub_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Operations with 2×5 matrices
if rank == 0
    println("[DEBUG] Test 5: Multiple operations with 2×5 matrices")
    flush(stdout)
end

# Only rank 0 contributes (using 2×5 non-square matrices)
if rank == 0
    A5 = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 5)  # 2×5 matrix
    B5 = sparse([1, 2], [2, 3], [3.0, 4.0], 2, 5)  # 2×5 matrix
else
    A5 = spzeros(Float64, 2, 5)
    B5 = spzeros(Float64, 2, 5)
end

matA5 = SafePETSc.Mat_sum(A5)
matB5 = SafePETSc.Mat_sum(B5)

# First addition
matC5a = matA5 + matB5
@test matC5a isa SafeMPI.DRef

# Verify values are correct
C5_local = SafePETSc._mat_to_local_sparse(matC5a)
C5_sum = zeros(2, 5)
MPI.Reduce!(Matrix(C5_local), C5_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A5 + B5)
    @test all(isapprox.(C5_sum, expected, atol=1e-10))
end

# Second addition (tests that multiple operations work correctly)
matC5b = matA5 + matB5
@test matC5b isa SafeMPI.DRef

# Verify values are still correct
C5b_local = SafePETSc._mat_to_local_sparse(matC5b)
C5b_sum = zeros(2, 5)
MPI.Reduce!(Matrix(C5b_local), C5b_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A5 + B5)
    @test all(isapprox.(C5b_sum, expected, atol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All non-square matrix addition/subtraction tests completed")
    flush(stdout)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix add/sub test file completed successfully")
    flush(stdout)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
