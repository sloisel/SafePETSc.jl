using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Block diagonal matrix tests starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Block diagonal matrix tests" begin

# Test 1: blockdiag with two square matrices
if rank == 0
    println("[DEBUG] Test 1: blockdiag with two 3x3 matrices")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 3, 3)
B_data = ones(Float64, 3, 3) * 2
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)

drC = blockdiag(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (6, 6)

# Verify result by comparing with Julia's blockdiag
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = blockdiag(sparse(A_data), sparse(B_data))
C_sum = zeros(6, 6)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, Matrix(C_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: blockdiag with three matrices
if rank == 0
    println("[DEBUG] Test 2: blockdiag with three matrices")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 2, 2)
B_data = ones(Float64, 3, 3) * 2
C_data = reshape(Float64.(1:12), 4, 3)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)
drC = SafePETSc.Mat_uniform(C_data)

drD = blockdiag(drA, drB, drC)
@test drD isa SafeMPI.DRef
@test size(drD) == (9, 8)  # (2+3+4) x (2+3+3)

# Verify result
D_local = SafePETSc._mat_to_local_sparse(drD)
D_julia = blockdiag(sparse(A_data), sparse(B_data), sparse(C_data))
D_sum = zeros(9, 8)
MPI.Reduce!(Matrix(D_local), D_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(D_sum, Matrix(D_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: blockdiag with rectangular matrices
if rank == 0
    println("[DEBUG] Test 3: blockdiag with rectangular matrices")
    flush(stdout)
end

A_data = reshape(Float64.(1:6), 2, 3)
B_data = reshape(Float64.(7:18), 3, 4)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)

drC = blockdiag(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (5, 7)  # (2+3) x (3+4)

# Verify result
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = blockdiag(sparse(A_data), sparse(B_data))
C_sum = zeros(5, 7)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, Matrix(C_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: blockdiag with sparse matrices
if rank == 0
    println("[DEBUG] Test 4: blockdiag with sparse matrices")
    flush(stdout)
end

A_data = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
B_data = sparse([1, 3], [2, 3], [3.0, 4.0], 3, 3)
drA = SafePETSc.Mat_uniform(Matrix(A_data))
drB = SafePETSc.Mat_uniform(Matrix(B_data))

drC = blockdiag(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (5, 5)

# Verify result
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = blockdiag(A_data, B_data)
C_sum = zeros(5, 5)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, Matrix(C_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: blockdiag with single matrix
if rank == 0
    println("[DEBUG] Test 5: blockdiag with single matrix")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4)
drA = SafePETSc.Mat_uniform(A_data)

drB = blockdiag(drA)
@test drB isa SafeMPI.DRef
@test size(drB) == (4, 4)

# Verify result (should be same as input)
B_local = SafePETSc._mat_to_local_sparse(drB)
B_sum = zeros(4, 4)
MPI.Reduce!(Matrix(B_local), B_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(B_sum, A_data, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: blockdiag with four matrices of different sizes
if rank == 0
    println("[DEBUG] Test 6: blockdiag with four different-sized matrices")
    flush(stdout)
end

A_data = ones(Float64, 2, 3) * 1.0
B_data = ones(Float64, 1, 1) * 2.0
C_data = ones(Float64, 3, 2) * 3.0
D_data = ones(Float64, 2, 4) * 4.0
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)
drC = SafePETSc.Mat_uniform(C_data)
drD = SafePETSc.Mat_uniform(D_data)

drE = blockdiag(drA, drB, drC, drD)
@test drE isa SafeMPI.DRef
@test size(drE) == (8, 10)  # (2+1+3+2) x (3+1+2+4)

# Verify result
E_local = SafePETSc._mat_to_local_sparse(drE)
E_julia = blockdiag(sparse(A_data), sparse(B_data), sparse(C_data), sparse(D_data))
E_sum = zeros(8, 10)
MPI.Reduce!(Matrix(E_local), E_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(E_sum, Matrix(E_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All blockdiag tests completed")
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
    println("Test Summary: Block diagonal matrix tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Block diagonal matrix test file completed successfully")
    flush(stdout)
end

