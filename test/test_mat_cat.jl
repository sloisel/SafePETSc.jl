using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Matrix concatenation tests starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Matrix concatenation tests" begin

# Test 1: vcat - vertical concatenation
if rank == 0
    println("[DEBUG] Test 1: vcat")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4)
B_data = reshape(Float64.(1:16), 4, 4)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)

drC = vcat(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (8, 4)

# Verify result by comparing with Julia's vcat
# Extract local portions and sum across ranks to reconstruct full matrix
C_local = SafePETSc._mat_to_local_sparse(drC)
# Convert to dense for easier comparison
C_julia = vcat(A_data, B_data)
C_sum = zeros(8, 4)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, C_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: hcat - horizontal concatenation
if rank == 0
    println("[DEBUG] Test 2: hcat")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4)
B_data = reshape(Float64.(1:16), 4, 4)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)

drC = hcat(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (4, 8)

# Verify result
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = hcat(A_data, B_data)
C_sum = zeros(4, 8)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, C_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: cat with dims=1 (same as vcat)
if rank == 0
    println("[DEBUG] Test 3: cat with dims=1")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 3, 3)
B_data = ones(Float64, 3, 3) * 2
C_data = ones(Float64, 3, 3) * 3
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)
drC = SafePETSc.Mat_uniform(C_data)

drD = cat(drA, drB, drC; dims=1)
@test drD isa SafeMPI.DRef
@test size(drD) == (9, 3)

# Verify result
D_local = SafePETSc._mat_to_local_sparse(drD)
D_julia = cat(A_data, B_data, C_data; dims=1)
D_sum = zeros(9, 3)
MPI.Reduce!(Matrix(D_local), D_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(D_sum, D_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: cat with dims=2 (same as hcat)
if rank == 0
    println("[DEBUG] Test 4: cat with dims=2")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 3, 3)
B_data = ones(Float64, 3, 3) * 2
C_data = ones(Float64, 3, 3) * 3
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)
drC = SafePETSc.Mat_uniform(C_data)

drD = cat(drA, drB, drC; dims=2)
@test drD isa SafeMPI.DRef
@test size(drD) == (3, 9)

# Verify result
D_local = SafePETSc._mat_to_local_sparse(drD)
D_julia = cat(A_data, B_data, C_data; dims=2)
D_sum = zeros(3, 9)
MPI.Reduce!(Matrix(D_local), D_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(D_sum, D_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: vcat with different matrix values
if rank == 0
    println("[DEBUG] Test 5: vcat with sparse matrices")
    flush(stdout)
end

# Create sparse matrices (identical on all ranks); use Mat_uniform on dense copies
A_data = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 3)
B_data = sparse([1, 2], [2, 3], [3.0, 4.0], 2, 3)
drA = SafePETSc.Mat_uniform(Matrix(A_data))
drB = SafePETSc.Mat_uniform(Matrix(B_data))

drC = vcat(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (4, 3)

# Verify result
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = vcat(Matrix(A_data), Matrix(B_data))
C_sum = zeros(4, 3)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, C_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: hcat with different matrix values
if rank == 0
    println("[DEBUG] Test 6: hcat with sparse matrices")
    flush(stdout)
end

A_data = sparse([1, 2], [1, 2], [1.0, 2.0], 3, 2)
B_data = sparse([1, 2], [1, 2], [3.0, 4.0], 3, 2)
drA = SafePETSc.Mat_uniform(Matrix(A_data))
drB = SafePETSc.Mat_uniform(Matrix(B_data))

drC = hcat(drA, drB)
@test drC isa SafeMPI.DRef
@test size(drC) == (3, 4)

# Verify result
C_local = SafePETSc._mat_to_local_sparse(drC)
C_julia = hcat(Matrix(A_data), Matrix(B_data))
C_sum = zeros(3, 4)
MPI.Reduce!(Matrix(C_local), C_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(C_sum, C_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All concatenation tests completed")
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
MPI.Reduce!(local_counts, global_counts, +, 0, comm)

if rank == 0
    println("Test Summary: Matrix concatenation tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix concatenation test file completed successfully")
    flush(stdout)
end
