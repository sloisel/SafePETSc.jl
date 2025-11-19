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
    println("[DEBUG] Matrix concatenation tests starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Matrix concatenation tests" begin

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

# Test 7: vcat with Vec objects (returns another Vec, not a Mat)
if rank == 0
    println("[DEBUG] Test 7: vcat with Vec objects")
    flush(stdout)
end

# Create vectors
v1_data = Float64[1, 2, 3, 4]
v2_data = Float64[5, 6, 7, 8]
v3_data = Float64[9, 10, 11, 12]
drV1 = SafePETSc.Vec_uniform(v1_data)
drV2 = SafePETSc.Vec_uniform(v2_data)
drV3 = SafePETSc.Vec_uniform(v3_data)

# Test vcat of vectors (should produce another Vec)
drV_vcat = vcat(drV1, drV2, drV3)
@test drV_vcat isa SafePETSc.Vec
@test length(drV_vcat) == 12

# Verify result by getting local array
V_local = PETSc.unsafe_localarray(drV_vcat.obj.v; read=true)
V_julia = vcat(v1_data, v2_data, v3_data)
# Each rank has part of the vector, so we need to gather
V_gathered = zeros(12)
local_range = drV_vcat.obj.row_partition[rank+1]:(drV_vcat.obj.row_partition[rank+2]-1)
V_local_portion = zeros(12)
V_local_portion[local_range] = V_local
MPI.Reduce!(V_local_portion, V_gathered, +, 0, comm)
if rank == 0
    @test all(isapprox.(V_gathered, V_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: hcat with Vec objects (tests _mat_to_local_sparse for Vec)
if rank == 0
    println("[DEBUG] Test 8: hcat with Vec objects")
    flush(stdout)
end

# Create vectors
v1_data = Float64[1, 2, 3]
v2_data = Float64[4, 5, 6]
v3_data = Float64[7, 8, 9]
drV1 = SafePETSc.Vec_uniform(v1_data)
drV2 = SafePETSc.Vec_uniform(v2_data)
drV3 = SafePETSc.Vec_uniform(v3_data)

# Test hcat of vectors (should produce a matrix with multiple columns)
drV_hcat = hcat(drV1, drV2, drV3)
@test drV_hcat isa SafePETSc.Mat  # hcat of vectors produces a Mat
@test size(drV_hcat) == (3, 3)

# Verify result
V_local = SafePETSc._mat_to_local_sparse(drV_hcat)
V_julia = hcat(v1_data, v2_data, v3_data)
V_sum = zeros(3, 3)
MPI.Reduce!(Matrix(V_local), V_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(V_sum, V_julia, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: Mixed Vec and Mat concatenation
if rank == 0
    println("[DEBUG] Test 9: Mixed Vec and Mat concatenation")
    flush(stdout)
end

# Create a vector and a matrix
v_data = Float64[1, 2, 3]
A_data = Float64[4 7; 5 8; 6 9]
drV = SafePETSc.Vec_uniform(v_data)
drA = SafePETSc.Mat_uniform(A_data)

# Test hcat of vector and matrix
drMixed = hcat(drV, drA)
@test drMixed isa SafePETSc.Mat  # Mixed hcat produces a Mat
@test size(drMixed) == (3, 3)

# Verify result
Mixed_local = SafePETSc._mat_to_local_sparse(drMixed)
Mixed_julia = hcat(v_data, A_data)
Mixed_sum = zeros(3, 3)
MPI.Reduce!(Matrix(Mixed_local), Mixed_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(Mixed_sum, Mixed_julia, rtol=1e-10))
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
MPI.Allreduce!(local_counts, global_counts, +, comm)

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

