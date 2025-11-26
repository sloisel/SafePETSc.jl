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
    println("[DEBUG] spdiagm tests starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "spdiagm tests" begin

# Test 1: spdiagm with main diagonal only
if rank == 0
    println("[DEBUG] Test 1: spdiagm with main diagonal")
    flush(stdout)
end

diag_data = Float64.(1:5)
drdiag = SafePETSc.Vec_uniform(diag_data)

drA = spdiagm(0 => drdiag)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 5)

# Verify result by comparing with Julia's spdiagm
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(0 => diag_data)
A_sum = zeros(5, 5)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: spdiagm with main diagonal and superdiagonal
if rank == 0
    println("[DEBUG] Test 2: spdiagm with superdiagonal")
    flush(stdout)
end

diag_data = ones(Float64, 4)
super_data = ones(Float64, 3) * 2
drdiag = SafePETSc.Vec_uniform(diag_data)
drsuper = SafePETSc.Vec_uniform(super_data)

drA = spdiagm(0 => drdiag, 1 => drsuper)
@test drA isa SafeMPI.DRef
@test size(drA) == (4, 4)

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(0 => diag_data, 1 => super_data)
A_sum = zeros(4, 4)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: spdiagm with main diagonal and subdiagonal
if rank == 0
    println("[DEBUG] Test 3: spdiagm with subdiagonal")
    flush(stdout)
end

diag_data = ones(Float64, 4)
sub_data = ones(Float64, 3) * 3
drdiag = SafePETSc.Vec_uniform(diag_data)
drsub = SafePETSc.Vec_uniform(sub_data)

drA = spdiagm(0 => drdiag, -1 => drsub)
@test drA isa SafeMPI.DRef
@test size(drA) == (4, 4)

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(0 => diag_data, -1 => sub_data)
A_sum = zeros(4, 4)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: spdiagm with tridiagonal matrix
if rank == 0
    println("[DEBUG] Test 4: spdiagm tridiagonal")
    flush(stdout)
end

lower_data = ones(Float64, 4) * (-1.0)
diag_data = ones(Float64, 5) * 2.0
upper_data = ones(Float64, 4) * (-1.0)
drlower = SafePETSc.Vec_uniform(lower_data)
drdiag = SafePETSc.Vec_uniform(diag_data)
drupper = SafePETSc.Vec_uniform(upper_data)

drA = spdiagm(-1 => drlower, 0 => drdiag, 1 => drupper)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 5)

# Verify result (this creates a standard tridiagonal matrix)
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(-1 => lower_data, 0 => diag_data, 1 => upper_data)
A_sum = zeros(5, 5)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: spdiagm with explicit dimensions (rectangular)
if rank == 0
    println("[DEBUG] Test 5: spdiagm with explicit dimensions")
    flush(stdout)
end

diag_data = Float64.(1:3)
drdiag = SafePETSc.Vec_uniform(diag_data)

drA = spdiagm(5, 4, 0 => drdiag)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 4)

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(5, 4, 0 => diag_data)
A_sum = zeros(5, 4)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: spdiagm with multiple off-diagonals
if rank == 0
    println("[DEBUG] Test 6: spdiagm with multiple off-diagonals")
    flush(stdout)
end

d0 = ones(Float64, 4)
d1 = ones(Float64, 3) * 0.5
d2 = ones(Float64, 2) * 0.25
drd0 = SafePETSc.Vec_uniform(d0)
drd1 = SafePETSc.Vec_uniform(d1)
drd2 = SafePETSc.Vec_uniform(d2)

drA = spdiagm(0 => drd0, 1 => drd1, 2 => drd2)
@test drA isa SafeMPI.DRef
ref6 = spdiagm(0 => d0, 1 => d1, 2 => d2)
@test size(drA) == size(ref6)  # Match Julia's size inference

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = ref6
A_sum = zeros(size(A_julia)...) 
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: spdiagm with negative off-diagonals
if rank == 0
    println("[DEBUG] Test 7: spdiagm with negative off-diagonals")
    flush(stdout)
end

d0 = ones(Float64, 3)
dm1 = ones(Float64, 2) * 2
dm2 = ones(Float64, 1) * 3
drd0 = SafePETSc.Vec_uniform(d0)
drdm1 = SafePETSc.Vec_uniform(dm1)
drdm2 = SafePETSc.Vec_uniform(dm2)

drA = spdiagm(0 => drd0, -1 => drdm1, -2 => drdm2)
@test drA isa SafeMPI.DRef
ref7 = spdiagm(0 => d0, -1 => dm1, -2 => dm2)
@test size(drA) == size(ref7)  # Match Julia's size inference

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = ref7
A_sum = zeros(size(A_julia)...) 
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: spdiagm with different values on diagonals
if rank == 0
    println("[DEBUG] Test 8: spdiagm with varying diagonal values")
    flush(stdout)
end

diag_data = Float64.(1:6)
super_data = Float64.((1:5) .* 0.1)
sub_data = Float64.((1:5) .* (-0.1))
drdiag = SafePETSc.Vec_uniform(diag_data)
drsuper = SafePETSc.Vec_uniform(super_data)
drsub = SafePETSc.Vec_uniform(sub_data)

drA = spdiagm(-1 => drsub, 0 => drdiag, 1 => drsuper)
@test drA isa SafeMPI.DRef
@test size(drA) == (6, 6)

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(-1 => sub_data, 0 => diag_data, 1 => super_data)
A_sum = zeros(6, 6)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: prefix parameter - MPIDENSE Vec to MPIAIJ Mat
if rank == 0
    println("[DEBUG] Test 9: prefix MPIDENSE -> MPIAIJ")
    flush(stdout)
end

using SafePETSc: MPIAIJ, MPIDENSE
diag_data = Float64.(1:5)
drdiag = SafePETSc.Vec_uniform(diag_data)

# Create sparse matrix from vector using prefix parameter
drA = spdiagm(0 => drdiag; prefix=MPIAIJ)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 5)
@test typeof(drA.obj) == SafePETSc._Mat{Float64, MPIAIJ}

# Verify result is correct
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(0 => diag_data)
A_sum = zeros(5, 5)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 10: prefix parameter - MPIDENSE Vec to MPIDENSE Mat
if rank == 0
    println("[DEBUG] Test 10: prefix MPIDENSE -> MPIDENSE")
    flush(stdout)
end

drdiag = SafePETSc.Vec_uniform(diag_data)
drA = spdiagm(0 => drdiag; prefix=MPIDENSE)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 5)
@test typeof(drA.obj) == SafePETSc._Mat{Float64, MPIDENSE}

# Verify result is correct
A_local = SafePETSc._mat_to_local_sparse(drA)
A_sum = zeros(5, 5)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 11: prefix parameter with multiple diagonals
if rank == 0
    println("[DEBUG] Test 11: prefix with tridiagonal MPIDENSE -> MPIAIJ")
    flush(stdout)
end

lower_data = ones(Float64, 4) * (-1.0)
diag_data = ones(Float64, 5) * 2.0
upper_data = ones(Float64, 4) * (-1.0)
drlower = SafePETSc.Vec_uniform(lower_data)
drdiag = SafePETSc.Vec_uniform(diag_data)
drupper = SafePETSc.Vec_uniform(upper_data)

drA = spdiagm(-1 => drlower, 0 => drdiag, 1 => drupper; prefix=MPIAIJ)
@test drA isa SafeMPI.DRef
@test size(drA) == (5, 5)
@test typeof(drA.obj) == SafePETSc._Mat{Float64, MPIAIJ}

# Verify result
A_local = SafePETSc._mat_to_local_sparse(drA)
A_julia = spdiagm(-1 => lower_data, 0 => diag_data, 1 => upper_data)
A_sum = zeros(5, 5)
MPI.Reduce!(Matrix(A_local), A_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(A_sum, Matrix(A_julia), rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All spdiagm tests completed")
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
    println("Test Summary: spdiagm tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] spdiagm test file completed successfully")
    flush(stdout)
end

