using Test
using MPI

# Use unified initializer (MPI then PETSc)
using SafePETSc
SafePETSc.Init()

# Now load PETSc explicitly for helpers used in tests
using SafePETSc
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc must be initialized by user code (library does not auto-init)
if !PETSc.initialized(PETSc.petsclibs[1])
    PETSc.initialize()
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Configure PETSc options: matrices with prefix "dense_" will be created as mpidense
SafePETSc.petsc_options_insert_string("-dense_mat_type mpidense")

if rank == 0
    println("[DEBUG] Mat operations test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Matrix operations tests" begin

# Test 1: Matrix-matrix multiplication A*B
if rank == 0
    println("[DEBUG] Test 1: Matrix-matrix multiplication")
    flush(stdout)
end

A_data = reshape(Float64.(1:16), 4, 4)
B_data = reshape(Float64.(1:16), 4, 4)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data)

drC = drA * drB
@test drC isa SafeMPI.DRef
@test size(drC) == (4, 4)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 16: In-place linear solve with matrix RHS ldiv!(X, A, B)
# Test 2: Matrix-vector multiplication A*x
if rank == 0
    println("[DEBUG] Test 2: Matrix-vector multiplication")
    flush(stdout)
end

A_data = reshape(Float64.(1:16), 4, 4)
x_data = Float64.(1:4)
drA = SafePETSc.Mat_uniform(A_data)
drx = SafePETSc.Vec_uniform(x_data)

dry = drA * drx
@test dry isa SafeMPI.DRef
@test size(dry) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Adjoint vector times matrix v'*A
if rank == 0
    println("[DEBUG] Test 3: Adjoint vector times matrix")
    flush(stdout)
end

v_data = Float64.(1:4)
A_data = reshape(Float64.(1:16), 4, 4)
drv = SafePETSc.Vec_uniform(v_data)
drA = SafePETSc.Mat_uniform(A_data)

drw_adj = drv' * drA
@test drw_adj isa LinearAlgebra.Adjoint

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Linear solve A\b (vector RHS)
if rank == 0
    println("[DEBUG] Test 4: Linear solve A\\b")
    flush(stdout)
end

# Create a simple diagonal matrix for easy solve
A_data = Matrix{Float64}(I, 4, 4) * 2.0  # 2*I
b_data = Float64.(1:4)
drA = SafePETSc.Mat_uniform(A_data)
drb = SafePETSc.Vec_uniform(b_data)

drx = drA \ drb
@test drx isa SafeMPI.DRef
@test size(drx) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Linear solve A\B (matrix RHS)
if rank == 0
    println("[DEBUG] Test 5: Linear solve A\\B")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
B_data = reshape(Float64.(1:8), 4, 2)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")  # Create as dense via prefix option

drX = drA \ drB
@test drX isa SafeMPI.DRef
@test size(drX) == (4, 2)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: Transpose solve A'\b
if rank == 0
    println("[DEBUG] Test 6: Transpose solve A'\\b")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
b_data = Float64.(1:4)
drA = SafePETSc.Mat_uniform(A_data)
drb = SafePETSc.Vec_uniform(b_data)

drx = drA' \ drb
@test drx isa SafeMPI.DRef
@test size(drx) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: Transpose solve A'\B
if rank == 0
    println("[DEBUG] Test 7: Transpose solve A'\\B")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
B_data = reshape(Float64.(1:8), 4, 2)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")  # Create as dense via prefix option

drX = drA' \ drB
@test drX isa SafeMPI.DRef
@test size(drX) == (4, 2)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: Right division b'/A
if rank == 0
    println("[DEBUG] Test 8: Right division b'/A")
    flush(stdout)
end

b_data = Float64.(1:4)
A_data = Matrix{Float64}(I, 4, 4) * 2.0
drb = SafePETSc.Vec_uniform(b_data)
drA = SafePETSc.Mat_uniform(A_data)

drx_adj = drb' / drA
@test drx_adj isa LinearAlgebra.Adjoint

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: Right division B/A
if rank == 0
    println("[DEBUG] Test 9: Right division B/A")
    flush(stdout)
end

B_data = reshape(Float64.(1:8), 2, 4)
A_data = Matrix{Float64}(I, 4, 4) * 2.0
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")  # Create as dense via prefix option
drA = SafePETSc.Mat_uniform(A_data)

drX = drB / drA
@test drX isa SafeMPI.DRef
@test size(drX) == (2, 4)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 10: Right division B/A'
if rank == 0
    println("[DEBUG] Test 10: Right division B/A'")
    flush(stdout)
end

B_data = reshape(Float64.(1:8), 2, 4)
A_data = Matrix{Float64}(I, 4, 4) * 2.0
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")  # Create as dense via prefix option
drA = SafePETSc.Mat_uniform(A_data)

drX = drB / drA'
@test drX isa SafeMPI.DRef
@test size(drX) == (2, 4)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 11: Solver constructor with prefix
if rank == 0
    println("[DEBUG] Test 11: Solver constructor")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
drA = SafePETSc.Mat_uniform(A_data)

ksp = SafePETSc.Solver(drA; prefix="test_")
@test ksp isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# ==================== IN-PLACE FUNCTION TESTS ====================

# Test 12: In-place matrix-vector multiply mul!(y, A, x)
if rank == 0
    println("[DEBUG] Test 12: In-place matrix-vector multiply mul!(y, A, x)")
    flush(stdout)
end

A_data = reshape(Float64.(1:16), 4, 4)
x_data = Float64.(1:4)
drA = SafePETSc.Mat_uniform(A_data)
drx = SafePETSc.Vec_uniform(x_data)

# Pre-allocate output vector
dry = SafePETSc.Vec_uniform(zeros(4))

# Call in-place version
result = mul!(dry, drA, drx)

# Verify it returns the same object and has correct size
@test result === dry
@test size(dry) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# NOTE: Tests for mul!(C, A, B) and transpose!(B, A) are skipped because
# PETSc's MAT_REUSE_MATRIX requires very specific structure setup that is
# difficult to test robustly. These functions work but require the user to
# ensure the output matrix has the correct pre-computed structure.
# Users should call A*B first to set up C, then mul!(C, A, B) for subsequent
# computations with the same structure.

# Test 13: In-place adjoint vector-matrix multiply mul!(w, v', A)
if rank == 0
    println("[DEBUG] Test 13: In-place adjoint vector-matrix multiply mul!(w, v', A)")
    flush(stdout)
end

v_data = Float64.(1:4)
A_data = reshape(Float64.(1:16), 4, 4)
drv = SafePETSc.Vec_uniform(v_data)
drA = SafePETSc.Mat_uniform(A_data)

# Pre-allocate output vector (should have length matching A's column count)
drw_preallocated = SafePETSc.Vec_uniform(zeros(4))

# Call in-place version
result = mul!(drw_preallocated, drv', drA)

# Verify it returns the same object and has correct size
@test result === drw_preallocated
@test size(drw_preallocated) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 14: In-place linear solve ldiv!(x, A, b)
if rank == 0
    println("[DEBUG] Test 14: In-place linear solve ldiv!(x, A, b)")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0  # 2*I
b_data = Float64.(1:4)
drA = SafePETSc.Mat_uniform(A_data)
drb = SafePETSc.Vec_uniform(b_data)

# Pre-allocate output vector
drx_preallocated = SafePETSc.Vec_uniform(zeros(4))

# Call in-place version
result = ldiv!(drx_preallocated, drA, drb)

# Verify it returns the same object and has correct size
@test result === drx_preallocated
@test size(drx_preallocated) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 15: In-place linear solve with solver reuse ldiv!(ksp, x, b)
if rank == 0
    println("[DEBUG] Test 15: In-place linear solve with solver reuse ldiv!(ksp, x, b)")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0  # 2*I
b1_data = Float64.(1:4)
b2_data = Float64.(5:8)
drA = SafePETSc.Mat_uniform(A_data)
drb1 = SafePETSc.Vec_uniform(b1_data)
drb2 = SafePETSc.Vec_uniform(b2_data)

# Create solver (can be reused)
ksp = SafePETSc.Solver(drA)

# Pre-allocate output vector
drx = SafePETSc.Vec_uniform(zeros(4))

# First solve
result1 = ldiv!(ksp, drx, drb1)
@test result1 === drx

# Second solve with different RHS (reuses solver and output vector)
result2 = ldiv!(ksp, drx, drb2)
@test result2 === drx

# Just verify the solve completed without errors
@test size(drx) == (4,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 16: In-place linear solve with matrix RHS ldiv!(X, A, B)
if rank == 0
    println("[DEBUG] Test 16: In-place linear solve with matrix RHS ldiv!(X, A, B)")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
B_data = reshape(Float64.(1:8), 4, 2)
drA = SafePETSc.Mat_uniform(A_data)
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")  # Must be dense

# Pre-allocate output matrix (must be dense)
drX_preallocated = SafePETSc.Mat_uniform(zeros(4, 2); prefix="dense_")

# Call in-place version
result = ldiv!(drX_preallocated, drA, drB)

# Verify it returns the same object
@test result === drX_preallocated
@test size(drX_preallocated) == (4, 2)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 17: In-place linear solve with matrix RHS and solver reuse ldiv!(ksp, X, B)
if rank == 0
    println("[DEBUG] Test 17: In-place linear solve with matrix RHS and solver reuse ldiv!(ksp, X, B)")
    flush(stdout)
end

A_data = Matrix{Float64}(I, 4, 4) * 2.0
B1_data = reshape(Float64.(1:8), 4, 2)
B2_data = reshape(Float64.(9:16), 4, 2)
drA = SafePETSc.Mat_uniform(A_data)
drB1 = SafePETSc.Mat_uniform(B1_data; prefix="dense_")
drB2 = SafePETSc.Mat_uniform(B2_data; prefix="dense_")

# Create solver (can be reused)
ksp = SafePETSc.Solver(drA)

# Pre-allocate output matrix (must be dense)
drX = SafePETSc.Mat_uniform(zeros(4, 2); prefix="dense_")

# First solve
result1 = ldiv!(ksp, drX, drB1)
@test result1 === drX

# Second solve with different RHS (reuses solver and output matrix)
result2 = ldiv!(ksp, drX, drB2)
@test result2 === drX

# Verify dimensions
@test size(drX) == (4, 2)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 18: In-place matrix-matrix multiply with structure reuse mul!(C, A, B)
if rank == 0
    println("[DEBUG] Test 18: In-place mat-mat mul! with structure reuse")
    flush(stdout)
end

# Use dense prefix to ensure MPIDENSE structure which is compatible with MAT_REUSE_MATRIX
A_data = reshape(Float64.(1:16), 4, 4)
B_data = reshape(Float64.(1:16), 4, 4)
drA = SafePETSc.Mat_uniform(A_data; prefix="dense_")
drB = SafePETSc.Mat_uniform(B_data; prefix="dense_")

# First compute out-of-place C0 = A*B (MAT_INITIAL_MATRIX inside)
drC0 = drA * drB

# Reuse the same output matrix to guarantee matching PETSc structure
result = mul!(drC0, drA, drB)
@test result === drC0
@test size(drC0) == (4, 4)

# Verify equality with a fresh out-of-place result
C_ref = drA * drB
C0_local = SafePETSc._mat_to_local_sparse(drC0)
Cref_local = SafePETSc._mat_to_local_sparse(C_ref)
C0_sum = zeros(4, 4)
Cref_sum  = zeros(4, 4)
MPI.Reduce!(Matrix(C0_local), C0_sum, +, 0, comm)
MPI.Reduce!(Matrix(Cref_local),  Cref_sum,  +, 0, comm)
if rank == 0
    @test all(isapprox.(C0_sum, Cref_sum, rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 19: In-place transpose! with structure reuse transpose!(B, A)
if rank == 0
    println("[DEBUG] Test 19: In-place transpose! with structure reuse")
    flush(stdout)
end

# Use dense prefix to ensure MPIDENSE structure which is compatible with MAT_REUSE_MATRIX
M, N = 4, 6  # rectangular to exercise dimension swap
A_data = reshape(Float64.(1:(M*N)), M, N)
drA = SafePETSc.Mat_uniform(A_data; prefix="dense_")

# Use the new constructor to materialize the adjoint: B = Mat(A')
drB = Mat(drA')

# First transpose into preallocated B
transpose!(drB, drA)

# Compare B with A'
B_local = SafePETSc._mat_to_local_sparse(drB)
B_sum = zeros(N, M)
MPI.Reduce!(Matrix(B_local), B_sum, +, 0, comm)
if rank == 0
    @test all(isapprox.(B_sum, A_data', rtol=1e-10))
end

# Modify A values without changing sparsity and transpose again, reusing B
PETSc.@chk ccall((:MatScale, PETSc.libpetsc), PETSc.PetscErrorCode,
                 (PETSc.CMat, Float64), drA.obj.A, 2.0)
transpose!(drB, drA)

# Validate B now equals (2A)'
B_local2 = SafePETSc._mat_to_local_sparse(drB)
B_sum2 = zeros(N, M)
MPI.Reduce!(Matrix(B_local2), B_sum2, +, 0, comm)
if rank == 0
    @test all(isapprox.(B_sum2, (2 .* A_data)', rtol=1e-10))
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All tests completed")
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
    println("Test Summary: Matrix operations tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Mat operations test file completed successfully")
    flush(stdout)
end
