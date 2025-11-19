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

if rank == 0
    println("[DEBUG] BlockProduct test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "BlockProduct tests" begin

# Test 1: Simple 2x2 block matrix multiplication with scalars
if rank == 0
    println("[DEBUG] Test 1: BlockProduct with scalar blocks")
    flush(stdout)
end

# Create simple block matrices with scalars
A = [1.0 2.0; 3.0 4.0]
B = [2.0 0.0; 0.0 2.0]

bp = BlockProduct([A, B])
result = calculate!(bp)

@test size(result) == (2, 2)
@test result[1,1] == 2.0  # 1*2 + 2*0
@test result[1,2] == 4.0  # 1*0 + 2*2
@test result[2,1] == 6.0  # 3*2 + 4*0
@test result[2,2] == 8.0  # 3*0 + 4*2

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Block matrix with PETSc Mat objects
if rank == 0
    println("[DEBUG] Test 2: BlockProduct with Mat blocks")
    flush(stdout)
end

# Create 2x2 block matrices where each block is a small PETSc Mat
M11 = SafePETSc.Mat_uniform([1.0 0.0; 0.0 1.0])  # 2x2 identity
M12 = SafePETSc.Mat_uniform([0.0 1.0; 1.0 0.0])  # 2x2 swap
M21 = SafePETSc.Mat_uniform([2.0 0.0; 0.0 2.0])  # 2x2 2*I
M22 = SafePETSc.Mat_uniform([1.0 1.0; 1.0 1.0])  # 2x2 ones

N11 = SafePETSc.Mat_uniform([1.0 0.0; 0.0 2.0])  # 2x2 diag(1,2)
N12 = SafePETSc.Mat_uniform([3.0 0.0; 0.0 3.0])  # 2x2 3*I
N21 = SafePETSc.Mat_uniform([0.0 0.0; 0.0 0.0])  # 2x2 zero
N22 = SafePETSc.Mat_uniform([1.0 0.0; 0.0 1.0])  # 2x2 identity

A_block = [M11 M12; M21 M22]
B_block = [N11 N12; N21 N22]

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (2, 2)
# result[1,1] = M11*N11 + M12*N21 = I*diag(1,2) + swap*0 = diag(1,2)
# result[1,2] = M11*N12 + M12*N22 = I*3I + swap*I = 3I + swap
# etc.

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Block matrix with Mat and Vec (matrix-vector product)
if rank == 0
    println("[DEBUG] Test 3: BlockProduct with Mat and Vec blocks")
    flush(stdout)
end

# Test Mat * Vec multiplication in block product
M = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
v = SafePETSc.Vec_uniform([1.0, 2.0])

# Single block matrices to test M * v
A_block = reshape([M], 1, 1)  # 1x1 block matrix containing M
B_block = reshape([v], 1, 1)  # 1x1 block matrix containing v

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (1, 1)
# result[1,1] = M*v which should be a Vec

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Scalar optimization - multiply by 0
if rank == 0
    println("[DEBUG] Test 4: Scalar optimization with zero")
    flush(stdout)
end

M = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
A_block = [M 0.0; 0.0 M]
B_block = [M 0.0; 0.0 M]

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (2, 2)
# result[1,1] = M*M + 0*0 = M*M
# result[1,2] = M*0 + 0*M = 0 (structural zero)
# result[2,1] = 0*M + M*0 = 0
# result[2,2] = 0*0 + M*M = M*M

@test result[1,2] == 0
@test result[2,1] == 0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Scalar optimization - multiply by 1
if rank == 0
    println("[DEBUG] Test 5: Scalar optimization with one")
    flush(stdout)
end

M = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])

if rank == 0
    println("[DEBUG] Test 5: Created M")
    flush(stdout)
end

A_block = [1.0 0.0; 0.0 1.0]
B_block = [M 2.0*M; 0.5*M M]

if rank == 0
    println("[DEBUG] Test 5: Created blocks, about to create BlockProduct")
    flush(stdout)
end

bp = BlockProduct([A_block, B_block])

if rank == 0
    println("[DEBUG] Test 5: Created BlockProduct, about to call calculate!")
    flush(stdout)
end

result = calculate!(bp)

if rank == 0
    println("[DEBUG] Test 5: Called calculate!, about to check size")
    flush(stdout)
end

@test size(result) == (2, 2)

if rank == 0
    println("[DEBUG] Test 5: Size check passed, about to call check_and_destroy!")
    flush(stdout)
end

# result[1,1] = 1*M + 0*0.5M = M
# result[1,2] = 1*2M + 0*M = 2M
# result[2,1] = 0*M + 1*0.5M = 0.5M
# result[2,2] = 0*2M + 1*M = M

SafeMPI.check_and_destroy!()

if rank == 0
    println("[DEBUG] Test 5: check_and_destroy! completed")
    flush(stdout)
end

MPI.Barrier(comm)

# Test 6: Three-way product
if rank == 0
    println("[DEBUG] Test 6: Three-way product A*B*C")
    flush(stdout)
end

# Simple scalar blocks for easy verification
A = [2.0 0.0; 0.0 2.0]
B = [1.0 1.0; 1.0 1.0]
C = [0.5 0.0; 0.0 0.5]

if rank == 0
    println("[DEBUG] Test 6: About to create BlockProduct")
    flush(stdout)
end

bp = BlockProduct([A, B, C])

if rank == 0
    println("[DEBUG] Test 6: Created BlockProduct, about to call calculate!")
    flush(stdout)
end

result = calculate!(bp)

if rank == 0
    println("[DEBUG] Test 6: Called calculate!, result type = $(typeof(result)), size = $(size(result))")
    println("[DEBUG] Test 6: result[1,1] type = $(typeof(result[1,1]))")
    flush(stdout)
end

@test size(result) == (2, 2)
# A*B = [2 2; 2 2]
# (A*B)*C = [2 2; 2 2] * [0.5 0; 0 0.5] = [1 1; 1 1]

if rank == 0
    println("[DEBUG] Test 6: About to check result[1,1]")
    flush(stdout)
end

@test result[1,1] == 1.0
@test result[1,2] == 1.0
@test result[2,1] == 1.0
@test result[2,2] == 1.0

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: Commented out - Vec' * Vec (inner product) needs special handling
# if rank == 0
#     println("[DEBUG] Test 7: Vec' * Vec inner product")
#     flush(stdout)
# end
#
# v1 = SafePETSc.Vec_uniform([1.0, 2.0, 3.0])
# v2 = SafePETSc.Vec_uniform([4.0, 5.0, 6.0])
#
# A_block = reshape([v1'], 1, 1)
# B_block = reshape([v2], 1, 1)
#
# bp = BlockProduct([A_block, B_block])
# result = calculate!(bp)
#
# @test size(result) == (1, 1)
# # v1' * v2 = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
#
# SafeMPI.check_and_destroy!()
# MPI.Barrier(comm)

# Test 8: Dimension validation
if rank == 0
    println("[DEBUG] Test 8: Dimension mismatch error")
    flush(stdout)
end

A = [1.0 2.0; 3.0 4.0]  # 2x2
B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]  # 3x3

try
    bp = BlockProduct([A, B])  # Should error - 2 != 3
    @test false  # Should not reach here
catch e
    @test e isa ErrorException
    @test occursin("Dimension mismatch", e.msg)
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: Repeated calculate! calls (reuse)
if rank == 0
    println("[DEBUG] Test 9: Multiple calculate! calls with cached reuse")
    flush(stdout)
end

M = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
A_block = [M M; M M]
B_block = [M M; M M]

# Constructor calls _calculate_init!, result is cached
bp = BlockProduct([A_block, B_block])
result1 = bp.result  # Get initial result

# Call calculate!() to recompute using cached objects
result2 = calculate!(bp)

@test size(result1) == size(result2)
# Both should produce the same result
# Verify result is cached (same reference)
@test result1 === result2

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 10: Mixed scalar and Mat multiplication
if rank == 0
    println("[DEBUG] Test 10: Mixed scalar and Mat")
    flush(stdout)
end

M = SafePETSc.Mat_uniform([1.0 0.0; 0.0 1.0])
A_block = [2.0 M; M 3.0]
B_block = [M 0.5; 2.0 M]

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (2, 2)
# result[1,1] = 2*M + M*2 = 2M + 2M = 4M
# result[1,2] = 2*0.5 + M*M = 1 + M
# result[2,1] = M*M + 3*2 = M + 6
# result[2,2] = M*0.5 + 3*M = 0.5M + 3M = 3.5M

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 11: Modify input and recalculate
if rank == 0
    println("[DEBUG] Test 11: Modify input and recalculate")
    flush(stdout)
end

# Create simple scalar blocks for easy verification
A = [2.0 0.0; 0.0 2.0]  # 2*I
B = [1.0 0.0; 0.0 1.0]  # I

bp = BlockProduct([A, B])
result1 = bp.result

# Initial result should be 2*I * I = 2*I
@test result1[1,1] == 2.0
@test result1[2,2] == 2.0

# Modify input matrix B to be 3*I
bp.prod[2][1,1] = 3.0
bp.prod[2][2,2] = 3.0

# Recalculate
result2 = calculate!(bp)

# Result should now be 2*I * 3*I = 6*I
@test result2[1,1] == 6.0
@test result2[2,2] == 6.0

# Verify we got the same cached object
@test result1 === result2

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 12: Block product with A' * B
if rank == 0
    println("[DEBUG] Test 12: BlockProduct with A' * B")
    flush(stdout)
end

M1 = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
M2 = SafePETSc.Mat_uniform([2.0 0.0; 0.0 2.0])

A_block = reshape([M1'], 1, 1)  # A'
B_block = reshape([M2], 1, 1)   # B

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (1, 1)
@test result[1,1] isa SafePETSc.Mat

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 13: Block product with A * B'
if rank == 0
    println("[DEBUG] Test 13: BlockProduct with A * B'")
    flush(stdout)
end

M3 = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
M4 = SafePETSc.Mat_uniform([2.0 0.0; 0.0 2.0])

A_block = reshape([M3], 1, 1)    # A
B_block = reshape([M4'], 1, 1)   # B'

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (1, 1)
@test result[1,1] isa SafePETSc.Mat

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 14: Block product with A' * B'
if rank == 0
    println("[DEBUG] Test 14: BlockProduct with A' * B'")
    flush(stdout)
end

M5 = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0])
M6 = SafePETSc.Mat_uniform([2.0 0.0; 0.0 2.0])

A_block = reshape([M5'], 1, 1)   # A'
B_block = reshape([M6'], 1, 1)   # B'

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (1, 1)
@test result[1,1] isa SafePETSc.Mat

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 15: Mixed block with all transpose combinations
if rank == 0
    println("[DEBUG] Test 15: BlockProduct with mixed Mat and Mat' blocks")
    flush(stdout)
end

M7 = SafePETSc.Mat_uniform([1.0 0.0; 0.0 1.0])  # Identity
M8 = SafePETSc.Mat_uniform([0.0 1.0; 1.0 0.0])  # Swap matrix

# Block matrix with all combinations: [M7   M8']
#                                     [M7'  M8 ]
# Note: Use reshape instead of [...] syntax to avoid Adjoint dimension mismatch issues
A_block = reshape([M7, M7', M8', M8], 2, 2)
B_block = reshape([M7, M7, M7, M7], 2, 2)

bp = BlockProduct([A_block, B_block])
result = calculate!(bp)

@test size(result) == (2, 2)
# Verify all results are Mat objects (not scalars)
@test result[1,1] isa SafePETSc.Mat
@test result[1,2] isa SafePETSc.Mat
@test result[2,1] isa SafePETSc.Mat
@test result[2,2] isa SafePETSc.Mat

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

end  # end testset

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
    println("Test Summary: BlockProduct tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] BlockProduct test file completed successfully")
    flush(stdout)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
