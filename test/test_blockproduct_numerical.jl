using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
using Random
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

if !PETSc.initialized(PETSc.petsclibs[1])
    PETSc.initialize()
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] BlockProduct numerical accuracy test starting")
    flush(stdout)
end

ts = @testset MPITestHarness.QuietTestSet "BlockProduct numerical accuracy" begin

# Test 1: Simple case - 3 random rectangular matrices (scalar elements)
if rank == 0
    println("[DEBUG] Test 1: A*B*C with random rectangular matrices (scalar elements)")
    flush(stdout)
end

Random.seed!(42)  # For reproducibility
A_full = randn(5, 7)
B_full = randn(7, 4)
C_full = randn(4, 6)

# Compute reference result with pure Julia
reference = A_full * B_full * C_full

# Create block matrices where each block is a scalar (Number)
# Treat each element as a 1x1 block
A_blocks = A_full
B_blocks = B_full
C_blocks = C_full

# Compute with BlockProduct - this will multiply element-wise and is NOT what we want!
# Actually, when blocks are scalars, BlockProduct treats them as block matrices where
# each scalar is a 1x1 block, so matrix multiplication becomes:
# result[i,k] = sum_j A[i,j] * B[j,k]
# This is exactly standard matrix multiplication!

bp = BlockProduct([A_blocks, B_blocks, C_blocks])
result = calculate!(bp)

@test size(result) == size(reference)
@test result ≈ reference rtol=1e-10

MPI.Barrier(comm)

# Test 2: Block structure with scalars - verify block multiplication formula
# Test that (A*B)*C where blocks are scalars follows correct block matrix multiplication
if rank == 0
    println("[DEBUG] Test 2: Block matrix multiplication with different sizes")
    flush(stdout)
end

Random.seed!(123)
# Create 4x6 * 6x3 * 3x5 = 4x5 result
A = randn(4, 6)
B = randn(6, 3)
C = randn(3, 5)

reference = A * B * C

bp = BlockProduct([A, B, C])
result = calculate!(bp)

@test size(result) == size(reference)
@test result ≈ reference rtol=1e-10

MPI.Barrier(comm)

# Test 3: Four matrix product
if rank == 0
    println("[DEBUG] Test 3: Four matrix product A*B*C*D")
    flush(stdout)
end

Random.seed!(456)
A = randn(3, 7)
B = randn(7, 5)
C = randn(5, 6)
D = randn(6, 4)

reference = A * B * C * D

bp = BlockProduct([A, B, C, D])
result = calculate!(bp)

@test size(result) == size(reference)
@test result ≈ reference rtol=1e-10

MPI.Barrier(comm)

# Test 4: Two matrix product (A*B)
if rank == 0
    println("[DEBUG] Test 4: Two matrix product A*B")
    flush(stdout)
end

Random.seed!(789)
A = randn(8, 11)
B = randn(11, 6)

reference = A * B

bp = BlockProduct([A, B])
result = calculate!(bp)

@test size(result) == size(reference)
@test result ≈ reference rtol=1e-10

MPI.Barrier(comm)

# Test 5: Non-square matrices with various aspect ratios
if rank == 0
    println("[DEBUG] Test 5: Tall and wide matrices")
    flush(stdout)
end

Random.seed!(999)
A_tall = randn(20, 5)   # Tall matrix
B_square = randn(5, 5)   # Square matrix
C_wide = randn(5, 15)    # Wide matrix

reference = A_tall * B_square * C_wide

bp = BlockProduct([A_tall, B_square, C_wide])
result = calculate!(bp)

@test size(result) == (20, 15)
@test result ≈ reference rtol=1e-10

MPI.Barrier(comm)

end  # testset

# Aggregate counts across ranks
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
    println("Test Summary: BlockProduct numerical accuracy (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
    println("[DEBUG] BlockProduct numerical accuracy test completed successfully")
    flush(stdout)
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
