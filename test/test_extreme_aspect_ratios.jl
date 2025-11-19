using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Extreme aspect ratio tests starting")
    flush(stdout)
end

ts = @testset MPITestHarness.QuietTestSet "Extreme aspect ratio tests" begin

# Test 1: Very wide matrix (2×20)
if rank == 0
    println("[DEBUG] Test 1: Very wide matrix 2×20")
    flush(stdout)
end

A_wide = reshape(Float64.(1:40), 2, 20)  # 2 rows, 20 columns
drA_wide = SafePETSc.Mat_uniform(A_wide)
@test size(drA_wide) == (2, 20)

# Test multiplication with appropriate dimensions
x_wide = Float64.(1:20)
drx_wide = SafePETSc.Vec_uniform(x_wide)
dry_wide = drA_wide * drx_wide
@test size(dry_wide) == (2,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Very tall matrix (20×2)
if rank == 0
    println("[DEBUG] Test 2: Very tall matrix 20×2")
    flush(stdout)
end

A_tall = reshape(Float64.(1:40), 20, 2)  # 20 rows, 2 columns
drA_tall = SafePETSc.Mat_uniform(A_tall)
@test size(drA_tall) == (20, 2)

# Test multiplication with appropriate dimensions
x_tall = Float64.([1.0, 2.0])
drx_tall = SafePETSc.Vec_uniform(x_tall)
dry_tall = drA_tall * drx_tall
@test size(dry_tall) == (20,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: Single row matrix (1×10)
if rank == 0
    println("[DEBUG] Test 3: Single row matrix 1×10")
    flush(stdout)
end

A_row = reshape(Float64.(1:10), 1, 10)  # 1 row, 10 columns
drA_row = SafePETSc.Mat_uniform(A_row)
@test size(drA_row) == (1, 10)

# Test multiplication
x_row = Float64.(1:10)
drx_row = SafePETSc.Vec_uniform(x_row)
dry_row = drA_row * drx_row
@test size(dry_row) == (1,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Single column matrix (10×1)
if rank == 0
    println("[DEBUG] Test 4: Single column matrix 10×1")
    flush(stdout)
end

A_col = reshape(Float64.(1:10), 10, 1)  # 10 rows, 1 column
drA_col = SafePETSc.Mat_uniform(A_col)
@test size(drA_col) == (10, 1)

# Test multiplication
x_col = Float64.([2.0])
drx_col = SafePETSc.Vec_uniform(x_col)
dry_col = drA_col * drx_col
@test size(dry_col) == (10,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Product of extreme aspect ratios (2×15) * (15×3) = (2×3)
if rank == 0
    println("[DEBUG] Test 5: Product of extreme aspect ratios")
    flush(stdout)
end

A_prod1 = reshape(Float64.(1:30), 2, 15)   # Very wide: 2×15
B_prod1 = reshape(Float64.(1:45), 15, 3)   # Tall: 15×3
drA_prod1 = SafePETSc.Mat_uniform(A_prod1)
drB_prod1 = SafePETSc.Mat_uniform(B_prod1)
drC_prod1 = drA_prod1 * drB_prod1
@test size(drC_prod1) == (2, 3)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: Transpose of extreme aspect ratio
if rank == 0
    println("[DEBUG] Test 6: Transpose of extreme aspect ratio")
    flush(stdout)
end

A_trans = reshape(Float64.(1:60), 3, 20)  # 3×20 matrix
drA_trans = SafePETSc.Mat_uniform(A_trans)
drAt = drA_trans'
@test size(drAt) == (20, 3)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: Addition of extreme aspect ratio matrices
if rank == 0
    println("[DEBUG] Test 7: Addition of extreme aspect ratio matrices")
    flush(stdout)
end

# Both matrices must have same dimensions for addition
A_add = ones(2, 18)  # 2×18
B_add = 2.0 * ones(2, 18)  # 2×18
drA_add = SafePETSc.Mat_uniform(A_add)
drB_add = SafePETSc.Mat_uniform(B_add)
drC_add = drA_add + drB_add
@test size(drC_add) == (2, 18)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: Very extreme ratio (1×50)
if rank == 0
    println("[DEBUG] Test 8: Very extreme ratio 1×50")
    flush(stdout)
end

A_extreme = reshape(Float64.(1:50), 1, 50)  # 1 row, 50 columns
drA_extreme = SafePETSc.Mat_uniform(A_extreme)
@test size(drA_extreme) == (1, 50)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: Matrix-vector with extreme aspect ratio
if rank == 0
    println("[DEBUG] Test 9: Mat-vec with very wide matrix")
    flush(stdout)
end

# (3×25) * 25-vector = 3-vector
A_matvec = reshape(Float64.(1:75), 3, 25)
x_matvec = ones(25)
drA_matvec = SafePETSc.Mat_uniform(A_matvec)
drx_matvec = SafePETSc.Vec_uniform(x_matvec)
dry_matvec = drA_matvec * drx_matvec
@test size(dry_matvec) == (3,)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 10: Chained products with extreme ratios
if rank == 0
    println("[DEBUG] Test 10: Chained products with extreme ratios")
    flush(stdout)
end

# (2×10) * (10×5) * (5×20) = (2×20)
A_chain1 = ones(2, 10)
B_chain1 = ones(10, 5)
C_chain1 = ones(5, 20)
drA_chain1 = SafePETSc.Mat_uniform(A_chain1)
drB_chain1 = SafePETSc.Mat_uniform(B_chain1)
drC_chain1 = SafePETSc.Mat_uniform(C_chain1)
drAB = drA_chain1 * drB_chain1  # (2×10) * (10×5) = (2×5)
@test size(drAB) == (2, 5)
drABC = drAB * drC_chain1  # (2×5) * (5×20) = (2×20)
@test size(drABC) == (2, 20)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All extreme aspect ratio tests completed")
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
    println("Test Summary: Extreme aspect ratio tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Extreme aspect ratio test file completed successfully")
    flush(stdout)
end

