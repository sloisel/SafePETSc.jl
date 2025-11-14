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
    println("[DEBUG] Mat_sum test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Mat_sum tests" begin

if rank == 0
    println("[DEBUG] Mat_sum Test 1 starting")
    flush(stdout)
end

# Test 1: Basic summing - each rank contributes to different entries
A = spzeros(Float64, 8, 8)  # Start with empty sparse matrix
if rank == 0
    A[1, 1] = 1.0
    A[1, 2] = 2.0
elseif rank == 1
    A[3, 3] = 10.0
    A[3, 4] = 20.0
elseif rank == 2
    A[5, 5] = 100.0
    A[5, 6] = 200.0
elseif rank == 3
    A[7, 7] = 1000.0
    A[7, 8] = 2000.0
end

dr = SafePETSc.Mat_sum(A)
@test dr isa SafeMPI.DRef
obj = dr.obj
@test obj.A isa PETSc.Mat
@test length(obj.row_partition) == nranks + 1
@test length(obj.col_partition) == nranks + 1

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Overlapping contributions - multiple ranks contribute to same entry
A2 = spzeros(Float64, 8, 8)
# All ranks contribute to entry (4, 4)
A2[4, 4] = Float64(rank + 1)  # rank 0 adds 1, rank 1 adds 2, rank 2 adds 3, rank 3 adds 4

dr2 = SafePETSc.Mat_sum(A2)
@test dr2 isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: own_rank_only=true with valid partitioning
A3 = spzeros(Float64, 8, 8)
# Each rank only sets entries in its own row partition
row_partition = SafePETSc.default_row_partition(8, nranks)
lo = row_partition[rank+1]
hi = row_partition[rank+2] - 1

if lo <= hi
    A3[lo, lo] = Float64(rank * 10)
end

dr3 = SafePETSc.Mat_sum(A3; own_rank_only=true)
@test dr3 isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: Custom row and column partition
if nranks == 4
    custom_row_partition = [1, 3, 5, 7, 9]
    custom_col_partition = [1, 3, 5, 7, 9]
    A4 = spzeros(Float64, 8, 8)

    # Each rank contributes to its own row partition
    partition_lo = custom_row_partition[rank+1]
    partition_hi = custom_row_partition[rank+2] - 1

    if partition_lo <= partition_hi
        A4[partition_lo, partition_lo] = Float64(100 * rank)
    end

    dr4 = SafePETSc.Mat_sum(A4; row_partition=custom_row_partition, col_partition=custom_col_partition, own_rank_only=true)
    @test dr4.obj.row_partition == custom_row_partition
    @test dr4.obj.col_partition == custom_col_partition

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

# Test 5: Empty sparse matrix (no nonzeros)
A5 = spzeros(Float64, 8, 8)
dr5 = SafePETSc.Mat_sum(A5)
@test dr5 isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: Non-square sparse matrix
A6 = spzeros(Float64, 12, 8)
row_partition6 = SafePETSc.default_row_partition(12, nranks)
col_partition6 = SafePETSc.default_row_partition(8, nranks)

# Each rank adds one entry in its row partition
lo6 = row_partition6[rank+1]
hi6 = row_partition6[rank+2] - 1

if lo6 <= hi6
    A6[lo6, 1] = Float64(rank + 1)
end

dr6 = SafePETSc.Mat_sum(A6; row_partition=row_partition6, col_partition=col_partition6, own_rank_only=true)
@test size(dr6.obj) == (12, 8)

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: Sparse matrix with multiple nonzeros per row
A7 = spzeros(Float64, 8, 8)
row_partition7 = SafePETSc.default_row_partition(8, nranks)
lo7 = row_partition7[rank+1]
hi7 = row_partition7[rank+2] - 1

# Each rank fills its local rows with some pattern
for i in lo7:hi7
    for j in 1:8
        if (i + j) % 2 == 0  # checkboard pattern
            A7[i, j] = Float64(i * 10 + j)
        end
    end
end

dr7 = SafePETSc.Mat_sum(A7; own_rank_only=true)
@test dr7 isa SafeMPI.DRef

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

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
    println("Test Summary: Mat_sum tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Mat_sum test file completed successfully")
    flush(stdout)
end
