using Test
using MPI
using SafePETSc
SafePETSc.Init()
using SafePETSc: MPIDENSE, MPIAIJ
using PETSc
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset MPITestHarness.QuietTestSet "eachrow on mpidense" begin
    # Build a simple 4x4 matrix with known rows
    A_data = reshape(Float64.(1:16), 4, 4)
    A = SafePETSc.Mat_uniform(A_data; Prefix=MPIDENSE)

    # Determine local rows for this rank
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # 1) Full iteration: all local rows should match
    i_local = 0
    for r in eachrow(A)
        i_local += 1
        i_global = row_lo + i_local - 1
        @test collect(r) == collect(A_data[i_global, :])
    end
    @test i_local == max(0, row_hi - row_lo + 1)

    # 2) Early break: acquire iterator, break early, then GC to exercise finalizer
    it = eachrow(A)
    st = iterate(it)
    if st !== nothing
        # One row then break
        _ = st[1]
    end
    # Drop reference and force GC; should not error
    it = nothing
    GC.gc()
end

MPI.Barrier(comm)

# Aggregate counts and print summary on root
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
    println("Test Summary: eachrow on mpidense (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
