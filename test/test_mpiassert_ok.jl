using Test
using MPI
using SafePETSc
SafePETSc.Init()
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

# MPI is initialized by SafePETSc.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

ts = @testset QuietTestSet "@mpiassert OK (no abort)" begin
    # Case: all ranks pass
    ok_all = true
    @mpiassert ok_all
    @test true

    MPI.Barrier(comm)

    # Case: non-trivial but still true everywhere
    ok_expr = (rank >= 0 && rank < nranks)  # always true
    @mpiassert ok_expr "bounds check across ranks"
    @test true

    MPI.Barrier(comm)
end

# Aggregate per-rank counts and print a single summary on root
local_counts = [
    ts.counts[:pass],
    ts.counts[:fail],
    ts.counts[:error],
    ts.counts[:broken],
    ts.counts[:skip],
]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: @mpiassert OK (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

# If any failures or errors occurred anywhere, exit non-zero from root so CI detects failure
if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end

if rank == 0
    println("@mpiassert OK test completed")
end

