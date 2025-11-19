using Test
using MPI
using SafePETSc
SafePETSc.Init()
using SafePETSc.SafeMPI: mpi_uniform
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# MPI is initialized by SafePETSc.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "mpi_uniform" begin
    # 1) Identical scalar on all ranks
    A1 = 42
@test mpi_uniform(A1) == true
    MPI.Barrier(comm)

    # 2) Different scalar on at least one rank
    A2 = (rank == 0 ? 1 : 2)
@test mpi_uniform(A2) == false
    MPI.Barrier(comm)

    # 3) Identical small array
    A3 = fill(1.0, 8)
@test mpi_uniform(A3) == true
    MPI.Barrier(comm)

    # 4) Different arrays (same length, different content)
    A4 = [0, 1, 2, 3]
    A4[1] = rank  # make rank-dependent
@test mpi_uniform(A4) == false
    MPI.Barrier(comm)

    # 5) Nested object (Dict) identical across ranks
    A5 = Dict("a" => 1, "b" => [2,3])
@test mpi_uniform(A5) == true
    MPI.Barrier(comm)

    # 6) Per-rank value
    A6 = rank
@test mpi_uniform(A6) == false
end

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
    println("Test Summary: mpi_uniform (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

