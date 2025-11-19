using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

const _VERBOSE = get(ENV, "VERBOSE_VEC_INDEXING", "0") == "1"

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Vec indexing test starting")
    flush(stdout)
end

function _vec_indexing_tests_body()
    if rank == 0
        println("[DEBUG] Vec indexing Test 1: Single element indexing")
        flush(stdout)
    end

    # Test 1: Single element indexing
    v_data = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    v = SafePETSc.Vec_uniform(v_data)

    # Get local range
    row_lo = v.obj.row_partition[rank+1]
    row_hi = v.obj.row_partition[rank+2] - 1

    # Test accessing each element in our local range
    for i in row_lo:row_hi
        val = v[i]
        @test val ≈ v_data[i]
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Vec indexing Test 2: Range indexing")
        flush(stdout)
    end

    # Test 2: Range indexing
    v2_data = Float64[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    v2 = SafePETSc.Vec_uniform(v2_data)

    row_lo = v2.obj.row_partition[rank+1]
    row_hi = v2.obj.row_partition[rank+2] - 1

    # Test full local range
    vals = v2[row_lo:row_hi]
    @test vals isa Vector{Float64}
    @test length(vals) == row_hi - row_lo + 1
    @test all(vals .≈ v2_data[row_lo:row_hi])

    # Test partial local range (if we have more than one element)
    if row_hi > row_lo
        vals_partial = v2[row_lo:row_lo+1]
        @test length(vals_partial) == 2
        @test all(vals_partial .≈ v2_data[row_lo:row_lo+1])
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Vec indexing Test 3: Different sizes")
        flush(stdout)
    end

    # Test 3: Different vector sizes
    v3_data = Float64[1.0, 2.0, 3.0, 4.0]
    v3 = SafePETSc.Vec_uniform(v3_data)

    row_lo = v3.obj.row_partition[rank+1]
    row_hi = v3.obj.row_partition[rank+2] - 1

    for i in row_lo:row_hi
        @test v3[i] ≈ v3_data[i]
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Vec indexing tests completed")
        flush(stdout)
    end
end

# Keep output tidy and aggregate at the end
if _VERBOSE
    @testset "Vec indexing tests" begin
        _vec_indexing_tests_body()
    end
else
    ts = @testset QuietTestSet "Vec indexing tests" begin
        _vec_indexing_tests_body()
    end
end

if !_VERBOSE
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
    println("Test Summary: Vec indexing tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Vec indexing test file completed successfully")
    flush(stdout)
end
end # !_VERBOSE

