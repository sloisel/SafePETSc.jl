using Test
using MPI
using SafePETSc
SafePETSc.Init()
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# Initialize MPI if not already initialized
# This allows the test to be run standalone or as part of a test suite
# MPI is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create a test type that tracks destruction
mutable struct TestObject
    id::Int
end

# Flag to track if destruction happened (global so destroy_obj! can access it)
const destroyed_ids = Int[]

# Opt-in to destruction management
SafeMPI.destroy_trait(::Type{TestObject}) = SafeMPI.CanDestroy()

# Define destruction behavior
function SafeMPI.destroy_obj!(obj::TestObject)
    push!(destroyed_ids, obj.id)
    return nothing
end

ts = @testset MPITestHarness.QuietTestSet "SafeMPI with MPI" begin

    begin # Single object GC destruction
        # Run GC test in a separate function to ensure clean scope
        # This is necessary because @testset can hold references
        function run_gc_test()
            # Create ref in local scope that will be GC'd
            obj = TestObject(42)
            ref = DRef(obj)
            # ref goes out of scope when function returns
            nothing
        end

        # Call the function - ref goes out of scope after return
        run_gc_test()

        # Poll for destruction - finalizers typically trigger quickly
        found = false
        for i in 1:10
            check_and_destroy!(SafeMPI.default_manager[]; max_check_count=1)

            if 42 in destroyed_ids
                found = true
                if rank == 0
                    println("✓ GC finalizer triggered on attempt $i")
                end
                break  # Safe to break: check_and_destroy! is a synchronization point
            end
        end

        MPI.Barrier(comm)

        # Test that the object was eventually destroyed via GC
        @test found
        @test 42 in destroyed_ids

        if !found && rank == 0
            println("⚠ GC test failed - object 42 not destroyed")
            println("  destroyed_ids: $destroyed_ids")
        end
    end # Single object GC destruction

    # Multiple object GC destruction
    begin
        # Test multiple objects being GC'd
        function run_multi_gc_test()
            obj1 = TestObject(100)
            ref1 = DRef(obj1)
            obj2 = TestObject(200)
            ref2 = DRef(obj2)
            obj3 = TestObject(300)
            ref3 = DRef(obj3)
            # All refs go out of scope when function returns
            nothing
        end

        run_multi_gc_test()

        # Poll for destruction
        found_all = false
        for i in 1:10
            check_and_destroy!(SafeMPI.default_manager[]; max_check_count=1)

            if 100 in destroyed_ids && 200 in destroyed_ids && 300 in destroyed_ids
                found_all = true
                if rank == 0
                    println("✓ All objects GC'd on attempt $i")
                end
                break  # Safe to break: check_and_destroy! is a synchronization point
            end
        end

        MPI.Barrier(comm)

        # Test that all objects were destroyed
        @test found_all
        @test 100 in destroyed_ids
        @test 200 in destroyed_ids
        @test 300 in destroyed_ids

        if !found_all && rank == 0
            println("⚠ Multi-GC test failed")
            println("  destroyed_ids: $destroyed_ids")
        end
    end # Multiple object GC destruction
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
    println("Test Summary: SafeMPI with MPI (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

# If any failures or errors occurred anywhere, exit non-zero from root so CI detects failure
if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end


# Note: We don't call MPI.Finalize() here because:
# 1. Julia's MPI.jl automatically finalizes MPI at exit via atexit hook
# 2. If we finalize here, subsequent test files cannot use MPI
# 3. Calling Finalize() multiple times causes errors
