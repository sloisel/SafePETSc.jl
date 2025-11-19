using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SparseArrays

# PETSc is initialized by SafePETSc.Init()

@testset "Stress test: 100 large vector allocations with 4 ranks" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    
    if nranks != 4
        error("This test must be run with 4 ranks, got $nranks")
    end
    
    N = 1_000_000  # vector dimension
    n_vecs = 100  # 100 large vectors
    
    if rank == 0
        println("Starting stress test: allocating $n_vecs vectors of dimension $N with $nranks ranks")
        flush(stdout)
    end
    
    start_time = time()
    
    # Allocate many large vectors
    # We don't keep references, so they should get GC'd and finalized
    for i in 1:n_vecs
        # Create a small sparse vector (mostly zeros)
        indices = [rand(1:N)]
        values = [Float64(i)]
        v = sparsevec(indices, values, N)
        
        # Create SafePETSc Vec (will be GC'd immediately after this iteration)
        _ = SafePETSc.Vec_sum(v)
        
        # Print progress every vector on rank 0
#        if rank == 0
#            elapsed = time() - start_time
#            rate = i / elapsed
#            println("  Progress: $i/$n_vecs vectors ($(round(rate, digits=1)) vec/s)")
#            flush(stdout)
#        end
    end
    
    # Final barrier and check_and_destroy to flush everything
    MPI.Barrier(MPI.COMM_WORLD)
    SafePETSc.SafeMPI.check_and_destroy!()
    
    elapsed = time() - start_time
    
    if rank == 0
        println("Completed: $n_vecs vectors in $(round(elapsed, digits=2))s ($(round(n_vecs/elapsed, digits=1)) vec/s)")
        flush(stdout)
    end
    
    # Verify we didn't leak resources
    manager = SafePETSc.SafeMPI.default_manager[]
    if rank == 0
        n_live = length(manager.objs)
        println("Live objects remaining: $n_live (should be 0 or very small)")
        if n_live >= 100
            error("Too many objects remaining: $n_live")
        end
        @test n_live < 100
    end
    
    MPI.Barrier(MPI.COMM_WORLD)
end

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()

# Rely on library finalization at exit
