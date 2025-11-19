using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("Testing matrix-matrix multiplication...")
    flush(stdout)
end

try
    A_data = reshape(Float64.(1:16), 4, 4)
    B_data = reshape(Float64.(1:16), 4, 4)
    drA = SafePETSc.Mat_uniform(A_data)
    drB = SafePETSc.Mat_uniform(B_data)
    
    if rank == 0
        println("Calling drA * drB...")
        flush(stdout)
    end
    
    drC = drA * drB
    
    if rank == 0
        println("Success! Size: ", size(drC))
        flush(stdout)
    end
    
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
    
    if rank == 0
        println("Test passed!")
    end
catch e
    if rank == 0
        println("Error: ", e)
        println(stacktrace(catch_backtrace()))
        flush(stdout)
    end
    rethrow()
end

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
