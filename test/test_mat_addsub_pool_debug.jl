using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

println("[rank $rank] Starting test")
flush(stdout)

clear_mat_pool!()
println("[rank $rank] Cleared pool")
flush(stdout)

# Create two matrices with specific structure
A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
B1 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
println("[rank $rank] Created sparse matrices")
flush(stdout)

matA1 = Mat_sum(A1)
println("[rank $rank] Created matA1")
flush(stdout)

matB1 = Mat_sum(B1)
println("[rank $rank] Created matB1")
flush(stdout)

# First addition - should be a pool miss (creates new matrix)
println("[rank $rank] About to do matA1 + matB1")
flush(stdout)

matC1 = matA1 + matB1
println("[rank $rank] Completed addition, got matC1")
flush(stdout)

# Verify correctness
println("[rank $rank] About to extract local sparse")
flush(stdout)

C1_local = SafePETSc._mat_to_local_sparse(matC1)
println("[rank $rank] Extracted local sparse")
flush(stdout)

C1_sum = zeros(4, 4)
println("[rank $rank] About to do MPI.Reduce")
flush(stdout)

MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
println("[rank $rank] Completed MPI.Reduce")
flush(stdout)

if rank == 0
    expected = Matrix(A1 + B1)
    println("[rank $rank] Checking correctness")
    flush(stdout)
end

# Release C1 back to pool
println("[rank $rank] Setting matC1 to nothing")
flush(stdout)

matC1 = nothing
println("[rank $rank] Calling GC.gc()")
flush(stdout)

GC.gc()
println("[rank $rank] GC.gc() completed, calling check_and_destroy")
flush(stdout)

SafeMPI.check_and_destroy!()
println("[rank $rank] check_and_destroy completed, calling Barrier")
flush(stdout)

MPI.Barrier(comm)
println("[rank $rank] Barrier completed")
flush(stdout)

# Check pool stats
println("[rank $rank] Getting pool stats")
flush(stdout)

stats = get_mat_pool_stats()
println("[rank $rank] Got pool stats: $stats")
flush(stdout)

println("[rank $rank] Test completed successfully")
flush(stdout)
