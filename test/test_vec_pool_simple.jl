using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("Simple vector pooling test")
end

# Clear pool to start
SafePETSc.clear_vec_pool!()

# Test 1: Create a vector
v = ones(16)
dr1 = SafePETSc.Vec_uniform(v)

if rank == 0
    println("Created first vector")
end

# Destroy the vector
dr1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("Destroyed vector, checking pool stats...")
    stats = SafePETSc.get_vec_pool_stats()
    println("Pool stats: ", stats)
end

# Create another vector with same size and prefix
dr2 = SafePETSc.Vec_uniform(v)

if rank == 0
    println("Created second vector (should be from pool)")
    stats = SafePETSc.get_vec_pool_stats()
    println("Pool stats after reuse: ", stats)
end

# Verify the vector works
local_view = PETSc.unsafe_localarray(dr2.obj.v; read=true, write=false)
try
    if rank == 0
        println("Local values: ", local_view[:])
    end
finally
    Base.finalize(local_view)
end

SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("Test completed successfully!")
end

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
