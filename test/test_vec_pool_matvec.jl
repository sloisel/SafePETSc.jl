using Test
using MPI
using SafePETSc
SafePETSc.Init()
using LinearAlgebra

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if rank == 0
    println("\n=== Testing vector pooling with matrix operations ===\n")
end

# Clear pool
SafePETSc.clear_vec_pool!()

# Create a simple matrix
N = 10
A_data = Matrix{Float64}(I, N, N)
A = SafePETSc.Mat_uniform(A_data)

# Create a vector
x_data = ones(N)
x = SafePETSc.Vec_uniform(x_data)

if rank == 0
    println("First matvec (should CREATE):")
end

# First mat-vec multiply - should create
y1 = A * x
y1 = nothing  # Release
GC.gc()
SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("\nSecond matvec (should POOL):")
end

# Second mat-vec multiply - should reuse from pool
y2 = A * x
y2 = nothing  # Release
GC.gc()
SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("\nThird matvec (should POOL again):")
end

# Third mat-vec multiply - should also reuse from pool
y3 = A * x

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("\n=== Test completed ===")
end

