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
    println("Simple matrix pooling test")
end

# Clear pool to start
SafePETSc.clear_mat_pool!()

# Test 1: Non-product matrix pooling
if rank == 0
    println("\n=== Test 1: Non-product matrix pooling ===")
end

# Create a matrix
A_data = ones(4, 4)
A1 = SafePETSc.Mat_uniform(A_data; prefix="test_")

if rank == 0
    println("Created first matrix")
end

# Destroy the matrix
A1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("Destroyed matrix, checking pool stats...")
    stats = SafePETSc.get_mat_pool_stats()
    println("Pool stats: ", stats)
end

# Create another matrix with same size and prefix
A2 = SafePETSc.Mat_uniform(A_data; prefix="test_")

if rank == 0
    println("Created second matrix (should be from pool)")
    stats = SafePETSc.get_mat_pool_stats()
    println("Pool stats after reuse: ", stats)
end

# Clean up
A2 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Product matrix pooling
if rank == 0
    println("\n=== Test 2: Product matrix pooling ===")
end

SafePETSc.clear_mat_pool!()

# Create two matrices and compute their product
B_data = ones(4, 4)
C_data = 2.0 * ones(4, 4)
B = SafePETSc.Mat_uniform(B_data; prefix="prod_")
C = SafePETSc.Mat_uniform(C_data; prefix="prod_")

# Compute product
D1 = B * C

if rank == 0
    println("Created first product matrix")
end

# Destroy the product
D1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("Destroyed product matrix, checking pool stats...")
    stats = SafePETSc.get_mat_pool_stats()
    println("Pool stats: ", stats)
end

# Compute the same product again (should reuse from pool)
D2 = B * C

if rank == 0
    println("Created second product matrix (should be from pool)")
    stats = SafePETSc.get_mat_pool_stats()
    println("Pool stats after reuse: ", stats)
end

# Clean up
B = nothing
C = nothing
D2 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("\nTest completed successfully!")
end
