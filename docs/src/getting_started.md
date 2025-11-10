# Getting Started

This guide will help you get started with SafePETSc.jl for distributed parallel computing with MPI and PETSc.

## Prerequisites

SafePETSc requires:
- Julia 1.6 or later
- MPI installation (OpenMPI, MPICH, etc.)
- PETSc installation

## Running with MPI

SafePETSc programs must be run with MPI. Use the MPI.jl wrapper to ensure compatibility:

```bash
# Run with 4 MPI processes
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. your_script.jl`)'
```

This ensures the correct MPI implementation and Julia executable are used.

## Basic Workflow

### 1. Initialize

Always start by initializing MPI and PETSc:

```julia
using SafePETSc
using MPI

SafePETSc.Init()
```

This ensures both MPI and PETSc are properly initialized.

### 2. Create Distributed Objects

SafePETSc provides two main patterns for creating distributed objects:

#### Uniform Distribution

Use when all ranks have the same data:

```julia
# Same matrix on all ranks
A = Mat_uniform([1.0 2.0; 3.0 4.0])

# Same vector on all ranks
v = Vec_uniform([1.0, 2.0])
```

#### Sum Distribution

Use when ranks contribute different sparse data:

```julia
using SparseArrays

# Each rank contributes sparse entries
# Entries are summed across ranks
A = Mat_sum(sparse([1], [1], [rank_value], 10, 10))
v = Vec_sum(sparsevec([rank_id], [rank_value], 10))
```

### 3. Perform Operations

```julia
# Matrix-vector multiplication
y = A * v

# In-place operations
y .= A * v .+ 1.0

# Linear solve
x = A \ b

# Matrix operations
C = A * B
D = A'  # Transpose
```

### 4. Cleanup

Objects are automatically cleaned up when they go out of scope. You can explicitly trigger cleanup:

```julia
SafeMPI.check_and_destroy!()
```

## Complete Example

Here's a complete example that solves a linear system:

```julia
using SafePETSc
using MPI

# Initialize
SafePETSc.Init()

# Get MPI info
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Create a simple 2D Laplacian matrix (uniform on all ranks)
n = 100
A_dense = zeros(n, n)
for i in 1:n
    A_dense[i, i] = 2.0
    if i > 1
        A_dense[i, i-1] = -1.0
    end
    if i < n
        A_dense[i, i+1] = -1.0
    end
end

# Create distributed PETSc matrix
A = Mat_uniform(A_dense)

# Create right-hand side
b = Vec_uniform(ones(n))

# Solve the system
x = A \ b

# Extract local portion for output (if needed)
if rank == 0
    println("System solved successfully")
end

# Explicit cleanup (optional - happens automatically at scope exit)
SafeMPI.check_and_destroy!()
```

## Running the Example

Save the above code as `example.jl` and run:

```bash
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. example.jl`)'
```

## Next Steps

- Learn about [Distributed Reference Management](guide/distributed_refs.md)
- Explore [Vector Operations](guide/vectors.md)
- Understand [Matrix Operations](guide/matrices.md)
- Use [Linear Solvers](guide/solvers.md)
