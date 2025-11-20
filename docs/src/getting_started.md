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
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) your_script.jl`)'
```

This ensures the correct MPI implementation and Julia executable are used.

!!! note "Using Project Environments"
    If your script requires a specific Julia project environment, add `--project=your_project_path` to the julia command:
    ```bash
    julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. your_script.jl`)'
    ```
    This is only needed if SafePETSc is not installed globally or if you need other project-specific dependencies.

### Using System MPI on HPC Clusters

On HPC clusters, you typically need to use the cluster's native MPI library (not the one Julia ships with) for optimal performance and compatibility with the job scheduler. Here's how to configure this:

#### Step 1: Load the MPI Module (Shell Command)

First, load your cluster's MPI module. This is a **shell command** (run in your terminal):

```bash
# Example for clusters using the module system
module load openmpi  # or module load mpich, module load intel-mpi, etc.
```

Check which MPI was loaded:

```bash
# Shell command
which mpiexec
```

#### Step 2: Configure Julia to Use System MPI

You need to tell Julia's MPI.jl package to use the system MPI library instead of its bundled version. This is done using **MPIPreferences.jl**.

Run this **Julia code** (in a Julia REPL or as a script):

```julia
# Julia code - run this once per project
using MPIPreferences
MPIPreferences.use_system_binary()
```

Alternatively, if you want to specify the MPI library explicitly:

```julia
# Julia code - specify the exact MPI library path
using MPIPreferences
MPIPreferences.use_system_binary(
    mpiexec = "/path/to/your/mpiexec",  # Get this from 'which mpiexec'
    vendor = "OpenMPI"  # or "MPICH", "IntelMPI", etc.
)
```

#### Step 3: Rebuild MPI.jl

After configuring MPIPreferences, you **must** rebuild MPI.jl. Run this **Julia code**:

```julia
# Julia code - rebuild MPI.jl to use the system library
using Pkg
Pkg.build("MPI"; verbose=true)
```

#### Step 4: Verify the Configuration

Check that MPI.jl is now using the system MPI. Run this **Julia code**:

```julia
# Julia code - verify MPI configuration
using MPI
MPI.versioninfo()
```

You should see your cluster's MPI library listed (e.g., OpenMPI 4.1.x, not MPItrampoline).

#### Step 5: Run Your Code

Now you can run your SafePETSc code. On clusters, you typically use the cluster's job scheduler:

```bash
# Shell command - example SLURM job submission
sbatch my_job.sh
```

Example `my_job.sh` script:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=1:00:00

# Load MPI module
module load openmpi

# Run Julia with MPI
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 32 $(Base.julia_cmd()) my_script.jl`)'
```

Or for PBS/Torque:

```bash
#!/bin/bash
#PBS -l nodes=2:ppn=16
#PBS -l walltime=1:00:00

cd $PBS_O_WORKDIR
module load openmpi

julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 32 $(Base.julia_cmd()) my_script.jl`)'
```

#### Important Notes

- **One-time setup**: Steps 1-3 only need to be done once per project/environment
- **Module loading**: You must load the MPI module in your job scripts (Step 1) every time you submit a job
- **Consistency**: Use the same MPI library that PETSc was built against on your cluster
- **Project-specific**: The MPI configuration is stored in your project's `LocalPreferences.toml` file

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
using LinearAlgebra

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

# Print result (only on rank 0)
println(io0(), "System solved successfully")
println(io0(), "Solution norm: ", norm(x))

# Explicit cleanup (optional - happens automatically at scope exit)
SafeMPI.check_and_destroy!()
```

## Running the Example

Save the above code as `example.jl` and run:

```bash
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) example.jl`)'
```

## Next Steps

- Learn about [Distributed Reference Management](guide/distributed_refs.md)
- Explore [Vectors](guide/vectors.md)
- Understand [Matrices](guide/matrices.md)
- Use [Linear Solvers](guide/solvers.md)
