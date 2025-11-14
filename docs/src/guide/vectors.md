# Vectors

SafePETSc provides distributed vectors through the `Vec{T}` type, which wraps PETSc's distributed vector functionality with automatic memory management.

## Creating Vectors

### Uniform Distribution

Use `Vec_uniform` when all ranks have the same data:

```julia
# Create a vector from uniform data
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# With custom partition
partition = [1, 3, 5]  # rank 0: rows 1-2, rank 1: rows 3-4
v = Vec_uniform([1.0, 2.0, 3.0, 4.0]; row_partition=partition)

# With PETSc options prefix
v = Vec_uniform([1.0, 2.0]; prefix="my_vec_")
```

### Sum Distribution

Use `Vec_sum` when ranks contribute sparse entries:

```julia
using SparseArrays

# Each rank contributes different entries
# All contributions are summed
rank = MPI.Comm_rank(MPI.COMM_WORLD)
indices = [rank * 2 + 1, rank * 2 + 2]
values = [1.0, 2.0]
v = Vec_sum(sparsevec(indices, values, 10))
```

### Helper Constructors

Create vectors similar to existing ones:

```julia
# Zero vector with same size/partition as x
y = zeros_like(x)

# Ones vector
y = ones_like(x)

# Filled with specific value
y = fill_like(x, 3.14)

# With different element type
y = zeros_like(x; T2=Float32)
```

## Vector Operations

### Broadcasting

Vectors support Julia's broadcasting syntax:

```julia
# Element-wise operations
y = x .+ 1.0
y = 2.0 .* x
y = x .^ 2

# Vector-vector operations
z = x .+ y
z = x .* y

# In-place operations
y .= x .+ 1.0
y .= 2.0 .* x .+ y
```

### Arithmetic

```julia
# Addition and subtraction
z = x + y
z = x - y

# Mixed with scalars
z = x + 1.0
z = 2.0 - x

# Unary operations
z = -x
z = +x
```

### Linear Algebra

```julia
# Adjoint (transpose)
x_adj = x'

# Adjoint-matrix multiplication
result = x' * A  # Returns adjoint vector
```

## Partitioning

Vectors are partitioned across ranks to distribute work and memory.

### Default Partitioning

```julia
# Equal distribution
n = 100
nranks = MPI.Comm_size(MPI.COMM_WORLD)
partition = default_row_partition(n, nranks)

# For n=100, nranks=4:
# partition = [1, 26, 51, 76, 101]
# rank 0: rows 1-25 (25 elements)
# rank 1: rows 26-50 (25 elements)
# rank 2: rows 51-75 (25 elements)
# rank 3: rows 76-100 (25 elements)
```

### Custom Partitioning

```julia
# Unequal distribution
partition = [1, 10, 30, 101]  # Different sizes per rank
v = Vec_uniform(data; row_partition=partition)
```

### Partition Requirements

- Length `nranks + 1`
- First element is `1`
- Last element is `n + 1` (where `n` is vector length)
- Strictly increasing

## Properties

```julia
# Element type
T = eltype(v)  # e.g., Float64

# Size
n = length(v)
n = size(v, 1)

# Partition information
partition = v.obj.row_partition
prefix = v.obj.prefix
```

## PETSc Options

Use prefixes to configure PETSc behavior:

```julia
# Set options globally
petsc_options_insert_string("-my_vec_type cuda")

# Create vector with prefix
v = Vec_uniform(data; prefix="my_vec_")

# Now PETSc will use the "cuda" type for this vector
```

## Examples

### Parallel Computation

```julia
using SafePETSc
using MPI

SafePETSc.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)

# Each rank creates local data
n_global = 1000
partition = default_row_partition(n_global, nranks)
lo = partition[rank + 1]
hi = partition[rank + 2] - 1

# Generate data based on rank
data = collect(range(1.0, length=n_global))

# Create distributed vector
v = Vec_uniform(data)

# Compute: y = 2x + 1
y = 2.0 .* v .+ 1.0

println(io0(), "Computation complete")
```

### Sparse Contributions

```julia
using SafePETSc
using SparseArrays
using MPI

SafePETSc.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
n = 100

# Each rank contributes to different parts
# Use own_rank_only=true to assert local contributions
lo = rank * 25 + 1
hi = (rank + 1) * 25
indices = collect(lo:hi)
values = ones(length(indices)) * (rank + 1)

v = Vec_sum(sparsevec(indices, values, n); own_rank_only=true)
```

## Performance Tips

1. **Use Broadcasting**: In-place broadcasting (`y .= ...`) avoids allocations
2. **Batch Operations**: Combine multiple operations in one broadcast
3. **Avoid Extraction**: Keep data in distributed vectors; don't extract to Julia arrays
4. **GPU-Aware**: Set PETSc options for GPU execution

```julia
# Good: in-place, batched
y .= 2.0 .* x .+ 3.0 .* z .+ 1.0

# Less good: multiple allocations
y = 2.0 * x
y = y + 3.0 * z
y = y + 1.0
```

## See Also

- [`Vec_uniform`](@ref)
- [`Vec_sum`](@ref)
- [`zeros_like`](@ref)
- [`ones_like`](@ref)
- [`fill_like`](@ref)
- [Converting to Native Julia Arrays](io.md#converting-to-native-julia-arrays) - Convert `Vec` to `Vector`

## Pooling and Cleanup

By default, released PETSc vectors are returned to an internal pool for reuse instead of being destroyed immediately. This reduces allocation overhead in vector-heavy workflows.

- Disable pooling: `ENABLE_VEC_POOL[] = false`
- Manually free pooled vectors: `clear_vec_pool!()`
- Cleanup points: `SafeMPI.check_and_destroy!()` performs partial GC and collective release processing; vectors in use remain valid, pooled vectors remain available for reuse
