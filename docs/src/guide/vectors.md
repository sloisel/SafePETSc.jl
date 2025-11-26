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

# Vectors always use MPIDENSE prefix internally
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

### Concatenation

Vectors can be concatenated to form new vectors or matrices:

```julia
x = Vec_uniform([1.0, 2.0, 3.0])
y = Vec_uniform([4.0, 5.0, 6.0])

# Vertical concatenation (stacking) - creates a longer vector
v = vcat(x, y)  # Vec{Float64} with 6 elements

# Horizontal concatenation - creates a matrix (auto-upgrades to MPIDENSE)
M = hcat(x, y)  # 3×2 Mat{Float64,MPIDENSE}
```

!!! tip "Vector Concatenation Behavior"
    - `vcat` of vectors returns a `Vec{T}` (preserves vector type)
    - `hcat` of vectors returns a `Mat{T,Prefix}` (creates a multi-column matrix)
    - `vcat` of matrices returns a `Mat{T,Prefix}`

    Vectors always use MPIDENSE prefix internally. Horizontal concatenation creates
    a dense matrix since vectors are inherently dense.

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
```

## Row Ownership and Indexing

### Determining Owned Rows

Use `own_row()` to find which indices are owned by the current rank:

```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# Get ownership range for this rank
owned = own_row(v)  # e.g., 1:2 on rank 0, 3:4 on rank 1

println(io0(), "Rank $(MPI.Comm_rank(MPI.COMM_WORLD)) owns indices: $owned")
```

### Indexing Vectors

**Important:** You can only index elements that are owned by the current rank. Attempting to access non-owned indices will result in an error.

```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
owned = own_row(v)

# ✓ CORRECT - Access owned elements
if 2 in owned
    val = v[2]  # Returns 2.0 on the rank that owns index 2
end

# ✓ CORRECT - Access range of owned elements
if owned == 1:2
    vals = v[1:2]  # Returns [1.0, 2.0] on the rank that owns these indices
end

# ❌ WRONG - Accessing non-owned indices causes an error
val = v[3]  # ERROR if rank doesn't own index 3!
```

**Indexing is non-collective** - each rank can independently access its owned data without coordination.

### Use Cases for Indexing

Indexing is useful when you need to:
- Extract specific local values for computation
- Implement custom local operations
- Interface with non-PETSc code on owned data

```julia
# Extract owned portion for local processing
v = Vec_uniform(randn(100))
owned = own_row(v)

# Get local values
local_vals = v[owned]

# Process locally
local_sum = sum(local_vals)
local_max = maximum(local_vals)

# Aggregate across ranks if needed
global_sum = MPI.Allreduce(local_sum, +, MPI.COMM_WORLD)
```

## Row-wise Operations with map_rows

The `map_rows()` function applies a function to each row of distributed vectors or matrices, similar to Julia's `map` but for distributed PETSc objects.

### Basic Usage

```julia
# Apply function to each element of a vector
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
squared = map_rows(x -> x[1]^2, v)  # Returns Vec([1.0, 4.0, 9.0, 16.0])

# Transform vector to matrix (using adjoint for row output)
powers = map_rows(x -> [x[1], x[1]^2, x[1]^3]', v)  # Returns 4×3 Mat
```

**Note:** For vectors, the function receives a 1-element view, so use `x[1]` to access the scalar value.

### Output Types

The return type depends on what your function returns:

- **Scalar** → Returns a `Vec` with same number of rows
- **Vector** → Returns a `Vec` with expanded rows (m*n rows if each returns n-element vector)
- **Adjoint Vector** (row vector) → Returns a `Mat{T,MPIDENSE}` with m rows (always dense)

```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# Scalar output: Vec with same size
doubled = map_rows(x -> 2 * x[1], v)

# Vector output: Vec with expanded size (4 * 2 = 8 elements)
expanded = map_rows(x -> [x[1], x[1]^2], v)

# Adjoint vector output: Mat with 4 rows, 3 columns
matrix_form = map_rows(x -> [x[1], x[1]^2, x[1]^3]', v)
```

### Combining Multiple Inputs

Process multiple vectors or matrices together:

```julia
v1 = Vec_uniform([1.0, 2.0, 3.0])
v2 = Vec_uniform([4.0, 5.0, 6.0])

# Combine two vectors element-wise
combined = map_rows((x, y) -> [x[1] + y[1], x[1] * y[1]]', v1, v2)
# Returns 3×2 matrix: [sum, product] for each pair
```

**Important:** All inputs must have the same row partition.

### Performance Notes

- `map_rows()` is a **collective operation** - all ranks must call it
- The function is applied only to locally owned rows on each rank
- Results are automatically assembled into a new distributed object
- Works efficiently with both vectors and matrices (see [Matrices guide](matrices.md#row-wise-operations-with-map_rows))

## PETSc Options

SafePETSc vectors always use the MPIDENSE prefix internally for PETSc options. Unlike matrices, vectors do not expose a Prefix type parameter because all PETSc vectors are inherently dense (they store all elements).

```julia
# Configure PETSc options for vectors (uses MPIDENSE prefix)
petsc_options_insert_string("-MPIDENSE_vec_type cuda")
```

See [Matrices: PETSc Options and Prefix Types](matrices.md#petsc-options-and-the-prefix-type-parameter) for the prefix system used by matrices.

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

## Converting to Julia Arrays

Convert a distributed `Vec` to a native Julia `Vector`:

```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
v_julia = Vector(v)  # Returns Vector{Float64}

# Or use the universal J() function
v_julia = J(v)       # Same result
```

**Important:** Conversion is a **collective operation** - all ranks must call it.

See [Converting to Native Julia Arrays](io.md#converting-to-native-julia-arrays) for complete documentation including:
- The universal `J()` conversion function
- Performance considerations
- When to use (and avoid) conversions
- Working with converted data

## See Also

- [`Vec_uniform`](@ref)
- [`Vec_sum`](@ref)
- [`zeros_like`](@ref)
- [`ones_like`](@ref)
- [`fill_like`](@ref)
- [Input/Output and Display](io.md) - Display and conversion operations

## Pooling and Cleanup

By default, released PETSc vectors are returned to an internal pool for reuse instead of being destroyed immediately. This reduces allocation overhead in vector-heavy workflows.

- Disable pooling: `ENABLE_VEC_POOL[] = false`
- Manually free pooled vectors: `clear_vec_pool!()`
- Cleanup points: `SafeMPI.check_and_destroy!()` performs partial GC and collective release processing; vectors in use remain valid, pooled vectors remain available for reuse
