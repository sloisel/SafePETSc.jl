# Matrices

SafePETSc provides distributed matrices through the `Mat{T}` type, which wraps PETSc's distributed matrix functionality with GPU-friendly operations and automatic memory management.

## Creating Matrices

### Uniform Distribution

Use `Mat_uniform` when all ranks have the same data:

```julia
# Create from dense matrix
A = Mat_uniform([1.0 2.0; 3.0 4.0])

# With custom partitions
row_part = [1, 2, 3]  # 2 ranks
col_part = [1, 2, 3]
A = Mat_uniform(data; row_partition=row_part, col_partition=col_part)

# With PETSc options prefix
A = Mat_uniform(data; prefix="my_mat_")
```

### Sum Distribution

Use `Mat_sum` when ranks contribute sparse entries:

```julia
using SparseArrays

# Each rank contributes different sparse entries
# All contributions are summed
rank = MPI.Comm_rank(MPI.COMM_WORLD)
I = [1, rank+1]
J = [1, rank+1]
V = [1.0, 2.0]
A = Mat_sum(sparse(I, J, V, 10, 10))

# Assert local ownership for validation
A = Mat_sum(sparse_local; own_rank_only=true)
```

## Matrix Operations

### Linear Algebra

```julia
# Matrix-vector multiplication
y = A * x

# In-place
mul!(y, A, x)

# Matrix-matrix multiplication
C = A * B

# In-place
mul!(C, A, B)

# Transpose
B = A'
B = Mat(A')  # Materialize transpose

# In-place transpose (reuses B)
transpose!(B, A)
```

### Concatenation

```julia
# Vertical concatenation
C = vcat(A, B)
C = cat(A, B; dims=1)

# Horizontal concatenation
D = hcat(A, B)
D = cat(A, B; dims=2)

# Block diagonal
E = blockdiag(A, B, C)
```

### Sparse Diagonal Matrices

```julia
using SparseArrays

# Create diagonal matrix from vectors
diag_vec = Vec_uniform(ones(100))
upper_vec = Vec_uniform(ones(99))
lower_vec = Vec_uniform(ones(99))

# Tridiagonal matrix
A = spdiagm(-1 => lower_vec, 0 => diag_vec, 1 => upper_vec)

# Explicit dimensions
A = spdiagm(100, 100, 0 => diag_vec, 1 => upper_vec)
```

## Transpose Operations

SafePETSc uses PETSc's efficient transpose operations:

```julia
# Create transpose (new matrix)
B = Mat(A')

# Reuse transpose storage
B = Mat(A')  # Initial creation
# ... later, after A changes:
transpose!(B, A)  # Reuse B's storage
```

Note: For `transpose!` to work correctly with PETSc's reuse mechanism, `B` should have been created as a transpose of `A` initially.

## Properties

```julia
# Element type
T = eltype(A)

# Size
m, n = size(A)
m = size(A, 1)
n = size(A, 2)

# Partition information
row_part = A.obj.row_partition
col_part = A.obj.col_partition
prefix = A.obj.prefix
```

## Partitioning

Matrices have both row and column partitions.

### Default Partitioning

```julia
m, n = 100, 80
nranks = MPI.Comm_size(MPI.COMM_WORLD)

row_part = default_row_partition(m, nranks)
col_part = default_row_partition(n, nranks)
```

### Requirements

- Row operations require matching row partitions
- Column operations require matching column partitions
- Matrix multiplication: `C = A * B` requires `A.col_partition == B.row_partition`

## GPU-Friendly Operations

SafePETSc prioritizes PETSc's native GPU-compatible operations:

```julia
# Good: Uses PETSc's MatTranspose (GPU-friendly)
B = Mat(A')

# Good: Uses PETSc's MatMatMult (GPU-friendly)
C = A * B

# Good: Uses PETSc's MatConvert (GPU-friendly)
# (internal to SafePETSc operations)
```

Avoid element-by-element access patterns that cause excessive GPU↔CPU transfers.

## Advanced Features

### Iterating Over Dense Matrix Rows

For `MATMPIDENSE` matrices:

```julia
# Iterate over local rows efficiently
for row in eachrow(A)
    # row is a view of the matrix row
    process(row)
end
```

This uses a single `MatDenseGetArrayRead` call for the entire iteration.

### PETSc Options

Configure matrix behavior via options:

```julia
# Set global options
petsc_options_insert_string("-dense_mat_type mpidense")

# Use prefix for specific matrices
A = Mat_uniform(data; prefix="my_mat_")
```

## Examples

### Assemble a Sparse Matrix

```julia
using SafePETSc
using SparseArrays
using MPI

SafePETSc.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)

n = 100

# Each rank builds a local piece
row_part = default_row_partition(n, nranks)
lo = row_part[rank + 1]
hi = row_part[rank + 2] - 1

# Build local sparse matrix (only owned rows)
I = Int[]
J = Int[]
V = Float64[]

for i in lo:hi
    # Diagonal
    push!(I, i)
    push!(J, i)
    push!(V, 2.0)

    # Off-diagonal
    if i > 1
        push!(I, i)
        push!(J, i-1)
        push!(V, -1.0)
    end
    if i < n
        push!(I, i)
        push!(J, i+1)
        push!(V, -1.0)
    end
end

local_sparse = sparse(I, J, V, n, n)

# Assemble global matrix
A = Mat_sum(local_sparse; own_rank_only=true)
```

### Build Block Matrices

```julia
# Create blocks
A11 = Mat_uniform(...)
A12 = Mat_uniform(...)
A21 = Mat_uniform(...)
A22 = Mat_uniform(...)

# Assemble block matrix
A = vcat(hcat(A11, A12), hcat(A21, A22))

# Or equivalently
A = [A11 A12; A21 A22]  # (if block-matrix syntax is supported)
```

### Tridiagonal System

```julia
n = 1000

# Create diagonal vectors
diag = Vec_uniform(2.0 * ones(n))
upper = Vec_uniform(-1.0 * ones(n-1))
lower = Vec_uniform(-1.0 * ones(n-1))

# Build tridiagonal matrix
A = spdiagm(-1 => lower, 0 => diag, 1 => upper)

# Create RHS
b = Vec_uniform(ones(n))

# Solve
x = A \ b
```

## Performance Tips

1. **Use Native Operations**: Prefer PETSc operations over element access
2. **Batch Assembly**: Build sparse matrices locally, then sum once
3. **Appropriate Matrix Type**: Use dense vs. sparse based on structure
4. **Reuse Solver Objects**: Create `Solver` once, reuse for multiple solves
5. **GPU Configuration**: Set PETSc options for GPU matrices

```julia
# Good: bulk assembly
local_matrix = sparse(I, J, V, m, n)
A = Mat_sum(local_matrix)

# Less good: element-by-element (if it were supported)
# A = Mat_sum(...)
# for each element
#     set_value(A, i, j, val)  # Repeated MPI calls
```

## Compatibility Notes

- **Transpose Reuse**: `transpose!(B, A)` requires that `B` was created via `Mat(A')` or has a compatible precursor
- **Matrix Multiplication Reuse**: `mul!(C, A, B)` requires pre-allocated `C` with correct partitions
- **Dense Operations**: Some operations (e.g., `\` with matrix RHS) require dense matrices

## See Also

- [`Mat_uniform`](@ref)
- [`Mat_sum`](@ref)
- [`spdiagm`](@ref)
- [`vcat`](@ref), [`hcat`](@ref), [`blockdiag`](@ref)
