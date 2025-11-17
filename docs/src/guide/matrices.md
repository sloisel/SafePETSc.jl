# Matrices

SafePETSc provides distributed matrices through the `Mat{T,Prefix}` type, which wraps PETSc's distributed matrix functionality with GPU-friendly operations and automatic memory management.

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

# With custom prefix type (advanced)
A = Mat_uniform(data; Prefix=MPIDENSE)
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
# Vertical concatenation (stacking)
C = vcat(A, B)
C = cat(A, B; dims=1)

# Horizontal concatenation (side-by-side)
D = hcat(A, B)
D = cat(A, B; dims=2)

# Block diagonal
E = blockdiag(A, B, C)

# Concatenating vectors to form matrices
x = Vec_uniform([1.0, 2.0, 3.0])
y = Vec_uniform([4.0, 5.0, 6.0])
M = hcat(x, y)  # Creates 3×2 Mat{Float64,MPIDENSE}
```

!!! note "Automatic Prefix Selection"
    The concatenation functions automatically determine the output `Prefix` type:
    - **Upgrades to `MPIDENSE`** when concatenating vectors horizontally (e.g., `hcat(x, y)`)
    - **Upgrades to `MPIDENSE`** if any input matrix has `Prefix=MPIDENSE`
    - Otherwise, preserves the first input's `Prefix`

    This ensures correctness since vectors are inherently dense, and horizontal concatenation
    of vectors produces a dense matrix. Vertical concatenation of vectors (`vcat`) preserves
    the sparse format since the result is still a single column.

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
```

## Row Ownership and Indexing

### Determining Owned Rows

Use `own_row()` to find which row indices are owned by the current rank:

```julia
A = Mat_uniform([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])

# Get ownership range for this rank
owned = own_row(A)  # e.g., 1:2 on rank 0, 3:4 on rank 1

println(io0(), "Rank $(MPI.Comm_rank(MPI.COMM_WORLD)) owns rows: $owned")
```

### Indexing Matrices

**Important:** You can only index rows that are owned by the current rank. Attempting to access non-owned rows will result in an error.

SafePETSc supports several indexing patterns:

```julia
A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0])
owned = own_row(A)

# ✓ Single element (owned row, any column)
if 2 in owned
    val = A[2, 3]  # Returns 6.0 on the rank that owns row 2
end

# ✓ Extract column (all ranks get their owned portion)
col_vec = A[:, 2]  # Returns distributed Vec with owned rows from column 2

# ✓ Row slice (owned row, column range)
if 3 in owned
    row_slice = A[3, 1:2]  # Returns [7.0, 8.0] on the rank that owns row 3
end

# ✓ Column slice (owned row range, single column)
if owned == 1:2
    col_slice = A[1:2, 2]  # Returns [2.0, 5.0] on the rank that owns these rows
end

# ✓ Submatrix (owned row range, column range)
if owned == 1:2
    submat = A[1:2, 2:3]  # Returns 2×2 Matrix on the rank that owns these rows
end

# ❌ WRONG - Accessing non-owned rows causes an error
val = A[4, 1]  # ERROR if rank doesn't own row 4!
```

**Indexing is non-collective** - each rank can independently access its owned rows without coordination.

### Use Cases for Indexing

Indexing is useful when you need to:
- Extract specific local values from owned rows
- Extract columns as distributed vectors
- Implement custom local operations
- Interface with non-PETSc code on owned data

```julia
# Extract owned portion for local processing
A = Mat_uniform(randn(100, 50))
owned = own_row(A)

# Get local submatrix
local_submat = A[owned, 1:10]  # First 10 columns of owned rows

# Process locally
local_norms = [norm(local_submat[i, :]) for i in 1:length(owned)]

# Aggregate across ranks if needed
max_norm = MPI.Allreduce(maximum(local_norms), max, MPI.COMM_WORLD)
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

## Row-wise Operations with map_rows

The `map_rows()` function applies a function to each row of distributed matrices or vectors, enabling powerful row-wise transformations.

### Basic Usage with Matrices

```julia
# Apply function to each row of a matrix
A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0])

# Compute row sums (returns Vec)
row_sums = map_rows(sum, A)  # Returns Vec([6.0, 15.0])

# Compute statistics per row (returns Mat)
stats = map_rows(row -> [sum(row), prod(row)]', A)
# Returns 2×2 Mat: [[6.0, 6.0]; [15.0, 48.0]]
```

**Note:** For matrices, the function receives a view of each row (like `eachrow`).

### Output Types

The return type depends on what your function returns:

- **Scalar** → Returns a `Vec` with m rows (one value per row)
- **Vector** → Returns a `Vec` with expanded rows (m*n total elements)
- **Adjoint Vector** (row vector) → Returns a `Mat` with m rows

```julia
B = Mat_uniform([1.0 2.0; 3.0 4.0; 5.0 6.0])

# Scalar output: Vec with 3 elements
means = map_rows(mean, B)

# Vector output: Vec with 3*2 = 6 elements
doubled = map_rows(row -> [row[1], row[2]], B)

# Matrix output: Mat with 3 rows, 2 columns
stats = map_rows(row -> [minimum(row), maximum(row)]', B)
```

### Combining Matrices and Vectors

Process matrices and vectors together row-wise:

```julia
B = Mat_uniform(randn(5, 3))
C = Vec_uniform(randn(5))

# Combine matrix rows with corresponding vector elements
result = map_rows((mat_row, vec_row) -> [sum(mat_row), prod(mat_row), vec_row[1]]', B, C)
# Returns 5×3 matrix with [row_sum, row_product, vec_value] per row
```

**Important:** All inputs must have compatible row partitions.

### Real-World Example

```julia
using Statistics

# Data matrix: each row is an observation
data = Mat_uniform(randn(1000, 50))

# Compute statistics for each observation
observation_stats = map_rows(data) do row
    [mean(row), std(row), minimum(row), maximum(row)]'
end
# Returns 1000×4 matrix with statistics per observation

# Convert to Julia matrix for analysis if small
if size(observation_stats, 1) < 10000
    stats_array = Matrix(observation_stats)
    # Further analysis with standard Julia tools
end
```

### Performance Notes

- `map_rows()` is a **collective operation** - all ranks must call it
- The function is applied only to locally owned rows on each rank
- Results are automatically assembled into a new distributed object
- More efficient than extracting rows individually and processing
- Works with both dense and sparse matrices (though sparse iteration may be less efficient)

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

### PETSc Options and the Prefix Type Parameter

SafePETSc matrices have a `Prefix` type parameter (e.g., `Mat{Float64,MPIAIJ}`) that determines both the matrix storage format and PETSc configuration. SafePETSc provides two built-in prefix types:

#### Built-in Prefix Types

- **`MPIAIJ`** (default): For sparse matrices
  - String prefix: `"MPIAIJ_"`
  - Default PETSc matrix type: `mpiaij` (MPI sparse matrix, compressed row storage)
  - Use for: Sparse linear algebra, iterative solvers
  - Memory efficient for matrices with few nonzeros per row

- **`MPIDENSE`**: For dense matrices
  - String prefix: `"MPIDENSE_"`
  - Default PETSc matrix type: `mpidense` (MPI dense matrix, row-major storage)
  - Use for: Dense linear algebra, direct solvers, operations like `eachrow`
  - Stores all matrix elements

**Important**: Unlike vectors (which are always dense internally), the `Prefix` parameter fundamentally changes matrix storage format. Choose `MPIDENSE` when you need dense storage, and `MPIAIJ` for sparse matrices.

#### Setting PETSc Options

Configure PETSc behavior for matrices with a specific prefix:

```julia
# Configure GPU-accelerated dense matrices
petsc_options_insert_string("-MPIDENSE_mat_type mpidense")
A = Mat_uniform(data; Prefix=MPIDENSE)

# Configure sparse matrices with custom solver
petsc_options_insert_string("-MPIAIJ_mat_type mpiaij")
B = Mat_uniform(sparse_data; Prefix=MPIAIJ)
```

The string prefix (e.g., `"MPIDENSE_"`, `"MPIAIJ_"`) is automatically prepended to option names when PETSc processes options for matrices with that prefix type.

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
4. **Reuse KSP Objects**: Create `KSP` once, reuse for multiple solves
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

## Converting to Julia Arrays

You can convert distributed `Mat` objects to native Julia arrays for interoperability, analysis, or export. SafePETSc provides two conversion options depending on matrix structure.

### Dense Matrix Conversion

Use `Matrix()` to convert to a dense Julia array:

```julia
# Create a distributed matrix
A = Mat_uniform([1.0 2.0; 3.0 4.0])

# Convert to Julia Matrix
A_dense = Matrix(A)  # Returns Matrix{Float64}
```

### Sparse Matrix Conversion

Use `sparse()` to convert to a sparse CSC matrix (preserves sparsity):

```julia
using SparseArrays

# Create sparse matrix
n = 10_000
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [2.0*ones(n); -ones(n-1); -ones(n-1)]
A = Mat_sum(sparse(I, J, V, n, n))

# Convert to CSC format (preserves sparsity)
A_csc = sparse(A)  # Returns SparseMatrixCSC{Float64, Int}

# Don't use Matrix() for sparse matrices!
A_dense = Matrix(A)  # Creates 10000×10000 dense array - wasteful!
```

### Checking Matrix Type with is_dense

Use `is_dense()` to determine if a matrix is stored in dense format:

```julia
using SparseArrays

A_dense = Mat_uniform([1.0 2.0; 3.0 4.0])
A_sparse = Mat_uniform(sparse([1, 2], [1, 2], [1.0, 4.0], 10, 10))

is_dense(A_dense)   # Returns true (matrix type contains "dense")
is_dense(A_sparse)  # Returns false (matrix type is sparse)

# Use appropriate conversion
if is_dense(A)
    A_julia = Matrix(A)      # Convert to dense
else
    A_julia = sparse(A)      # Convert to sparse CSC
end
```

The `is_dense()` function checks the PETSc matrix type string and returns `true` if it contains "dense" (case-insensitive). This handles various dense types like "seqdense", "mpidense", and vendor-specific dense matrix types.

### Important: Collective Operation

**Both conversion functions are collective operations** - all ranks must call them:

```julia
# ✓ CORRECT - All ranks participate
A_julia = Matrix(A)  # All ranks get the complete matrix

# ❌ WRONG - Will hang MPI!
if rank == 0
    A_julia = Matrix(A)  # Only rank 0 calls, others wait forever
end
```

After conversion, **all ranks receive the complete matrix**. The data is gathered from all ranks using MPI collective operations.

### When to Use Conversions

**Good use cases:**
- **Interoperability**: Pass data to packages that don't support PETSc
- **Small-scale analysis**: Compute eigenvalues, determinants, etc.
- **Data export**: Save results to files
- **Visualization**: Convert for plotting libraries

```julia
using LinearAlgebra

# Solve distributed system
A = Mat_uniform([2.0 1.0; 1.0 3.0])
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Convert for analysis (small matrix, so conversion is cheap)
A_julia = Matrix(A)
λ = eigvals(A_julia)       # Compute eigenvalues
det_A = det(A_julia)       # Compute determinant

println(io0(), "Eigenvalues: ", λ)
println(io0(), "Determinant: ", det_A)
```

**Avoid conversions for:**
- **Large matrices**: Gathers all data to all ranks (very expensive!)
- **Intermediate computations**: Keep data in PETSc format
- **Dense conversion of sparse matrices**: Use `sparse()` instead

### Performance Considerations

Conversion performance scales with:
- **Matrix size**: Larger matrices take longer to gather
- **Rank count**: More ranks means more communication
- **Sparsity**: Sparse conversions are more efficient than dense for large sparse matrices

```julia
# Small: fast conversion (< 1ms)
A_small = Mat_uniform(ones(100, 100))
A_julia = Matrix(A_small)

# Large sparse: use sparse() not Matrix()
n = 1_000_000
A_large_sparse = Mat_sum(sparse(..., n, n))
A_csc = sparse(A_large_sparse)    # Efficient
# A_dense = Matrix(A_large_sparse)  # Would allocate n×n dense array!
```

See [Converting to Native Julia Arrays](io.md#converting-to-native-julia-arrays) for more details and examples.

## Compatibility Notes

- **Transpose Reuse**: `transpose!(B, A)` requires that `B` was created via `Mat(A')` or has a compatible precursor
- **Matrix Multiplication Reuse**: `mul!(C, A, B)` requires pre-allocated `C` with correct partitions
- **Dense Operations**: Some operations (e.g., `\` with matrix RHS) require dense matrices

## See Also

- [`Mat_uniform`](@ref)
- [`Mat_sum`](@ref)
- [`spdiagm`](@ref)
- [`vcat`](@ref), [`hcat`](@ref), [`blockdiag`](@ref)
- [`SafePETSc.is_dense`](@ref)
- [Input/Output and Display](io.md) - Display and conversion operations
