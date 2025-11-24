# Input/Output and Display

When working with distributed MPI applications, managing output can be challenging. SafePETSc provides tools to handle display and IO operations effectively across multiple ranks.

## The Challenge with MPI Output

In MPI applications, each rank executes the same code. Without careful handling, printing operations can produce duplicate output:

```julia
# ❌ Bad: This prints from every rank
println("Solution: ", x)  # Prints 4 times with 4 ranks!
```

## The io0() Helper

SafePETSc provides the `io0()` function to easily control which ranks produce output:

```julia
# ✓ Good: Prints only on rank 0
println(io0(), "Solution: ", x)
```

### How io0() Works

`io0()` returns the provided IO stream if the current rank is in the set of selected ranks, and `devnull` on all other ranks:

```julia
io0(io=stdout; r::Set{Int}=Set{Int}([0]), dn=devnull)
```

**Parameters:**
- `io`: The IO stream to use (default: `stdout`)
- `r`: Set of ranks that should produce output (default: `Set{Int}([0])`)
- `dn`: The IO stream to return for non-selected ranks (default: `devnull`)

**Return value:**
- Returns `io` if the current rank is in `r`
- Returns `dn` otherwise

This allows all ranks to execute the same code, but only the selected ranks actually write output.

### Basic Usage

```julia
using SafePETSc
using MPI

SafePETSc.Init()

# Create and solve system
A = Mat_uniform([2.0 1.0; 1.0 3.0])
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Print only on rank 0 (default)
println(io0(), "Solution computed")
println(io0(), x)  # Displays the vector
```

### Selecting Different Ranks

You can specify which ranks should produce output:

```julia
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Print from rank 0 (default)
println(io0(), "This is from rank 0")

# Print from rank 1
println(io0(r=Set([1])), "This is from rank 1")

# Print from the last rank
println(io0(r=Set([nranks-1])), "This is from the last rank")

# Print from multiple ranks (e.g., ranks 0 and 2)
println(io0(r=Set([0, 2])), "This is from ranks 0 and 2")
```

### Writing to Files

`io0()` works with any IO stream, including files:

```julia
using LinearAlgebra  # For norm()

# Write to file only on rank 0 (assumes x, A, b from previous example)
open("results.txt", "w") do f
    println(io0(f), "Results:")
    println(io0(f), "Solution: ", x)
    println(io0(f), "Residual norm: ", norm(A*x - b))
end
```

## Display Methods and show()

SafePETSc implements Julia's `show()` interface for vectors and matrices, allowing them to be displayed naturally.

### Automatic Conversion

When you display a `Vec` or `Mat`, it is automatically converted to a Julia array for display:

```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# All of these work and trigger conversion:
println(io0(), v)           # Uses show(io, v)
display(v)                  # Uses show(io, MIME"text/plain"(), v)
@show v                     # Macro uses show()
```

### How show() Works

The `show()` methods are **collective operations** - all ranks participate:

1. All ranks call the conversion function (e.g., `Vector(v)` or `Matrix(A)`)
2. Data is gathered from all ranks to all ranks
3. Each rank has the complete array
4. `show()` displays it on each rank's IO stream

When combined with `io0()`, only one rank actually displays:

```julia
# All ranks execute this, but only rank 0 prints
println(io0(), v)

# Behind the scenes:
# 1. All ranks: v_julia = Vector(v)  (collective)
# 2. All ranks: show(io0(), v_julia)  (rank 0 shows, others write to devnull)
```

**⚠️ WARNING: Never Wrap show() in Rank-Dependent Conditionals!**

Because `show()` is collective, wrapping it in a rank-dependent conditional will desynchronize the MPI cluster and cause it to hang:

```julia
# ❌ WRONG - This will hang MPI!
if rank == 0
    println(v)  # Only rank 0 calls Vector(v), others wait forever
end

# ❌ WRONG - This will also hang!
if rank == 0
    @show v  # Only rank 0 participates in collective operations
end

# ✓ CORRECT - All ranks participate, only rank 0 prints
println(io0(), v)  # All ranks call Vector(v), rank 0 displays
```

If you want rank-specific output, **always use `io0()`** - it ensures all ranks participate in the collective operations while controlling which ranks produce the output.

### Dense vs Sparse Display

Matrices automatically choose the appropriate display format:

```julia
# Dense matrix - displays as Matrix
A_dense = Mat_uniform([1.0 2.0; 3.0 4.0])
println(io0(), A_dense)  # Shows dense format

# Sparse matrix - displays using sparse() format
using SparseArrays
A_sparse = Mat_uniform(sparse([1, 2], [1, 2], [1.0, 4.0], 10, 10))
println(io0(), A_sparse)  # Shows sparse format with (i, j, value) triples
```

### MIME Type Support

SafePETSc supports MIME type display for rich output in notebooks and other environments:

```julia
# In Jupyter notebooks, Pluto, etc.
display(v)  # Uses MIME"text/plain"
display(A)  # Uses MIME"text/plain"

# The three-argument show method is called:
# show(io, MIME"text/plain"(), v)
```

## Converting to Native Julia Arrays

SafePETSc provides constructors to explicitly convert distributed PETSc objects to native Julia arrays. This is useful for interoperability with other Julia packages, data export, or small-scale analysis.

### Available Conversions

```julia
# Vector: Convert Vec to Julia Vector
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
v_julia = Vector(v)  # Returns Vector{Float64}

# Matrix: Convert Mat to Julia Matrix (dense)
A = Mat_uniform([1.0 2.0; 3.0 4.0])
A_dense = Matrix(A)  # Returns Matrix{Float64}

# Sparse: Convert Mat to SparseMatrixCSC
using SparseArrays
A_sparse = Mat_uniform(sparse([1, 2], [1, 2], [1.0, 4.0], 10, 10))
A_csc = sparse(A_sparse)  # Returns SparseMatrixCSC{Float64, Int}
```

### Universal Conversion with J()

The `J()` function provides a uniform interface for converting any SafePETSc object to its native Julia equivalent. It automatically chooses the appropriate conversion based on the input type:

```julia
# J() automatically selects the right conversion
v = Vec_uniform([1.0, 2.0, 3.0])
v_julia = J(v)  # Vector{Float64} (same as Vector(v))

A_dense = Mat_uniform([1.0 2.0; 3.0 4.0])
A_julia = J(A_dense)  # Matrix{Float64} (same as Matrix(A))

A_sparse = Mat_uniform(sparse([1.0 0.0; 0.0 2.0]))
A_julia = J(A_sparse)  # SparseMatrixCSC (same as sparse(A))

# Scalars and other types pass through unchanged
x = J(3.14)  # 3.14
```

`J()` is particularly useful in generic code where you want to convert multiple PETSc objects without knowing their specific types:

```julia
# Convert all arguments at once using broadcasting
args_julia = J.((A, b, x))  # Converts each to its Julia equivalent
```

**Conversion rules:**
- `Vec{T}` → `Vector{T}`
- `Mat{T}` (dense) → `Matrix{T}`
- `Mat{T}` (sparse) → `SparseMatrixCSC{T,Int}`
- Adjoint types preserve the adjoint wrapper
- Non-PETSc types pass through unchanged

### Important: Collective Operations

**All conversion functions are collective operations** - every rank must call them:

```julia
# ✓ CORRECT - All ranks participate
v_julia = Vector(v)  # All ranks get the complete vector

# ❌ WRONG - Will hang MPI!
if rank == 0
    v_julia = Vector(v)  # Only rank 0 calls, others wait forever
end
```

After conversion, **all ranks receive the complete array**. This is necessary because the conversion gathers distributed data from all ranks.

### When to Use Conversions

**Good use cases:**
1. **Interoperability**: Pass data to packages that don't support PETSc
2. **Small-scale analysis**: Compute properties on small matrices/vectors
3. **Data export**: Save results to files in native Julia formats
4. **Visualization**: Convert for plotting libraries

```julia
using Plots

# Solve distributed system
A = Mat_uniform([2.0 1.0; 1.0 3.0])
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Convert for plotting (small vector, so conversion is cheap)
x_julia = Vector(x)
plot(x_julia, label="Solution")
savefig(io0(), "solution.png")  # Only rank 0 saves
```

**When to avoid:**
1. **Large datasets**: Conversion gathers all data to all ranks (expensive!)
2. **Intermediate computations**: Keep data in PETSc format for efficiency
3. **Repeated access**: Don't convert in loops

```julia
# ❌ BAD - Expensive conversion in loop
for i in 1:1000
    v_julia = Vector(v)  # Wasteful! Gathers data every iteration
    process(v_julia[1])
end

# ✓ BETTER - Convert once if needed
v_julia = Vector(v)
for i in 1:1000
    process(v_julia[1])
end

# ✓ BEST - Don't convert at all if possible
# Use PETSc operations directly instead
```

### Performance Considerations

Conversion performance scales with:
- **Data size**: Larger vectors/matrices take longer to gather
- **Rank count**: More ranks means more communication
- **Network**: Collective operations require all-to-all communication

```julia
# Small: fast conversion (< 1ms)
v_small = Vec_uniform(ones(100))
v_julia = Vector(v_small)

# Large: expensive conversion (can be seconds!)
v_large = Vec_uniform(ones(10_000_000))
v_julia = Vector(v_large)  # All 10M elements sent to all ranks
```

### Working with Converted Data

After conversion, you have standard Julia arrays:

```julia
using LinearAlgebra
using Statistics

# Convert PETSc objects
x = Vec_uniform([1.0, 2.0, 3.0, 4.0])
A = Mat_uniform([2.0 1.0; 1.0 3.0])

x_julia = Vector(x)
A_julia = Matrix(A)

# Use with any Julia package
μ = mean(x_julia)           # Statistics
σ = std(x_julia)
λ = eigvals(A_julia)        # LinearAlgebra
det_A = det(A_julia)

println(io0(), "Mean: $μ, Std: $σ")
println(io0(), "Eigenvalues: ", λ)
```

### Sparse Matrix Conversion

For sparse matrices, use `sparse()` to preserve sparsity:

```julia
using SparseArrays

# Create large sparse matrix
n = 10_000
I = [1:n; 1:n-1; 2:n]
J = [1:n; 2:n; 1:n-1]
V = [2.0*ones(n); -ones(n-1); -ones(n-1)]
A = Mat_sum(sparse(I, J, V, n, n))

# Convert to CSC format (preserves sparsity)
A_csc = sparse(A)  # SparseMatrixCSC - efficient!

# Don't use Matrix() for sparse matrices!
A_dense = Matrix(A)  # Creates 10000×10000 dense array - wasteful!

# Work with sparse format
nnz_A = nnz(A_csc)
println(io0(), "Nonzeros: $nnz_A out of $(n^2) entries")
```

### Example: Data Export

```julia
using SafePETSc
using MPI
using SparseArrays
using DelimitedFiles  # For writedlm

SafePETSc.Init()

# Solve system
A = Mat_uniform([2.0 1.0; 1.0 3.0])
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Convert to Julia arrays (collective operation)
x_julia = Vector(x)
A_julia = Matrix(A)

# Write to files (only rank 0)
open("solution.txt", "w") do f
    writedlm(io0(f), x_julia)
end

open("matrix.txt", "w") do f
    writedlm(io0(f), A_julia)
end

println(io0(), "Data exported to solution.txt and matrix.txt")
```

## Advanced IO Patterns

### Per-Rank Output Files

Sometimes you want each rank to write its own file:

```julia
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Each rank writes to its own file - don't use io0()!
open("output_rank_$rank.txt", "w") do f
    println(f, "Output from rank $rank")
    println(f, "Local data: ", get_local_data())
end
```

In this case, **don't use `io0()`** because you want all ranks to write.

### Debugging Output

For debugging, you might temporarily want output from all ranks:

```julia
# Debugging: show output from all ranks
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
println("Rank $rank: x = ", x)  # All ranks print (useful for debugging)

# Production: only rank 0
println(io0(), "Solution: ", x)  # Only rank 0 prints
```

## Best Practices

1. **Use `io0()` for normal output**: Always use `println(io0(), ...)` for user-facing output
2. **Show vectors and matrices with io0()**: `println(io0(), x)` not `println(x)`
3. **Per-rank files don't need io0()**: When each rank writes its own file, write directly
4. **Debugging is the exception**: Temporarily allowing all-rank output is fine for debugging
5. **File IO works with io0()**: `open("file.txt", "w") do f; println(io0(f), ...) end`

## Example: Complete IO Pattern

```julia
using SafePETSc
using MPI
using LinearAlgebra

SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Setup and solve
A = Mat_uniform([2.0 1.0; 1.0 3.0])
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Standard output (rank 0 only)
println(io0(), "="^60)
println(io0(), "Parallel Linear Solve Complete")
println(io0(), "  MPI ranks: ", nranks)
println(io0(), "  System size: ", length(b))
println(io0(), "="^60)

# Display results
println(io0(), "\nSolution vector:")
println(io0(), x)

println(io0(), "\nResidual norm: ", norm(A*x - b))

# Write detailed results to file (rank 0 only)
open("results.txt", "w") do f
    println(io0(f), "Detailed Results")
    println(io0(f), "="^60)
    println(io0(f), "\nMatrix A:")
    println(io0(f), A)
    println(io0(f), "\nRight-hand side b:")
    println(io0(f), b)
    println(io0(f), "\nSolution x:")
    println(io0(f), x)
end

println(io0(), "\nResults written to results.txt")
```

This pattern provides clean, professional output without duplication across MPI ranks.
