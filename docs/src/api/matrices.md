# Matrices API Reference

Distributed matrix operations in SafePETSc.

## Type

```@docs
SafePETSc.Mat
```

## Prefix Types

The `Prefix` type parameter determines matrix storage format and PETSc configuration:

```@docs
SafePETSc.MPIAIJ
SafePETSc.MPIDENSE
SafePETSc.prefix
```

## Constructors

```@docs
SafePETSc.Mat_uniform
SafePETSc.Mat_sum
```

## Concatenation

```@docs
Base.cat
Base.vcat
Base.hcat
SparseArrays.blockdiag
```

## Sparse Diagonal Matrices

```@docs
SparseArrays.spdiagm
```

## Conversion and Display

Convert distributed matrices to Julia arrays for inspection and display:

```@docs
Base.Matrix(::SafePETSc.Mat)
SparseArrays.sparse(::SafePETSc.Mat)
SafePETSc.is_dense
```

Display methods (automatically used by `println`, `display`, etc.):
- `show(io::IO, A::Mat)` - Display matrix contents (uses dense or sparse format based on type)
- `show(io::IO, mime::MIME, A::Mat)` - Display with MIME type support

## Utilities

```@docs
SafePETSc.own_row(::SafePETSc.Mat)
```

## Row-wise Operations

See [`map_rows`](@ref) in the [Vectors API](vectors.md#Row-wise-Operations) - works with both vectors and matrices.

## Indexing

Non-collective element and range access:

```@docs
Base.getindex(::SafePETSc.Mat, ::Colon, ::Int)
Base.getindex(::SafePETSc.Mat, ::Int, ::Int)
Base.getindex(::SafePETSc.Mat, ::UnitRange{Int}, ::Int)
Base.getindex(::SafePETSc.Mat, ::Int, ::UnitRange{Int})
Base.getindex(::SafePETSc.Mat, ::UnitRange{Int}, ::UnitRange{Int})
```

## Operations

### Linear Algebra

```julia
# Matrix-vector multiplication
y = A * x
LinearAlgebra.mul!(y, A, x)

# Matrix-matrix multiplication
C = A * B
LinearAlgebra.mul!(C, A, B)

# Transpose
B = A'
B = Mat(A')                            # Materialize transpose
LinearAlgebra.transpose!(B, A)         # In-place transpose

# Adjoint-vector multiplication
w = v' * A
LinearAlgebra.mul!(w, v', A)
```

### Properties

```julia
T = eltype(A)                          # Element type
m, n = size(A)                         # Dimensions
m = size(A, 1)                         # Rows
n = size(A, 2)                         # Columns
```

### Iteration

```julia
# Iterate over rows (works for both dense and sparse matrices)
for row in eachrow(A)
    # For dense (MPIDENSE): row is a view of the matrix row
    # For sparse (MPIAIJ): row is a dense Vector representation
    process(row)
end
```

## Block Matrix Products

```@docs
SafePETSc.BlockProduct
SafePETSc.calculate!
```
