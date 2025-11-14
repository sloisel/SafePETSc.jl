# Matrices API Reference

Distributed matrix operations in SafePETSc.

## Type

```@docs
SafePETSc.Mat
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
# Iterate over rows (dense matrices only)
for row in eachrow(A)
    # row is a view of the matrix row
    process(row)
end
```

## Block Matrix Products

```@docs
SafePETSc.BlockProduct
SafePETSc.calculate!
```
