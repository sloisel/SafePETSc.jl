# Solvers API Reference

Linear solver functionality in SafePETSc.

## Type

```@docs
SafePETSc.KSP
```

## Initialization

```@docs
SafePETSc.Init
SafePETSc.Initialized
SafePETSc.petsc_options_insert_string
SafePETSc.has_strumpack
```

## Linear Solves

### Direct Solve

```julia
# Vector RHS
x = A \ b                              # Solve Ax = b
x = A' \ b                             # Solve A^T x = b

# Matrix RHS (must be dense)
X = A \ B                              # Solve AX = B
X = A' \ B                             # Solve A^T X = B
```

### Reusable Solver with inv(A)

```julia
# Create reusable solver (factorization happens here)
Ainv = inv(A)                          # Returns KSP object

# Left multiplication (solve Ax = b)
x = Ainv * b                           # Solve Ax = b
X = Ainv * B                           # Solve AX = B

# Transpose solver
Aitinv = inv(A')                       # Returns Adjoint{KSP}
x = Aitinv * b                         # Solve A'x = b
X = Aitinv * B                         # Solve A'X = B

# Right multiplication (solve xA = b)
xt = b' * Ainv                         # Solve x'A = b' (returns adjoint)
X = B * Ainv                           # Solve XA = B
X = B' * Ainv                          # Solve XA = B'
```

### In-Place Solve

```julia
# Vector RHS
LinearAlgebra.ldiv!(x, A, b)           # x = A \ b
LinearAlgebra.ldiv!(ksp, x, b)         # Using reusable solver

# Matrix RHS (must be dense)
LinearAlgebra.ldiv!(X, A, B)           # X = A \ B
LinearAlgebra.ldiv!(ksp, X, B)         # Using reusable solver
```

### Right Division

```julia
# Vector
x_adj = b' / A                         # Solve x^T A = b^T

# Matrix (must be dense)
X = B / A                              # Solve XA = B
X = B / A'                             # Solve XA^T = B
```

## Properties

```julia
m, n = size(ksp)                       # KSP matrix dimensions
```
