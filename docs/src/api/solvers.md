# Solvers API Reference

Linear solver functionality in SafePETSc.

## Type

```@docs
SafePETSc.Solver
```

## Initialization

```@docs
SafePETSc.Init
SafePETSc.Initialized
SafePETSc.petsc_options_insert_string
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
m, n = size(ksp)                       # Solver matrix dimensions
```
