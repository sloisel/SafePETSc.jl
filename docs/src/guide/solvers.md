# Linear Solvers

SafePETSc provides linear solver functionality through PETSc's KSP interface, using direct (sparse LU factorization) solvers.

!!! note "Direct Solvers"
    SafePETSc uses **direct solvers** (sparse LU factorization), not iterative methods. This means:
    - No convergence parameters to tune
    - No preconditioner selection needed
    - Exact solutions (up to floating-point precision)
    - The coefficient matrix `A` should be sparse (MPIAIJ prefix, which is the default)

## Basic Usage

### Direct Solve

The simplest way to solve a linear system:

```julia
# Solve Ax = b
x = A \ b

# Solve A^T x = b
x = A' \ b
```

This creates a solver internally, solves the system, and cleans up automatically.

### Multiple Right-Hand Sides

```julia
# Matrix RHS: solve AX = B
X = A \ B

# Transpose: solve A^T X = B
X = A' \ B
```

!!! warning "Dense RHS Required"
    For matrix right-hand sides, `B` must be a dense matrix (created with `Prefix=MPIDENSE`). The coefficient matrix `A` should be sparse (default MPIAIJ).

## Reusable Solvers with inv(A)

For multiple solves with the same matrix, use `inv(A)` to get a reusable solver:

```julia
# Create solver with factorization
Ainv = inv(A)            # Factorization happens here

# Solve multiple systems efficiently
x1 = Ainv * b1           # Solve A*x1 = b1 (reuses factorization)
x2 = Ainv * b2           # Solve A*x2 = b2 (reuses factorization)

# Transpose solves
Aitinv = inv(A')         # Solver for transpose
y = Aitinv * c           # Solve A'*y = c

# Right-hand side multiplication
xt = b' * Ainv           # Solve x'*A = b' (returns row vector)
X = B * Ainv             # Solve X*A = B
```

The `inv(A)` function returns a `KSP` solver object that represents the "inverse" of `A`. The expensive factorization or preconditioner setup happens once when `inv(A)` is called, and all subsequent multiplications reuse this work.

### Supported Operations

| Operation | Solves | Returns |
|-----------|--------|---------|
| `inv(A) * b` | Ax = b | Vec x |
| `inv(A) * B` | AX = B | Mat X |
| `inv(A') * b` | A'x = b | Vec x |
| `inv(A') * B` | A'X = B | Mat X |
| `b' * inv(A)` | x'A = b' | Adjoint Vec |
| `B * inv(A)` | XA = B | Mat X |
| `B' * inv(A)` | XA = B' | Mat X |

### Benefits of Reuse

- **Performance**: Avoids repeated factorization/preconditioner setup
- **Memory**: Single solver object instead of multiple temporary solvers
- **Configuration**: Set PETSc options once

### Alternative: KSP Constructor

You can also create a reusable solver directly with the `KSP` constructor:

```julia
# Create solver once
ksp = KSP(A)

# Solve multiple systems
x1 = zeros_like(b1)
ldiv!(ksp, x1, b1)

x2 = zeros_like(b2)
ldiv!(ksp, x2, b2)

# KSP is cleaned up automatically when ksp goes out of scope
```

## In-Place Operations

For pre-allocated result vectors/matrices:

```julia
# Vector solve
x = zeros_like(b)
ldiv!(x, A, b)  # x = A \ b (creates solver internally)

# With reusable solver
ldiv!(ksp, x, b)  # Reuse solver

# Matrix solve
X = zeros_like(B)
ldiv!(X, A, B)  # Solve AX = B
ldiv!(ksp, X, B)  # With reusable solver
```

## Right Division

Solve systems where the unknown is on the left:

```julia
# Solve x^T A = b^T (equivalent to A^T x = b)
x_adj = b' / A  # Returns adjoint vector

# Solve XA = B
X = B / A

# Transpose: solve XA^T = B
X = B / A'
```

Note: `B` and `X` must be dense matrices.

## Examples

### Basic Linear System

```julia
using SafePETSc
using MPI

SafePETSc.Init()

# Create a simple system
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

A = Mat_uniform(A_dense)
b = Vec_uniform(ones(n))

# Solve
x = A \ b

# Check residual
r = b - A * x
# (In practice, use PETSc's built-in convergence monitoring)
```

### Multiple Solves

```julia
using SafePETSc

SafePETSc.Init()

# System matrix
A = Mat_uniform(...)

# Create solver once
ksp = KSP(A)

# Solve many RHS vectors
for i in 1:100
    b = Vec_uniform(rhs_data[i])
    x = zeros_like(b)
    ldiv!(ksp, x, b)
    results[i] = extract_result(x)
end

# KSP automatically cleaned up
```

### Block Solves

```julia
using SafePETSc

SafePETSc.Init()

# System matrix
A = Mat_uniform(...)

# Multiple right-hand sides as columns of a dense matrix
B = Mat_uniform(rhs_matrix)  # Must be dense

# Solve all systems at once
X = A \ B

# Each column of X is a solution
```

## Performance Tips

1. **Reuse Factorizations**: Use `inv(A)` or `KSP(A)` when solving multiple systems with the same matrix. The expensive factorization happens once and is reused for each solve.

2. **Use Sparse Matrices**: Direct solvers are designed for sparse matrices. Dense coefficient matrices will be slow and memory-intensive.

## KSP Properties

Check solver dimensions:

```julia
ksp = KSP(A)

m, n = size(ksp)  # Matrix dimensions
m = size(ksp, 1)  # Rows
n = size(ksp, 2)  # Columns
```

## Troubleshooting

### Singular Matrix

If the direct solver fails, the matrix may be singular or nearly singular. Check:
- Matrix has full rank
- No zero rows or columns
- Appropriate scaling

### Dimension Mismatch

Ensure:
- Matrix `A` is square for `\` operator
- Vector `b` has same row partition as `A`
- For `inv(A) * B`, columns of `B` match rows of `A`

## See Also

- [`SafePETSc.KSP`](@ref)
