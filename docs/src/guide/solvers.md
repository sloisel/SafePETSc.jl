# Linear Solvers

SafePETSc provides linear solver functionality through PETSc's KSP (Krylov Subspace) interface, wrapped with automatic memory management.

!!! note "Sparse Matrices"
    The default direct solvers (MUMPS/STRUMPACK) are designed for **sparse matrices**. The coefficient matrix `A` should be sparse (MPIAIJ). Dense coefficient matrices are not supported by the sparse direct solvers.

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

## Configuring Solvers

### PETSc Options

Control solver behavior via PETSc options:

```julia
# Global configuration
petsc_options_insert_string("-ksp_type gmres")
petsc_options_insert_string("-ksp_rtol 1e-8")
petsc_options_insert_string("-pc_type bjacobi")

# With prefix for specific solvers
# First define a custom prefix type (advanced)
struct MyPrefix end
SafePETSc.prefix(::Type{MyPrefix}) = "my_"

petsc_options_insert_string("-my_ksp_type cg")
A = Mat_uniform(data; Prefix=MyPrefix)
ksp = KSP(A)  # Will use CG
```

Common KSP options:
- `-ksp_type`: KSP type (cg, gmres, bcgs, etc.)
- `-ksp_rtol`: Relative tolerance
- `-ksp_atol`: Absolute tolerance
- `-ksp_max_it`: Maximum iterations
- `-pc_type`: Preconditioner (jacobi, bjacobi, ilu, etc.)

### Monitoring Convergence

```julia
petsc_options_insert_string("-ksp_monitor")
petsc_options_insert_string("-ksp_converged_reason")

x = A \ b
# PETSc will print convergence information
```

## KSP Types

SafePETSc supports all PETSc KSP types. Common choices:

### Direct Methods
```julia
# For small to medium problems
petsc_options_insert_string("-ksp_type preonly -pc_type lu")
x = A \ b
```

### Iterative Methods
```julia
# Conjugate Gradient (symmetric positive definite)
petsc_options_insert_string("-ksp_type cg -pc_type jacobi")

# GMRES (general nonsymmetric)
petsc_options_insert_string("-ksp_type gmres -ksp_gmres_restart 30")

# BiCGStab
petsc_options_insert_string("-ksp_type bcgs")
```

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

### Iterative KSP with Monitoring

```julia
using SafePETSc

SafePETSc.Init()

# Configure solver
petsc_options_insert_string("-ksp_type cg")
petsc_options_insert_string("-ksp_rtol 1e-10")
petsc_options_insert_string("-ksp_monitor")
petsc_options_insert_string("-pc_type jacobi")

# Build system (e.g., Laplacian)
n = 1000
diag = Vec_uniform(2.0 * ones(n))
off = Vec_uniform(-1.0 * ones(n-1))
A = spdiagm(-1 => off, 0 => diag, 1 => off)

b = Vec_uniform(ones(n))

# Solve (will print iteration info)
x = A \ b
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

1. **Reuse KSP Objects**: Create `KSP` once for multiple solves
2. **Choose Appropriate Method**: Direct for small problems, iterative for large
3. **Tune Preconditioner**: Can dramatically affect convergence
4. **Monitor Convergence**: Use `-ksp_monitor` to tune parameters
5. **GPU Acceleration**: Set PETSc options for GPU execution

```julia
# GPU configuration example
petsc_options_insert_string("-mat_type aijcusparse")
petsc_options_insert_string("-vec_type cuda")
```

## KSP Properties

Check solver dimensions:

```julia
ksp = KSP(A)

m, n = size(ksp)  # Matrix dimensions
m = size(ksp, 1)  # Rows
n = size(ksp, 2)  # Columns
```

## Troubleshooting

### Convergence Issues

```julia
# Increase iterations
petsc_options_insert_string("-ksp_max_it 10000")

# Relax tolerance
petsc_options_insert_string("-ksp_rtol 1e-6")

# Try different solver/preconditioner
petsc_options_insert_string("-ksp_type gmres -pc_type asm")

# View solver details
petsc_options_insert_string("-ksp_view")
```

### Memory Issues

```julia
# Use iterative method instead of direct
petsc_options_insert_string("-ksp_type cg")

# Reduce GMRES restart
petsc_options_insert_string("-ksp_gmres_restart 10")
```

### Assertion Failures

Ensure:
- Matrix is square for `\` operator
- Partitions match (A.row_partition == b.row_partition)
- Same prefix on all objects

## See Also

- [`SafePETSc.KSP`](@ref)
