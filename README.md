# SafePETSc.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/SafePETSc.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/SafePETSc.jl/dev/) [![CI](https://github.com/sloisel/SafePETSc.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sloisel/SafePETSc.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/sloisel/SafePETSc.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/SafePETSc.jl)

A Julia package that makes distributed [PETSc](https://petsc.org/) linear algebra feel like native Julia.

## Overview

SafePETSc.jl lets you write natural Julia expressions like `A * B + C`, `A \ b`, or `y .= 2 .* x .+ 3` for distributed PETSc matrices and vectors. Instead of managing PETSc's verbose C API with explicit pointer manipulation and multi-step operations, you get Julia's familiar array interface: arithmetic operators, broadcasting, standard constructors (`spdiagm`, `vcat`, `hcat`), and iteration patterns. The package handles PETSc's complexity and automatically manages object lifecycles across MPI ranks, so your code stays readable and errors like segfaults from pointer mistakes are eliminated.

## Features

- **Julia-Native Syntax**: Use `A * B`, `A \ b`, `A + B`, `A'` just like regular Julia matrices
- **Broadcasting**: Full support for `.+`, `.*`, `.=` and function broadcasting `f.(x)`
- **Standard Constructors**: `spdiagm`, `vcat`, `hcat`, `blockdiag` work on distributed matrices
- **Linear Algebra**: Matrix multiplication, solving (`\`, `/`), transpose, in-place operations
- **Automatic Memory Management**: Objects are cleaned up automatically across MPI ranks
- **Vector Pooling**: Temporary vectors are reused for efficiency (configurable with `ENABLE_VEC_POOL[]`)
- **Iteration**: Use `eachrow(A)` for row-wise processing

## Installation

```julia
using Pkg
Pkg.add("SafePETSc")
```

Or in development mode:

```julia
using Pkg
Pkg.develop(url="https://github.com/sloisel/SafePETSc.jl")
```

## Quick Start

```julia
using SafePETSc
using MPI
using SparseArrays

# Initialize MPI
MPI.Init()

# Create distributed vectors and matrices
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
A = Mat_uniform(sparse([1.0 2.0; 3.0 4.0]))
B = Mat_uniform(sparse([5.0 6.0; 7.0 8.0]))

# Linear algebra (just like regular Julia!)
C = A * B           # Matrix multiplication
y = A * v           # Matrix-vector product
x = A \ v           # Linear solve
z = A' * v          # Transpose multiply

# Display results (io0() prints only on rank 0 to avoid duplicate output)
println(io0(), "Solution: ", x)
println(io0(), "Result vector: ", y)

# Broadcasting
w = 2.0 .* v .+ 3.0      # Scalar operations
result = v .+ w          # Element-wise addition

# Matrix construction
D = spdiagm(0 => [1.0, 2.0], 1 => [0.5])  # Diagonal matrix
E = vcat(A, B)                             # Vertical concatenation
F = hcat(A, B)                             # Horizontal concatenation

# In-place operations (more efficient)
mul!(y, A, v)       # y = A * v (no allocation)
ldiv!(x, A, v)      # x = A \ v (solve in-place)

# Resources are automatically cleaned up; vectors are pooled by default
MPI.Finalize()
```

## Core Types

SafePETSc provides three main distributed types that wrap PETSc objects:

- **Vec{T}**: Distributed vectors with automatic lifetime management and pooling
- **Mat{T,Prefix}**: Distributed matrices supporting dense and sparse formats
- **KSP{T,Prefix}**: Linear solver contexts for efficient repeated solves

### Printing and Display

When printing distributed objects in MPI applications, use `io0()` to avoid duplicate output from all ranks:

```julia
# Print only on rank 0 (default)
println(io0(), "Solution: ", x)
println(io0(), v)  # Displays vector contents

# Print on a specific rank
println(io0(r=Set([2])), "Message from rank 2")

# Print on multiple ranks
println(io0(r=Set([0, 2])), "Message from ranks 0 and 2")

# Write to file only on rank 0
open("results.txt", "w") do f
    println(io0(f), "Results: ", x)
end
```

The `io0()` function returns the provided IO stream (default `stdout`) if the current rank is in the specified set of ranks, and `devnull` on all other ranks. This ensures output appears only from selected ranks while allowing all ranks to execute the same code.

### Memory Management

PETSc objects are automatically cleaned up when they go out of scope through distributed garbage collection. No explicit cleanup is required.

## Testing

The package uses a dual-file testing approach for MPI:

```bash
# Run all tests (automatically spawns 4 MPI ranks)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific MPI test directly
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_mat_uniform.jl`)'
```

The test suite includes:
- Reference counting across MPI ranks
- Matrix and vector operations
- Linear solver correctness
- Resource cleanup verification

## Development

```bash
# Activate package environment
julia --project=.

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Start REPL with package loaded
julia --project=. -e 'using SafePETSc'
```

## Requirements

- Julia ≥ 1.10 (LTS)
- MPI implementation (OpenMPI, MPICH, etc.)
- PETSc ≥ 0.3.1

## Author

Sébastien Loisel (S.Loisel@hw.ac.uk)

## License

MIT License - see [LICENSE](LICENSE) for details.
