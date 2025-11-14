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
- **GPU-Friendly**: Bulk operations that work efficiently on GPU devices
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

## Architecture (Implementation Details)

*This section describes the internal implementation for developers and contributors. Users don't need to understand these details to use SafePETSc effectively.*

### SafeMPI Module

The package implements a reference-counting garbage collection system for distributed objects:

- **DistributedRefManager**: Coordinates reference counting across MPI ranks
  - All ranks run the same ID allocation logic and keep the same `free_ids` vector, ensuring deterministic IDs without a dedicated root
  - Each rank enqueues local releases without MPI; at safe points, ranks collectively `Allgather` release IDs and update counters identically
  - Objects ready for destruction are identified deterministically on all ranks; destruction runs collectively without an extra broadcast
  - Automatic ID recycling prevents unbounded growth via the shared `free_ids` vector

- **DRef{T}**: Generic wrapper for distributed objects requiring coordinated destruction
  - Only works with types that explicitly opt-in via the trait system
  - Finalizers automatically call `release!()` when garbage collected

### SafePETSc Module

Provides safe wrappers around PETSc functionality:

- **Vec{T}**: Distributed vectors with automatic lifetime management
- **Mat{T}**: Distributed matrices with GPU-friendly operations
- **KSP{T}**: Linear solver contexts

### Trait-Based Destruction System

Types must explicitly opt-in to distributed management:

```julia
# Enable distributed management
destroy_trait(::Type{YourType}) = CanDestroy()

# Implement cleanup logic
destroy_obj!(obj::YourType) = ...
```

This prevents accidental misuse and provides clear error messages when attempting to wrap unsupported types.

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

## GPU Support

SafePETSc prioritizes GPU-compatible operations by using PETSc's native bulk functions:

- `MatConvert`: Type conversion (e.g., sparse to dense) preserving GPU storage
- `MatTranspose`: Efficient GPU-based transposition
- `MatMatMult`: GPU-accelerated matrix multiplication

Avoid element-by-element extraction patterns which cause excessive GPU↔CPU transfers.

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

- Julia ≥ 1.11
- MPI implementation (OpenMPI, MPICH, etc.)
- PETSc ≥ 0.3.1

## Author

Sébastien Loisel (S.Loisel@hw.ac.uk)

## License

MIT License - see [LICENSE](LICENSE) for details.
