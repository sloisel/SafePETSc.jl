# SafePETSc.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/SafePETSc.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/SafePETSc.jl/dev/) [![CI](https://github.com/sloisel/SafePETSc.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sloisel/SafePETSc.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/sloisel/SafePETSc.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/SafePETSc.jl)

A Julia package providing safe distributed reference management for MPI-based parallel computing with PETSc.

## Overview

SafePETSc.jl wraps [PETSc](https://petsc.org/) functionality with automatic distributed garbage collection, ensuring that objects distributed across MPI ranks are released only when all ranks have released their references. By default, PETSc vectors are pooled for reuse rather than destroyed immediately; you can disable pooling or clear the pool to free memory. This prevents common pitfalls in parallel computing such as premature destruction, memory leaks, and race conditions.

## Features

- **Distributed Reference Counting**: Automatic lifetime management for objects shared across MPI ranks (mirrored counters on all ranks)
- **Trait-Based Safety**: Explicit opt-in system prevents accidental misuse
- **PETSc Integration**: Seamless wrapping of PETSc vectors, matrices, and linear solvers
- **GPU-Friendly**: Prioritizes bulk operations that work efficiently on GPU devices
- **Finalizer-Based Cleanup**: Automatic resource release via Julia's garbage collector at safe points
- **Vector Pooling**: Released vectors are returned to a reuse pool by default; disable with `ENABLE_VEC_POOL[] = false` or call `clear_vec_pool!()` to free pooled vectors
- **ID Recycling**: Prevents integer overflow in long-running applications

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

# Create a distributed vector from uniform data
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# Create a vector from sparse contributions (summed across ranks)
v_sparse = Vec_sum(sparsevec([1, 3], [1.0, 3.0], 4))

# Vector operations
y = v .+ 1.0        # Element-wise addition
z = 2.0 .* v        # Scaling

# Create matrices and solve linear systems
A = Mat_uniform(sparse([1.0 2.0; 3.0 4.0]))
b = Vec_uniform([1.0, 2.0])
x = A \ b

# Resources are automatically released at safe points; vectors are pooled by default
MPI.Finalize()
```

## Architecture

### SafeMPI Module

The core of the package implements a reference-counting garbage collection system:

- **DistributedRefManager**: Coordinates reference counting across MPI ranks
  - Rank 0 allocates object IDs and recycles them; reference counters are mirrored on all ranks
  - Each rank enqueues local releases without MPI; at safe points, ranks collectively `Allgather` release IDs and update counters identically
  - Objects ready for destruction are identified deterministically on all ranks; destruction runs collectively without an extra broadcast
  - Automatic ID recycling prevents unbounded growth (recycled on rank 0)

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

## Documentation

For detailed documentation on the architecture and implementation, see:
- [CLAUDE.md](CLAUDE.md) - Development guide and architecture overview

## Requirements

- Julia ≥ 1.11
- MPI implementation (OpenMPI, MPICH, etc.)
- PETSc ≥ 0.3.1

## Author

Sébastien Loisel (S.Loisel@hw.ac.uk)

## License

MIT License - see [LICENSE](LICENSE) for details.
