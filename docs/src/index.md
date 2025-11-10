# SafePETSc.jl

SafePETSc is a Julia package that provides distributed reference management for MPI-based parallel computing with PETSc. The core purpose is to safely manage the lifecycle of distributed objects across MPI ranks, ensuring objects are destroyed only when all ranks have released their references.

## Features

- **Automatic Memory Management**: Distributed objects are automatically tracked and released when all MPI ranks have released their references
- **GPU-Friendly Operations**: Prioritizes PETSc's native GPU-compatible operations
- **Type-Safe API**: Uses Julia's trait system to ensure only appropriate types are managed
- **Efficient Cleanup**: Centralized cleanup with configurable throttling to reduce overhead
- **Rich Linear Algebra**: Comprehensive support for distributed vectors, matrices, and solvers

## Quick Example

```julia
using SafePETSc
using MPI

# Initialize MPI and PETSc
SafePETSc.Init()

# Create a distributed matrix
A = Mat_uniform([2.0 1.0; 1.0 3.0])

# Create a distributed vector
b = Vec_uniform([1.0, 2.0])

# Solve the linear system
x = A \ b

# Objects are automatically cleaned up when they go out of scope
```

## Package Architecture

SafePETSc consists of two main modules:

### SafeMPI

The `SafeMPI` module implements the distributed reference management system:

- **`DRef{T}`**: A distributed reference wrapper that tracks objects across MPI ranks
- **`DistributedRefManager`**: Allocates IDs on rank 0, mirrors reference counters on all ranks, and performs collective cleanup via Allgather/Allgatherv
- **Trait-based destruction**: Types must opt-in to distributed management
- **Automatic cleanup**: Finalizers enqueue releases locally; `check_and_destroy!` performs GC and collective Allgather at safe points

### SafePETSc

The main module wraps PETSc functionality with safe distributed reference management:

- **`Vec{T}`**: Distributed vectors with automatic memory management and pooling (vectors are returned to a reuse pool by default)
- **`Mat{T}`**: Distributed matrices with GPU-friendly operations
- **`Solver{T}`**: Linear solver objects that can be reused

## Installation

```julia
using Pkg
Pkg.add("SafePETSc")
```

## Getting Started

See the [Getting Started](getting_started.md) guide for a tutorial on using SafePETSc.

## Index

```@index
```
