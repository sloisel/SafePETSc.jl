# SafePETSc.jl

SafePETSc is a Julia package that makes distributed [PETSc](https://petsc.org/) linear algebra feel like native Julia. Instead of writing verbose PETSc C API calls with explicit pointer management and multi-step operations, you can write natural Julia expressions like `A * B + C`, `A \ b`, or `y .= 2 .* x .+ 3`.

**The Problem**: PETSc is powerful but cumbersome. Multiplying two matrices takes many lines of code, requires careful pointer management, and errors cause segfaults and bus errors. Decomposing algebraic expressions into sequences of low-level operations is error-prone and verbose.

**The Solution**: SafePETSc implements Julia's array interface for distributed PETSc objects. You get arithmetic operators (`+`, `-`, `*`, `\`, `/`), broadcasting (`y .= f.(x)`), standard constructors (`spdiagm`, `vcat`, `hcat`, `blockdiag`), and familiar iteration patterns (`eachrow`). The package handles the complexity of PETSc's C API and manages object lifecycles automatically through distributed reference management.

## Features

- **Julia-Native Syntax**: Use `A * B`, `A \ b`, `A + B`, `A'` just like regular Julia matrices
- **Broadcasting**: Full support for `.+`, `.*`, `.=` and function broadcasting `f.(x)`
- **Standard Constructors**: `spdiagm`, `vcat`, `hcat`, `blockdiag` work on distributed matrices
- **Linear Algebra**: Matrix multiplication, solving (`\`, `/`), transpose, in-place operations
- **Automatic Memory Management**: Objects are cleaned up automatically across MPI ranks
- **Vector Pooling**: Temporary vectors are reused for efficiency (configurable with `ENABLE_VEC_POOL[]`)
- **Iteration**: Use `eachrow(A)` for row-wise processing

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

# Display the solution (only on rank 0)
println(io0(), x)
```

## Core Types

SafePETSc provides three main types for distributed linear algebra:

- **`Vec{T}`**: Distributed vectors with Julia array-like operations (broadcasting, arithmetic, etc.) and automatic pooling for efficiency
- **`Mat{T}`**: Distributed matrices with arithmetic operators (`+`, `-`, `*`, `\`), broadcasting, transpose (`A'`), and GPU-friendly operations
- **`KSP{T}`**: Linear solver objects that can be reused for multiple solves with the same matrix

### Memory Management

Objects are automatically cleaned up when they go out of scope through distributed garbage collection. For faster cleanup in memory-intensive applications, you can manually trigger cleanup:

```julia
GC.gc()                          # Run Julia's garbage collector
SafeMPI.check_and_destroy!()     # Perform distributed cleanup
```

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
