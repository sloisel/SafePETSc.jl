```@meta
CurrentModule = SafePETSc
```

```@eval
using Markdown
using Pkg
using SafePETSc
v = string(pkgversion(SafePETSc))
md"# SafePETSc.jl $v"
```

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
- **`Mat{T,Prefix}`**: Distributed matrices with arithmetic operators (`+`, `-`, `*`, `\`), broadcasting, transpose (`A'`), and GPU-friendly operations
- **`KSP{T,Prefix}`**: Linear solver objects that can be reused for multiple solves with the same matrix

### The Prefix Type Parameter

Matrices and KSP objects take a `Prefix` type parameter that controls PETSc object configuration. Vectors always use the MPIDENSE prefix internally and do not expose this parameter. SafePETSc provides two built-in prefix types:

- **`MPIAIJ`** (default): For sparse matrices
  - String prefix: `"MPIAIJ_"`
  - Default PETSc matrix type: `mpiaij` (MPI sparse matrix)
  - Use for: Sparse linear algebra, general computations

- **`MPIDENSE`**: For dense matrices
  - String prefix: `"MPIDENSE_"`
  - Default PETSc matrix type: `mpidense` (MPI dense matrix)
  - Default PETSc vector type: `mpi` (standard MPI vector)
  - Use for: Dense linear algebra, operations requiring dense storage (e.g., `eachrow`)

The string prefix is used when setting PETSc options:

```julia
# Configure GPU acceleration for dense matrices
petsc_options_insert_string("-MPIDENSE_mat_type mpidense")

# Create matrix with that prefix
A = Mat_uniform(data; Prefix=MPIDENSE)
```

**Note**: All PETSc vectors are inherently dense (they store all elements). Vectors always use the MPIDENSE prefix internally for PETSc options.

### Memory Management

Objects are automatically cleaned up when they go out of scope through distributed garbage collection. No explicit cleanup is required.

## Installation

```julia
using Pkg
Pkg.add("SafePETSc")
```

## Getting Started

See the [Getting Started](getting_started.md) guide for a tutorial on using SafePETSc.

## Author's remarks

In the early 90s, Bill Gropp and Barry Smith gave us PETSc and made the impossible possible, to solve mathematical problems on large clusters of computers. Although PETSc is a great boon, it is not my favorite thing to debug, especially this particular project, which consists essentially of exposing existing PETSc functionality to the Julia environment -- "glue code". Because of this, I "vibe-coded" this whole Julia-PETSc interface into existence with the help of an AI tool named Claude. As I monitored Claude, I regularly found it making all sorts of mistakes that I fixed as best I could, but I am fairly well convinced that I did not catch all of them. It remains to be seen which is worse: AI mistakes, or human mistakes!

Sincerely,

S. Loisel

## Index

```@index
```
