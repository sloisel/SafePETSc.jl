# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafePETSc is a Julia package that makes distributed PETSc linear algebra feel like native Julia. Instead of writing verbose PETSc C API calls with explicit pointer management and multi-step operations, users can write natural Julia expressions like `A * B + C`, `A \ b`, or `y .= 2 .* x .+ 3`.

**The Problem**: PETSc is powerful but cumbersome. Multiplying two matrices takes many lines of code, requires careful pointer management, and errors cause segfaults and bus errors. Decomposing algebraic expressions into sequences of low-level operations is error-prone and verbose.

**The Solution**: SafePETSc implements Julia's array interface for distributed PETSc objects. You get arithmetic operators (`+`, `-`, `*`, `\`, `/`), broadcasting (`y .= f.(x)`), standard constructors (`spdiagm`, `vcat`, `hcat`, `blockdiag`), and familiar iteration patterns (`eachrow`). The package handles the complexity of PETSc's C API and manages object lifecycles automatically through distributed reference management.

## Development Commands

### Package Management
```bash
# Activate the package environment
julia --project=.

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Start Julia REPL with package loaded
julia --project=. -e 'using SafePETSc'
```

### Testing with MPI
The package uses a dual-file testing approach for MPI:
- `test/runtests.jl` - Entry point that spawns MPI processes
- `test/test_mpi.jl` - Actual MPI tests run across multiple ranks

```bash
# Run tests (automatically spawns 4 MPI ranks)
julia --project=. -e 'using Pkg; Pkg.test()'

# Or run MPI tests directly
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(Base.active_project()) test/test_mat_uniform.jl`)'
```

This pattern allows `Pkg.test()` to work correctly by having `runtests.jl` call `mpiexec` to spawn the test workers.

### Git and github.

- Do not automatically commit changes into git unless asked by the user.
- Do not automatically push commits to github unless asked by the user.
- Do not tag releases, TagBot will tag on github.

## User-Facing Features

SafePETSc implements a Julia-native interface for PETSc, allowing natural mathematical expressions.

### Vector Operations

```julia
# Arithmetic and broadcasting (feels like regular Julia)
y = 2 .* x .+ 3           # Broadcasting with scalars
z = x .+ y                # Element-wise addition
w = x .* y                # Element-wise multiplication
y .= f.(x)                # Apply function to each element

# Standard operations
result = x + y            # Vector addition
result = x - y            # Vector subtraction
result = -x               # Negation
result = x' * y           # Dot product (via adjoint)
result = x' * A           # Adjoint-matrix multiply
```

### Matrix Operations

```julia
# Linear algebra (just like dense Julia matrices)
C = A * B                 # Matrix-matrix multiplication
y = A * x                 # Matrix-vector multiplication
x = A \ b                 # Linear solve (single RHS)
X = A \ B                 # Linear solve (multiple RHS)
x = A' \ b                # Solve with transpose
y = b' / A                # Left division (row vector)
Y = B / A                 # Right division (matrix)

# Arithmetic
C = A + B                 # Matrix addition
C = A - B                 # Matrix subtraction
B = A'                    # Transpose (adjoint)
C = α * A                 # Scalar multiplication
C = α + A                 # Add scalar to diagonal

# Construction
A = spdiagm(0 => diag, 1 => upper, -1 => lower)  # Diagonal/tridiagonal
C = vcat(A, B)           # Vertical concatenation
C = hcat(A, B)           # Horizontal concatenation
C = blockdiag(A, B)      # Block diagonal

# Iteration
for row in eachrow(A)    # Iterate over rows (dense matrices)
    # ... process row ...
end
```

### In-Place Operations

```julia
# Efficient in-place versions (no temporaries)
mul!(y, A, x)            # y = A * x
mul!(C, A, B)            # C = A * B
ldiv!(x, A, b)           # x = A \ b (solve in-place)
transpose!(B, A)         # B = A' (transpose in-place)
```

### Construction Patterns

```julia
# Uniform matrices/vectors (same data on all ranks)
A = Mat_uniform(petsc_mat)
x = Vec_uniform(petsc_vec)

# Sum matrices/vectors (data split across ranks)
A = Mat_sum(petsc_mat)
x = Vec_sum(petsc_vec)

# Helpers
y = zeros_like(x)        # Create zero vector with same structure
y = ones_like(x)         # Create ones vector
y = fill_like(x, val)    # Fill with value
```

### Key Differences from Native Julia

- **No scalar indexing**: Use PETSc's bulk operations instead of `A[i,j]` (prevents GPU→CPU transfers)
- **Explicit partitioning**: Use `Vec_uniform` vs `Vec_sum` to control data distribution
- **Solver reuse**: Create `Solver(A)` objects for repeated solves with same matrix
- **Memory pooling**: Control temporary vector allocation with vector pools

## Architecture (Implementation Details)

*This section describes the internal implementation. Users don't need to understand these details to use SafePETSc effectively.*

### Core Components

**SafeMPI Module** (`src/SafeMPI.jl`):
The heart of the package, implementing a reference-counting garbage collection system for MPI-distributed objects.

- **DistributedRefManager**: Manages reference counting across MPI ranks
  - All ranks allocate IDs deterministically and recycle them via a shared `free_ids` vector; counters remain mirrored on all ranks
  - Finalizers enqueue local releases without MPI
  - At safe points (`check_and_destroy!`), ranks Allgather counts and then Allgatherv release IDs, update mirrored counters identically, and destroy ready objects collectively
  - Uses `free_ids` for ID recycling to prevent unbounded growth (mirrored on all ranks)

- **DRef{T}**: A wrapper around objects that need coordinated destruction
  - Only works with types that opt-in via the trait system
  - Local releases are gathered collectively at safe points

### Trait-Based Destruction System

The package uses a trait-based approach to control which types can be managed:

1. **Opt-in Model**: Types must explicitly declare support for distributed management
   - Default: `destroy_trait(::Type) = CannotDestroy()`
   - To enable: Define `destroy_trait(::Type{YourType}) = CanDestroy()`

2. **Destruction Hook**: Types must implement their cleanup logic
   - Define: `destroy_obj!(obj::YourType) = ...`
   - This is called when all ranks have released the object

3. **Error Safety**: Attempting to create a DRef for an unsupported type produces a clear error message

### MPI Communication Flow

1. **Object Creation**: Every rank runs the same deterministic ID allocation, inserts the new ID into its mirrored `counter_pool`, and initializes `free_ids` state identically
2. **Automatic Release**: Finalizers enqueue local release IDs (no MPI calls in finalizers)
3. **Destruction Check**: `check_and_destroy!()` performs a full GC, drains local queues, Allgathers counts and payload to all ranks, each rank updates counters, computes ready IDs, and all ranks destroy their local copies simultaneously (no additional broadcast needed)

### Key Design Decisions

- **Automatic Cleanup**: Uses finalizers to automatically call `release!()` when DistributedRefs are garbage collected
- **Explicit `release!()` Optional**: Users can call `release!()` manually for immediate cleanup, but it's not required
- **Synchronous Destruction**: `check_and_destroy!()` must be called explicitly to perform the actual cleanup, allowing controlled cleanup points
- **ID Recycling**: Released IDs are reused to prevent integer overflow in long-running applications
- **Trait System**: Prevents accidental misuse by requiring explicit opt-in for destruction support
- **No advertisement**: Putting advertisements for Claude in commit messages or anywhere else is forbidden. Claude is not a co-author. Sebastien Loisel is the one and only author.

## SafePETSc Module

The main module (`src/SafePETSc.jl`) wraps PETSc functionality with safe distributed reference management, using SafeMPI as the foundation.

### PETSc Integration Constants

**MAT_INITIAL_MATRIX** (`src/SafePETSc.jl`):
- Module-wide constant defined as `const MAT_INITIAL_MATRIX = Cint(0)`
- Used in PETSc C API calls like `MatConvert`, `MatTranspose`, `MatMatMult`
- Indicates that a new matrix should be created (not reusing existing storage)
- **Critical**: Must be `Cint(0)`, not `-1` or other values (causes segfaults)
- Centralized to ensure consistency across all PETSc matrix operations

### GPU-Friendly Matrix Operations

The package prioritizes GPU-compatible matrix operations by using PETSc's native functions:

- **MatConvert**: Converts matrix types (e.g., sparse to dense) while preserving GPU storage
- **MatTranspose**: Transposes matrices efficiently on GPU
- **MatMatMult**: Matrix-matrix multiplication with GPU support

**Avoid** element-by-element extraction patterns (e.g., nested loops with `MatGetValues`) as they cause excessive GPU→CPU transfers. Use PETSc's bulk operations instead.
