# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafePETSc is a Julia package that provides distributed reference management for MPI-based parallel computing. The core purpose is to safely manage the lifecycle of distributed objects across MPI ranks, ensuring objects are destroyed only when all ranks have released their references.

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

## Architecture

### Core Components

**SafeMPI Module** (`src/SafeMPI.jl`):
The heart of the package, implementing a reference-counting garbage collection system for MPI-distributed objects.

- **DistributedRefManager**: Manages reference counting across MPI ranks
  - Rank 0 acts as the coordinator, tracking reference counts via `counter_pool`
  - Other ranks send release messages to rank 0 using MPI tag `RELEASE_TAG = 1001`
  - Uses `free_ids` for ID recycling to prevent unbounded growth

- **DRef{T}**: A wrapper around objects that need coordinated destruction
  - Only works with types that opt-in via the trait system
  - Tracks whether the local reference has been released

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

1. **Object Creation**: Rank 0 allocates a unique ID, broadcasts to all ranks
2. **Automatic Release**: When a DistributedRef is garbage collected, its finalizer automatically calls `release!()`, which either increments the local counter (rank 0) or sends a message to rank 0
3. **Destruction Check**: `check_and_destroy!()` polls for release messages, identifies objects ready for destruction (count == nranks), broadcasts IDs, and all ranks destroy their local copies simultaneously

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
