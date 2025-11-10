# PETSc module initialization occurs before MPI can be initialized, causing segfault

## Description

When using PETSc.jl in an MPI environment, the module's `__init__()` function runs during `using PETSc`, which occurs before user code can call `MPI.Init()`. This causes segfaults because PETSc's initialization tries to access MPI functionality before MPI is properly initialized.

This is particularly problematic when PETSc.jl is loaded as a dependency of another package, as users have no control over when `using PETSc` executes.

## Environment
- Julia version: 1.11.7
- PETSc.jl version: 0.3.1
- MPI.jl version: 0.20.23
- OS: macOS (Darwin 25.1.0)

## Minimal Reproduction

**File: `test_petsc_segfault.jl`**
```julia
using MPI

# This is the problematic sequence:
# 1. Load PETSc BEFORE calling MPI.Init()
using PETSc  # <-- PETSc.__init__() runs here, tries to access MPI

# 2. Initialize MPI (too late!)
if !MPI.Initialized()
    MPI.Init()
end

# 3. Initialize PETSc
if !PETSc.initialized(PETSc.petsclibs[1])
    PETSc.initialize()
end

println("If we got here, it worked!")
```

**Run with:**
```bash
mpiexec -n 2 julia test_petsc_segfault.jl
```

**Expected result:** Program runs without error

**Actual result:** Segmentation fault

## Root Cause

The execution sequence is:
1. `mpiexec` launches Julia → MPI environment is set up by mpiexec
2. User code: `using MPI`
3. User code: `using PETSc`
   - **PETSc.jl's `__init__()` runs here**, potentially accessing MPI before it's initialized
4. User code: `MPI.Init()` is called (**too late!**)
5. User code: `PETSc.initialize()` is called

## Current Workaround

The only workaround is to ensure `MPI.Init()` is called **before** `using PETSc`:

**File: `test_petsc_workaround.jl`**
```julia
using MPI

# WORKAROUND: Initialize MPI BEFORE loading PETSc
if !MPI.Initialized()
    MPI.Init()
end

# Now safe to load PETSc
using PETSc

# Initialize PETSc
if !PETSc.initialized(PETSc.petsclibs[1])
    PETSc.initialize()
end

println("Success with workaround!")
```

## Impact

This issue makes it nearly impossible to use PETSc.jl as a dependency in other packages, because:
1. When a user does `using MyPackage` (which depends on PETSc.jl), the `using PETSc` happens during module loading
2. This occurs before the user's test file can call `MPI.Init()`
3. Result: immediate segfault

## Suggested Fix

PETSc.jl's `__init__()` function should avoid accessing MPI functionality, or should check if MPI is initialized first. The actual PETSc initialization should be deferred to an explicit call to `PETSc.initialize()`, which users already call after `MPI.Init()`.

Alternatively, document that PETSc.jl should only be loaded after `MPI.Init()` has been called, though this would require significant restructuring of code that depends on PETSc.jl.
