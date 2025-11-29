# Direct Solvers

SafePETSc uses PETSc's direct solver infrastructure for linear systems. The solver selection is automatic based on your PETSc configuration.

## Default Behavior

When you call `SafePETSc.Init()`, the package automatically configures the direct solver:

- **STRUMPACK** is preferred when detected (offers GPU acceleration and better MPI scalability)
- **MUMPS** is used as the fallback (included in the default PETSc_jll, works out of the box)

You don't need to do anything special - just use the solver:

```julia
using SafePETSc
SafePETSc.Init()

# Direct solve uses STRUMPACK if available, otherwise MUMPS
x = A \ b
```

## Checking Available Solver

You can check if STRUMPACK is available:

```julia
if has_strumpack()
    println("Using STRUMPACK")
else
    println("Using MUMPS (default)")
end
```

STRUMPACK is detected when the `JULIA_PETSC_LIBRARY` environment variable points to a STRUMPACK-enabled PETSc build.

## When to Consider STRUMPACK

STRUMPACK offers advantages for:
- Large sparse systems where MPI scalability matters
- Problems that benefit from GPU acceleration (NVIDIA CUDA)
- Applications using low-rank compression (BLR, HSS) to reduce memory

For most users, the default MUMPS solver works well with zero configuration.

## STRUMPACK Options

When STRUMPACK is available, you can configure it via PETSc options:

```julia
# Enable BLR compression
petsc_options_insert_string("-mat_strumpack_compression BLR")

# Enable GPU acceleration (if built with CUDA support)
petsc_options_insert_string("-mat_strumpack_gpu")

# Verbose output
petsc_options_insert_string("-mat_strumpack_verbose")
```

## See Also

- [Linear Solvers](solvers.md) - General solver documentation
- [PETSc STRUMPACK Documentation](https://petsc.org/release/manualpages/Mat/MATSOLVERSTRUMPACK/)
