# SafeMPI API Reference

The `SafeMPI` module provides distributed reference management for MPI-based parallel computing.

## Core Types

```@docs
SafePETSc.SafeMPI.DRef
SafePETSc.SafeMPI.DistributedRefManager
```

## Reference Management

```@docs
SafePETSc.SafeMPI.check_and_destroy!
SafePETSc.SafeMPI.destroy_obj!
SafePETSc.SafeMPI.default_manager
SafePETSc.SafeMPI.default_check
```

## Trait System

```@docs
SafePETSc.SafeMPI.DestroySupport
SafePETSc.SafeMPI.CanDestroy
SafePETSc.SafeMPI.CannotDestroy
SafePETSc.SafeMPI.destroy_trait
```

## MPI Utilities

```@docs
SafePETSc.SafeMPI.mpi_any
SafePETSc.SafeMPI.mpi_uniform
SafePETSc.SafeMPI.mpierror
SafePETSc.SafeMPI.@mpiassert
```

## Configuration

```@docs
SafePETSc.SafeMPI.enable_assert
SafePETSc.SafeMPI.set_assert
```
