# SafeMPI API Reference

The `SafeMPI` module provides distributed reference management for MPI-based parallel computing.

## Core Types

```@docs
SafeMPI.DRef
SafeMPI.DistributedRefManager
```

## Reference Management

```@docs
SafeMPI.check_and_destroy!
SafeMPI.destroy_obj!
SafeMPI.default_manager
```

## Trait System

```@docs
SafeMPI.DestroySupport
SafeMPI.CanDestroy
SafeMPI.CannotDestroy
SafeMPI.destroy_trait
```

## MPI Utilities

```@docs
SafeMPI.mpi_any
SafeMPI.mpierror
SafeMPI.@mpiassert
```

## Configuration

```@docs
SafeMPI.enable_assert
SafeMPI.set_assert
```
