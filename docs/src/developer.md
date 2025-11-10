# Developer Guide

This guide is for developers who want to contribute to SafePETSc or extend it with custom distributed types.

## Architecture Overview

SafePETSc consists of two main layers:

1. **SafeMPI**: Low-level distributed reference management
2. **SafePETSc**: High-level PETSc wrappers using SafeMPI

### SafeMPI Layer

The `SafeMPI` module implements reference counting across MPI ranks:

```
┌─────────────────────────────────────────────┐
│  User Code                                  │
│  creates DRef-wrapped objects               │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  SafeMPI.DRef{T}                            │
│  - Wraps object                             │
│  - Finalizer calls _release!                │
│  - Enqueues release message                 │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  DistributedRefManager (Rank 0)             │
│  - Receives release messages                │
│  - Tracks reference counts                  │
│  - Broadcasts destroy commands              │
└─────────────┬───────────────────────────────┘
              │
┌─────────────▼───────────────────────────────┐
│  destroy_obj!(obj)                          │
│  - Called on all ranks simultaneously       │
│  - User-defined cleanup routine             │
└─────────────────────────────────────────────┘
```

### SafePETSc Layer

Wraps PETSc objects with `DRef`:

```julia
struct _Vec{T}
    v::PETSc.Vec{T}
    row_partition::Vector{Int}
    prefix::String
end

const Vec{T} = DRef{_Vec{T}}

# Opt-in to distributed management
SafeMPI.destroy_trait(::Type{_Vec{T}}) = SafeMPI.CanDestroy()

# Define cleanup
function SafeMPI.destroy_obj!(x::_Vec{T})
    _destroy_petsc_vec!(x.v)
end
```

## Adding New Distributed Types

To add your own distributed type:

### 1. Define the Internal Type

```julia
struct _MyDistributedType
    # Your fields here
    handle::Ptr{Cvoid}  # e.g., MPI handle
    data::Vector{Float64}
    # ... other fields
end
```

### 2. Create Type Alias

```julia
const MyDistributedType = SafeMPI.DRef{_MyDistributedType}
```

### 3. Opt-In to Management

```julia
SafeMPI.destroy_trait(::Type{_MyDistributedType}) = SafeMPI.CanDestroy()
```

### 4. Implement Cleanup

```julia
function SafeMPI.destroy_obj!(obj::_MyDistributedType)
    # IMPORTANT: This is called on ALL ranks simultaneously
    # Must be a collective operation

    # Example: Free MPI resource
    if obj.handle != C_NULL
        MPI.Free(obj.handle)
    end

    # Clean up other resources
    # ...
end
```

### 5. Create Constructor

```julia
function MyDistributedType(data::Vector{Float64})
    # Allocate distributed resource
    handle = allocate_mpi_resource(data)

    # Wrap in internal type
    obj = _MyDistributedType(handle, data)

    # Wrap in DRef (triggers cleanup coordination)
    return DRef(obj)
end
```

## Testing

### Unit Tests Structure

SafePETSc uses a dual-file testing approach:

- `test/runtests.jl`: Entry point that spawns MPI processes
- `test/test_*.jl`: Individual test files run with MPI

Example test file:

```julia
# test/test_myfeature.jl
using SafePETSc
using Test
using MPI

SafePETSc.Init()

@testset "My Feature" begin
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Test uniform distribution
    v = Vec_uniform([1.0, 2.0, 3.0])
    @test size(v) == (3,)

    # Test operations
    y = v .+ 1.0
    @test eltype(y) == Float64

    # Cleanup
    SafeMPI.check_and_destroy!()
end
```

### Running Tests

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_myfeature.jl`)'
```

## Coding Guidelines

### Reference Management

1. **Always use `DRef`**: Wrap distributed objects in `DRef` to ensure cleanup
2. **Cleanup at creation**: `_make_ref` automatically calls `check_and_destroy!`
3. **No manual cleanup in operations**: Avoid `check_and_destroy!` in regular functions
4. **Collective operations**: `destroy_obj!` must be collective

### Error Handling

1. **Use `@mpiassert`**: For collective error checking
2. **Coalesce assertions**: Combine conditions into single `@mpiassert`
3. **Informative messages**: Include context in error messages

```julia
# Good: single assertion with multiple conditions
@mpiassert (size(A, 2) == size(B, 1) &&
            A.obj.col_partition == B.obj.row_partition) "Matrix dimensions and partitions must match for multiplication"

# Less good: multiple assertions
@mpiassert size(A, 2) == size(B, 1) "Dimension mismatch"
@mpiassert A.obj.col_partition == B.obj.row_partition "Partition mismatch"
```

### PETSc Interop

1. **Use PETSc.@for_libpetsc**: For multi-precision support
2. **GPU-friendly operations**: Prefer bulk operations over element access
3. **Const for MAT_INITIAL_MATRIX**: Use module constant `MAT_INITIAL_MATRIX = Cint(0)`

```julia
PETSc.@for_libpetsc begin
    function my_petsc_operation(mat::PETSc.Mat{$PetscScalar})
        PETSc.@chk ccall((:PetscFunction, $libpetsc), ...)
    end
end
```

## Performance Considerations

### Cleanup Overhead

- `check_and_destroy!` triggers full GC and MPI synchronization
- Default: cleanup every 10 object creations (`max_check_count=10` in `_make_ref`)
- Tune based on application: more frequent cleanup = less memory, more overhead

### Memory Management

- Use `DRef` scoping to control lifetimes
- Avoid global `DRef` variables (prevent cleanup)
- Consider explicit cleanup in long loops

### GPU Support

- SafePETSc prioritizes GPU-friendly PETSc operations
- Set PETSc options for GPU: `-mat_type aijcusparse -vec_type cuda`
- Avoid element-wise access (causes GPU↔CPU transfers)

## Documentation

### Docstrings

Follow Julia documentation conventions:

```julia
"""
    my_function(x::Type; option=default) -> ReturnType

Brief one-line description.

Extended description with more details about the function's behavior,
parameters, and return values.

# Arguments
- `x::Type`: Description of x
- `option::Type=default`: Description of optional parameter

# Returns
- `ReturnType`: Description of return value

# Examples
```julia
result = my_function(input)
```

See also: [`related_function`](@ref), [`another_function`](@ref)
"""
function my_function(x; option=default)
    # Implementation
end
```

### Adding Documentation Pages

1. Create markdown file in `docs/src/`
2. Add to `pages` in `docs/make.jl`
3. Build: `julia --project=docs docs/make.jl`

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Run tests: `julia --project=. -e 'using Pkg; Pkg.test()'`
6. Submit pull request

### Code Review Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Docstrings added for public API
- [ ] Reference management correct
- [ ] Collective operations properly coordinated
- [ ] Performance considerations addressed

## Debugging Tips

### MPI Hangs

If program hangs, likely causes:

1. **Non-collective operation**: One rank skipped a collective call
2. **Unbalanced branching**: Ranks took different code paths
3. **Missing `@mpiassert`**: Error on one rank, others waiting

Debug with:
```julia
# Add at suspicious points
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Reached checkpoint A")
end
MPI.Barrier(MPI.COMM_WORLD)
```

### Memory Leaks

Check for:

1. Global `DRef` variables
2. Skipped `check_and_destroy!` in long loops
3. Circular references preventing GC

Inspect manager state:
```julia
manager = SafeMPI.default_manager[]
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Active objects: ", length(manager.counter_pool))
    println("Pending releases: ", length(manager.pending_releases))
end
```

### Assertion Failures

Enable verbose output:
```julia
# Assertions enabled by default
SafeMPI.enable_assert[]  # true

# Check conditions
@mpiassert condition "Detailed error message"
```

## Resources

- [PETSc Documentation](https://petsc.org/release/documentation/)
- [MPI.jl Documentation](https://juliaparallel.org/MPI.jl/)
- [Documenter.jl Guide](https://documenter.juliadocs.org/)
