# Distributed Reference Management

SafePETSc enables native Julia syntax for distributed linear algebra by implementing automatic distributed reference management via the `SafeMPI` module. This implementation detail ensures that distributed objects are properly cleaned up across all MPI ranks, allowing users to write natural Julia expressions like `A * B + C` without manual memory management.

## The Problem

In MPI-based parallel computing, objects like PETSc vectors and matrices exist on all ranks. Destroying such objects requires collective MPI calls—all ranks must participate. This creates challenges:

1. **Premature destruction**: If one rank destroys an object while others still need it, the program crashes
2. **Memory leaks**: If ranks don't coordinate cleanup, objects leak memory
3. **Complex coordination**: Manual reference counting is error-prone

## The Solution: `DRef`

SafePETSc uses `DRef{T}` (Distributed Reference) to automatically track object lifetimes:

```julia
using SafePETSc

# Create a distributed vector (returns a DRef{_Vec{Float64}})
v = Vec_uniform([1.0, 2.0, 3.0])

# Use it normally
y = v .+ 1.0

# When v goes out of scope and is garbage collected,
# SafePETSc coordinates cleanup across all ranks
```

## How It Works

### Reference Counting

1. **Mirrored Counters**: Each rank runs the same deterministic ID allocation, keeping a mirrored `counter_pool` and shared `free_ids` stack; all ranks recycle IDs identically without a designated root
2. **Automatic Release**: When a `DRef` is garbage collected, its finalizer enqueues the ID locally (no MPI in finalizers)
3. **Cleanup Points**: At `check_and_destroy!` calls (automatically invoked at object creation), SafePETSc:
   - Periodically triggers partial garbage collection (`GC.gc(false)`) so finalizers run
   - Drains each rank's local release queue
   - Allgathers counts and then Allgathervs the release IDs so every rank sees the same global sequence
   - Each rank updates its mirrored counters identically and computes the same set of ready IDs
   - All ranks destroy ready objects simultaneously

## Trait-Based Opt-In

Types must explicitly opt-in to distributed management:

```julia
# Define your distributed type
struct MyDistributedObject
    data::Vector{Float64}
    # ... MPI-based fields
end

# Opt-in to distributed management
SafeMPI.destroy_trait(::Type{MyDistributedObject}) = SafeMPI.CanDestroy()

# Implement cleanup
function SafeMPI.destroy_obj!(obj::MyDistributedObject)
    # Perform collective cleanup (e.g., MPI_Free, PETSc destroy)
    # This is called on ALL ranks simultaneously
    cleanup_mpi_resources(obj)
end

# Now you can wrap it
ref = DRef(MyDistributedObject(...))
```

## Automatic Cleanup

Cleanup is handled automatically by SafePETSc. At every object creation, the library internally calls `check_and_destroy!` which:
- Periodically triggers partial garbage collection (`GC.gc(false)`) to run finalizers
- Processes pending releases via MPI communication

This means users don't need to call `check_and_destroy!()` explicitly in normal code. The throttle frequency is controlled by `SafePETSc.default_check[]` (default: 10).

For PETSc vectors, the default behavior is to return released vectors to a reuse pool instead of destroying them. Disable pooling with `ENABLE_VEC_POOL[] = false` or call `clear_vec_pool!()` to free pooled vectors.

### Disabling Assertions

For performance in production:

```julia
SafeMPI.set_assert(false)  # Disable @mpiassert checks
```

## Best Practices

### Let Scoping Work for You

```julia
function compute_something()
    A = Mat_uniform(...)
    b = Vec_uniform(...)
    x = A \ b
    # A, b, x cleaned up when function exits
    return extract_result(x)
end
```

In long-running loops, cleanup happens automatically when new objects are created, so no explicit calls are needed.

## Debugging

### Check Reference Counts

```julia
# Access the default manager
manager = SafeMPI.default_manager[]

# Inspect state (mirrored on all ranks)
println(io0(), "Active objects: ", length(manager.counter_pool))
println(io0(), "Free IDs: ", length(manager.free_ids))
```

### Enable Verbose Assertions

```julia
# Assertions are enabled by default
SafeMPI.enable_assert[]  # true

# Use @mpiassert for collective checks
@mpiassert all_data_valid "Data validation failed"
```

## Performance Considerations

- **Cleanup Cost**: The internal cleanup mechanism uses collective `Allgather/Allgatherv` operations and periodically triggers partial garbage collection
- **Throttling**: Adjust `SafePETSc.default_check[]` to control how often partial garbage collection is triggered (default: 10). Higher values reduce GC overhead but may delay object finalization

## See Also

- [`SafePETSc.SafeMPI.DRef`](@ref)
- [`SafePETSc.SafeMPI.DistributedRefManager`](@ref)
