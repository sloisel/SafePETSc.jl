# Distributed Reference Management

SafePETSc's core feature is automatic distributed reference management via the `SafeMPI` module. This ensures that distributed objects are properly cleaned up across all MPI ranks.

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
3. **Cleanup Points**: At `check_and_destroy!` calls (automatically throttled at construction), SafePETSc:
   - Triggers garbage collection so finalizers run
   - Drains each rank’s local release queue
   - Allgathers counts and then Allgathervs the release IDs so every rank sees the same global sequence
   - Each rank updates its mirrored counters identically and computes the same set of ready IDs
   - All ranks destroy ready objects simultaneously

### Architecture

```
All ranks (mirrored)
┌──────────────────────────────────────────────────────────┐
│ 1) Drain local release queue                             │
│ 2) Allgather counts; if total==0, quiescent              │
│ 3) Allgatherv payloads; everyone sees same release IDs   │
│ 4) Update counters; when count == nranks, mark ready     │
│ 5) Destroy ready objects on all ranks                    │
└──────────────────────────────────────────────────────────┘
```

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

## Controlling Cleanup

### Automatic Cleanup and Pooling

Cleanup happens automatically at object creation (throttled by `default_check`). For PETSc vectors, the default behavior is to return released vectors to a reuse pool instead of destroying them. Disable pooling with `ENABLE_VEC_POOL[] = false` or call `clear_vec_pool!()` to free pooled vectors.

```julia
# Every 10th object creation triggers cleanup
v1 = Vec_uniform([1.0, 2.0])  # May trigger cleanup
v2 = Vec_uniform([3.0, 4.0])  # May trigger cleanup
# ...
```

### Explicit Cleanup

You can manually trigger cleanup:

```julia
# Force immediate cleanup
SafeMPI.check_and_destroy!()

# Or with throttling
SafeMPI.check_and_destroy!(max_check_count=10)
```

### Disabling Assertions

For performance in production:

```julia
SafeMPI.set_assert(false)  # Disable @mpiassert checks
```

## Best Practices

### 1. Let Scoping Work for You

```julia
function compute_something()
    A = Mat_uniform(...)
    b = Vec_uniform(...)
    x = A \ b
    # A, b, x cleaned up when function exits
    return extract_result(x)
end
```

### 2. Avoid Premature Manual Cleanup

```julia
# Don't do this:
v = Vec_uniform([1.0, 2.0])
SafeMPI.check_and_destroy!()  # v might still be in use!
y = v .+ 1.0  # May crash

# Instead, let automatic cleanup handle it
v = Vec_uniform([1.0, 2.0])
y = v .+ 1.0
# cleanup happens automatically at safe points
```

### 3. Explicit Cleanup in Long-Running Loops

```julia
for i in 1:1000000
    v = Vec_uniform(data[i])
    result[i] = compute(v)

    # Periodic cleanup to avoid accumulation
    if i % 100 == 0
        SafeMPI.check_and_destroy!()
    end
end
```

## Debugging

### Check Reference Counts

```julia
# Access the default manager
manager = SafeMPI.default_manager[]

# Inspect state (mirrored on all ranks)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Active objects: ", length(manager.counter_pool))
    println("Free IDs: ", length(manager.free_ids))
end
```

### Enable Verbose Assertions

```julia
# Assertions are enabled by default
SafeMPI.enable_assert[]  # true

# Use @mpiassert for collective checks
@mpiassert all_data_valid "Data validation failed"
```

## Performance Considerations

- **Cleanup Cost**: `check_and_destroy!` triggers a full GC and uses collective `Allgather/Allgatherv`
- **Throttling**: The `max_check_count` parameter reduces overhead by skipping some cleanup points
- **ID Recycling**: Released IDs are reused via the shared `free_ids` vector to prevent unbounded growth

## See Also

- [`DRef`](@ref)
- [`DistributedRefManager`](@ref)
- [`check_and_destroy!`](@ref)
