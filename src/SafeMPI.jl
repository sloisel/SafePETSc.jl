module SafeMPI

using MPI
using SHA
using Serialization

export DistributedRefManager, DRef, check_and_destroy!, destroy_obj!, default_manager, enable_assert, set_assert
export DestroySupport, CanDestroy, CannotDestroy, destroy_trait, mpi_any, mpierror, @mpiassert

const RELEASE_TAG = 1001
const RELEASE_DATA_TAG = 1002

"""
    DestroySupport

Abstract type for the trait system controlling which types can be managed by DRef.
See [`CanDestroy`](@ref) and [`CannotDestroy`](@ref).
"""
abstract type DestroySupport end

"""
    CanDestroy <: DestroySupport

Trait indicating that a type can be managed by DRef and supports collective destruction.
Types must opt-in by defining `destroy_trait(::Type{YourType}) = CanDestroy()`.
"""
struct CanDestroy <: DestroySupport end

"""
    CannotDestroy <: DestroySupport

Trait indicating that a type cannot be managed by DRef (default for all types).
"""
struct CannotDestroy <: DestroySupport end

"""
    destroy_trait(::Type) -> DestroySupport

Trait function determining whether a type can be managed by DRef.

Returns `CanDestroy()` for types that opt-in to distributed reference management,
or `CannotDestroy()` for types that don't support it (default).

# Example
```julia
# Opt-in a custom type
SafeMPI.destroy_trait(::Type{MyType}) = SafeMPI.CanDestroy()
```
"""
destroy_trait(::Type) = CannotDestroy()

"""
    DistributedRefManager

Manages reference counting and collective destruction of distributed objects across MPI ranks.

Rank 0 acts as the coordinator, tracking reference counts in `counter_pool`. Other ranks
send release messages to rank 0 using MPI tag `RELEASE_TAG = 1001`. Released IDs are
recycled via `free_ids` to prevent unbounded growth in long-running applications.

# Fields
- `destroy_early::Bool`: If `false` (default), objects are destroyed when all ranks release them.
  If `true`, objects are destroyed as soon as any rank releases. **Warning**: Setting this to
  `true` is safe for pure SPMD code, but unsafe when rank-dependent flow control leads to
  different local storage of PETSc data structures across ranks.

See also: [`DRef`](@ref), [`check_and_destroy!`](@ref), [`default_manager`](@ref)
"""
mutable struct DistributedRefManager
    counter_pool::Dict{Int, Int}
    objs::Dict{Int, Any}  # Store the actual objects, not the refs
    free_ids::Set{Int}
    next_id::Int
    check_count::Int
    pending_releases_lock::Base.Threads.SpinLock
    pending_releases::Vector{Int}
    destroy_early::Bool  # If true, destroy when counter_pool[id] >= 1; if false, when == nranks
    destroyed::Dict{Int, Bool}  # Track which objects have been destroyed to prevent duplicate releases

    function DistributedRefManager()
        new(Dict{Int, Int}(), Dict{Int, Any}(), Set{Int}(), 1, 0, Base.Threads.SpinLock(), Int[], false, Dict{Int, Bool}())
    end
end

"""
    default_manager

The default `DistributedRefManager` instance used by all `DRef` objects unless
explicitly overridden. Automatically initialized when the module loads.
"""
const default_manager = Ref{DistributedRefManager}()

"""
    enable_assert

Global flag controlling whether `@mpiassert` macros perform their checks.
Set to `false` to disable all MPI assertions for performance. Default is `true`.

See also: [`set_assert`](@ref), [`@mpiassert`](@ref)
"""
const enable_assert = Ref{Bool}(true)

"""
    set_assert(x::Bool) -> nothing

Enable (`true`) or disable (`false`) MPI assertion checks via `@mpiassert`.

# Example
```julia
SafeMPI.set_assert(false)  # Disable assertions
SafeMPI.set_assert(true)   # Re-enable assertions
```
"""
set_assert(x::Bool) = (enable_assert[] = x; nothing)

function __init__()
    default_manager[] = DistributedRefManager()
end

"""
    DRef{T}

A distributed reference to an object of type `T` that is managed across MPI ranks.

When all ranks have released their references (via garbage collection or explicit `release!`),
the object is collectively destroyed on all ranks using the type's `destroy_obj!` method.

# Constructor
    DRef(obj::T; manager=default_manager[]) -> DRef{T}

Create a distributed reference to `obj`. The type `T` must opt-in to distributed management
by defining `destroy_trait(::Type{T}) = CanDestroy()` and implementing `destroy_obj!(obj::T)`.

Finalizers automatically call `release!()` when the `DRef` is garbage collected, so manual
cleanup is optional. Call `check_and_destroy!()` to perform the actual collective destruction.

# Example
```julia
# Define a type that can be managed
struct MyDistributedObject
    data::Vector{Float64}
end

SafeMPI.destroy_trait(::Type{MyDistributedObject}) = SafeMPI.CanDestroy()
SafeMPI.destroy_obj!(obj::MyDistributedObject) = println("Destroying object")

# Create a distributed reference
ref = DRef(MyDistributedObject([1.0, 2.0, 3.0]))
# ref.obj accesses the underlying object
# When ref is garbage collected and check_and_destroy!() is called, the object is destroyed
```

See also: [`DistributedRefManager`](@ref), [`check_and_destroy!`](@ref), [`destroy_trait`](@ref)
"""
mutable struct DRef{T}
    obj::T
    manager::DistributedRefManager
    counter_id::Int
end

# Outer constructor with type assertion
function DRef(obj::T; manager=default_manager[]) where T
    destroy_trait(T)::CanDestroy
    _make_ref(obj, manager)
end

function _make_ref(obj::T, manager) where T
    # Check and destroy at constructor entry for semiregular collection
    # This consolidates cleanup that was previously scattered across constructors
    check_and_destroy!(manager; max_check_count=10)

    counter_id = allocate_id!(manager)
    ref = DRef{T}(obj, manager, counter_id)
    manager.objs[counter_id] = obj  # Store the object itself (strong reference)
    manager.destroyed[counter_id] = false  # Initialize destroyed tracking
    # Add finalizer to automatically release when GC'd
    finalizer(_release!, ref)
    return ref
end

"""
    destroy_obj!(obj)

Trait method called to collectively destroy an object when all ranks have released their
references. Types that opt-in to distributed reference management must implement this method.

# Example
```julia
SafeMPI.destroy_obj!(obj::MyType) = begin
    # Perform collective cleanup (e.g., free MPI/PETSc resources)
    cleanup_resources(obj)
end
```

See also: [`DRef`](@ref), [`destroy_trait`](@ref)
"""
function destroy_obj!(obj)
    error("No destroy_obj! method defined for type $(typeof(obj))")
end

# Enqueue a release ID (called from finalizers, must be fast and nonblocking)
function _enqueue_release!(manager::DistributedRefManager, counter_id::Int)
    lock(manager.pending_releases_lock)
    try
        push!(manager.pending_releases, counter_id)
    finally
        unlock(manager.pending_releases_lock)
    end
    return nothing
end

# Drain pending releases into a local vector (atomic swap and clear)
function _drain_pending_releases!(manager::DistributedRefManager)
    lock(manager.pending_releases_lock)
    ids = copy(manager.pending_releases)
    empty!(manager.pending_releases)
    unlock(manager.pending_releases_lock)
    return ids
end

function allocate_id!(manager::DistributedRefManager)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    if rank == 0
        if !isempty(manager.free_ids)
            counter_id = pop!(manager.free_ids)
        else
            counter_id = manager.next_id
            manager.next_id += 1
        end
        manager.counter_pool[counter_id] = 0
    else
        counter_id = 0
    end
    
    return MPI.bcast(counter_id, 0, MPI.COMM_WORLD)
end

function _release!(ref::DRef)
    # Check if MPI is still initialized (guards against calls after MPI_Finalize)
    if !MPI.Initialized() || MPI.Finalized()
        return
    end

    # Check if object was already destroyed (prevents duplicate release messages)
    if get(ref.manager.destroyed, ref.counter_id, false)
        delete!(ref.manager.destroyed, ref.counter_id)
        return
    end

    # Simply enqueue for processing at the next safe point (check_and_destroy!)
    # NO MPI CALLS HERE - this is safe to call from GC finalizers
    _enqueue_release!(ref.manager, ref.counter_id)
    
    nothing
end

# Flush all pending releases to achieve global quiescence
# Uses blocking Send/Recv (4A) and Bcast (6A) at a known safe point
function _flush_releases!(manager::DistributedRefManager, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    
    # Loop until global quiescence (no pending releases anywhere)
    #while true
        # 1) Drain local queue
        local_ids = _drain_pending_releases!(manager)
        local_n = length(local_ids)
        
        # 2) Global check for quiescence
        total_n = MPI.Allreduce(local_n, +, comm)
        if total_n == 0
            return nothing  # quiescent, done
        end
        
        # 3) Gather counts on root
        if rank == 0
            counts = MPI.Gather(local_n, 0, comm)
        else
            MPI.Gather(local_n, 0, comm)  # participate in collective but don't use result
        end
        
        # 4) Move payloads to root (collective Gatherv)
        all_ids = Int[]
        if rank == 0
            # Build displacements from gathered counts
            counts32 = Int32.(counts)
            displs32 = Vector{Int32}(undef, nranks)
            displs32[1] = 0
            @inbounds for i in 2:nranks
                displs32[i] = displs32[i-1] + counts32[i-1]
            end
            total = Int(sum(counts32))
            recvbuf = Vector{Int}(undef, total)
            MPI.Gatherv!(local_ids, MPI.VBuffer(recvbuf, counts32, displs32), 0, comm)
            all_ids = recvbuf

            # 5) Process releases on root: update refcounts and build destroy commands
            ready_ids = Int[]
            threshold = manager.destroy_early ? 1 : nranks
            for counter_id in all_ids
                if haskey(manager.counter_pool, counter_id)
                    manager.counter_pool[counter_id] += 1
                    if manager.counter_pool[counter_id] >= threshold
                        push!(ready_ids, counter_id)
                    end
                end
                # Silently ignore stale/duplicate releases for IDs not in pool
            end
            
            # 6A) Broadcast decisions: first length, then payload
            num_ready = length(ready_ids)
            num_ready = MPI.bcast(num_ready, 0, comm)
            if num_ready > 0
                MPI.Bcast!(ready_ids, 0, comm)
            end
        else
            # Non-root: contribute payload (possibly zero-length) to Gatherv
            MPI.Gatherv!(local_ids, nothing, 0, comm)
            
            # 6A) Receive decisions
            num_ready = MPI.bcast(0, 0, comm)  # receive length from root
            if num_ready > 0
                ready_ids = Vector{Int}(undef, num_ready)
                MPI.Bcast!(ready_ids, 0, comm)
            else
                ready_ids = Int[]
            end
        end
        
        # 7) Execute destroy commands on all ranks
        for counter_id in ready_ids
            if haskey(manager.objs, counter_id)
                obj = manager.objs[counter_id]
                destroy_obj!(obj)
                delete!(manager.objs, counter_id)

                if rank == 0
                    delete!(manager.counter_pool, counter_id)
                    push!(manager.free_ids, counter_id)
                end
            end
        end

        # Mark all ready_ids as destroyed (prevents duplicate releases from later finalizers)
        for counter_id in ready_ids
            manager.destroyed[counter_id] = true
        end
        
        # 8) Loop again to catch cascaded finalizers/releases
    #end
    
    return nothing
end

"""
    check_and_destroy!(manager=default_manager[]; max_check_count::Integer=1)

Perform garbage collection and process pending object releases, destroying objects when
all ranks have released their references.

This function must be called explicitly to allow controlled cleanup points in the application.
It performs a full garbage collection to trigger finalizers, then processes all pending
release messages and collectively destroys objects that are ready.

The `max_check_count` parameter controls throttling: the function only performs cleanup
every `max_check_count` calls. This reduces overhead in tight loops.

# Example
```julia
SafeMPI.check_and_destroy!()  # Process releases immediately
SafeMPI.check_and_destroy!(max_check_count=10)  # Only cleanup every 10th call
```

See also: [`DRef`](@ref), [`DistributedRefManager`](@ref)
"""
function check_and_destroy!(manager=default_manager[]; max_check_count::Integer=1)
    manager.check_count += 1
    if manager.check_count < max_check_count
        return
    end
    manager.check_count = 0

    GC.gc(true)  # full GC
    # Flush all pending releases at this safe point
    _flush_releases!(manager, MPI.COMM_WORLD)

    return Int[]  # ready_ids are now processed inside _flush_releases!
end

"""
    mpi_any(local_bool::Bool, comm=MPI.COMM_WORLD) -> Bool

Collective logical OR reduction across all ranks in `comm`.

Returns `true` on all ranks if any rank has `local_bool == true`, otherwise returns `false`
on all ranks. This is useful for checking whether any rank encountered an error or
special condition.

# Example
```julia
local_error = (x < 0)  # Some local condition
if SafeMPI.mpi_any(local_error)
    # At least one rank has an error, all ranks enter this branch
    error("Error detected on at least one rank")
end
```

See also: [`@mpiassert`](@ref)
"""
function mpi_any(local_bool::Bool, comm=MPI.COMM_WORLD)
    # Returns true on all ranks if any rank has local_bool == true
    return MPI.Allreduce(local_bool, MPI.LOR, comm)
end

"""
    mpierror(msg::AbstractString, trace::Bool; comm=MPI.COMM_WORLD, code::Integer=1)

Best-effort MPI-wide error terminator that avoids hangs:
- Prints `[rank N] ERROR: msg` on each process that reaches it
- If `trace` is true, prints a backtrace
- If MPI is initialized, aborts the communicator to cleanly stop all ranks
  (avoids deadlocks if other ranks are not in the same code path)
- Falls back to `exit(code)` if MPI is not initialized or already finalized
"""
function mpierror(msg::AbstractString, trace::Bool; comm::MPI.Comm=MPI.COMM_WORLD, code::Integer=1)
    rank = MPI.Comm_rank(comm)
    println("[rank $rank] ERROR: ", msg)
    if trace
        Base.show_backtrace(stdout, backtrace())
    end
    flush(stdout); flush(stderr)
    MPI.Abort(comm, Int(code))
    return nothing
end

"""
    @mpiassert cond [message]

MPI-aware assertion that checks `cond` on all ranks and triggers collective error handling
if any rank fails the assertion.

Each rank evaluates `cond` locally. If any rank has `cond == false`, all ranks are notified
via `mpi_any()` and collectively enter error handling via `mpierror()`. Only ranks where
the assertion failed will print a backtrace.

The assertion is skipped entirely if `enable_assert[] == false` (see `set_assert`).

# Arguments
- `cond`: Boolean expression to check (assertion passes when `cond == true`)
- `message`: Optional custom error message (defaults to auto-generated message with file/line info)

# Example
```julia
# Assert that all ranks have the same value
@mpiassert SafeMPI.mpi_uniform(A) "Matrix A must be uniform across ranks"

# Assert a local condition that must hold on all ranks
@mpiassert n > 0 "Array size must be positive"
```

See also: [`mpi_any`](@ref), [`mpierror`](@ref), [`set_assert`](@ref)
"""
macro mpiassert(cond, msg="")
    file_node = QuoteNode(__source__.file)
    line_val  = __source__.line
    cond_node = QuoteNode(cond)
    # Minimal state outside slow path
    ok_g       = gensym(:ok)
    # Only keep user message evaluated once in slow path
    usermsg_g  = gensym(:user_msg)
    return quote
        if enable_assert[]
            let $ok_g = Bool($(esc(cond)))
                if mpi_any(!$ok_g, MPI.COMM_WORLD)
                    let $usermsg_g = $(esc(msg))
                        # Default message built inline to avoid extra bindings
                        local _msg = ($usermsg_g === "" ? (
                            "assertion failed: " * string($cond_node) *
                            " at " * String($file_node) * ":" * string($line_val) *
                            " (rank " * string(MPI.Comm_rank(MPI.COMM_WORLD)) * "/" * string(MPI.Comm_size(MPI.COMM_WORLD)) * ")"
                        ) : string($usermsg_g))
                        mpierror(_msg, !$ok_g; comm=MPI.COMM_WORLD)
                    end
                end
            end
        end
        nothing
    end
end

function mpi_uniform(A)
    # Compute a consistent hash of A (for any type)
    local_hash = _hash_object(A)  # Vector{UInt8}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Broadcast rank 0's hash to all ranks
    ref_hash = MPI.bcast(rank == 0 ? local_hash : Vector{UInt8}(), 0, MPI.COMM_WORLD)

    # Compare locally and reduce with logical AND across ranks
    local_equal = (local_hash == ref_hash)
    return MPI.Allreduce(local_equal, MPI.LAND, MPI.COMM_WORLD)
end

# helper to create a stable hash from any Julia object
function _hash_object(A)
    io = IOBuffer()
    serialize(io, A)
    return sha1(take!(io))
end

end # module
