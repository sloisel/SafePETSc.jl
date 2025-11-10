# -----------------------------------------------------------------------------
# Vector Pooling Infrastructure
# -----------------------------------------------------------------------------

"""
    PooledVec{T}

Stores a pooled PETSc vector along with its partition information for reuse.
"""
struct PooledVec{T}
    vec::PETSc.Vec{T}
    row_partition::Vector{Int}
end

"""
    ENABLE_VEC_POOL

Global flag to enable/disable vector pooling. Set to `false` to disable pooling.
"""
const ENABLE_VEC_POOL = Ref{Bool}(true)

# Vector pool: Dict{(nglobal, prefix) => Vector{PooledVec{T}}}
# Separate pool per PetscScalar type, initialized below
PETSc.@for_libpetsc begin
    const $(Symbol(:VEC_POOL_, PetscScalar)) = Dict{Tuple{Int,String}, Vector{PooledVec{$PetscScalar}}}()
end

# -----------------------------------------------------------------------------
# Vector Construction
# -----------------------------------------------------------------------------

"""
    Vec_uniform(v::Vector{T}; row_partition=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)), prefix="") -> Vec{T}

Create a distributed PETSc vector from a Julia vector, asserting uniform distribution across ranks (on MPI.COMM_WORLD).

- `v::Vector{T}` must be identical on all ranks (`mpi_uniform`).
- `row_partition` is a Vector{Int} of length `nranks+1` with 1-based inclusive starts.
- `prefix` sets `VecSetOptionsPrefix` for PETSc options.
- Returns a `Vec{T}` (aka `DRef{_Vec{T}}`) managed collectively; by default vectors are returned to a reuse pool when released, not immediately destroyed. Use `ENABLE_VEC_POOL[] = false` or `clear_vec_pool!()` to force destruction.
"""
function Vec_uniform(v::Vector{T};
                               row_partition::Vector{Int}=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)),
                               prefix::String="") where T
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank   = MPI.Comm_rank(MPI.COMM_WORLD)

    # Preconditions - coalesced into single MPI synchronization
    partition_valid = length(row_partition) == nranks + 1 &&
                      row_partition[1] == 1 &&
                      row_partition[end] == length(v) + 1 &&
                      all(r -> row_partition[r] <= row_partition[r+1], 1:nranks)
    @mpiassert SafeMPI.mpi_uniform(v) && partition_valid "Vec_uniform requires v to be mpi_uniform across all ranks, and row_partition must have length nranks+1, start at 1, end at N+1, and be strictly increasing"

    # Local sizes
    lo = row_partition[rank+1]
    hi = row_partition[rank+2] - 1
    nlocal = hi - lo + 1
    nglobal = length(v)

    # Create distributed PETSc Vec (no finalizer; collective destroy via DRef)
    petsc_vec = _vec_create_mpi_for_T(T, nlocal, nglobal, prefix, row_partition)

    # Fill local portion from v
    local_view = PETSc.unsafe_localarray(petsc_vec; read=true, write=true)
    try
        @inbounds local_view[:] = @view v[lo:hi]
    finally
        Base.finalize(local_view)  # restores array view in PETSc
    end
    PETSc.assemble(petsc_vec)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Vec{T}(petsc_vec, row_partition, prefix)
    return SafeMPI.DRef(obj)
end

"""
    Vec_sum(v::SparseVector{T}; row_partition=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)), prefix="", own_rank_only=false) -> Vec{T}

Create a distributed PETSc vector by summing sparse vectors across ranks (on MPI.COMM_WORLD).

- `v::SparseVector{T}` can differ across ranks; nonzeros are summed across all ranks.
- `row_partition` is a Vector{Int} of length `nranks+1` with 1-based inclusive starts.
- `prefix` sets `VecSetOptionsPrefix` for PETSc options.
- `own_rank_only::Bool` (default=false): if true, asserts that all nonzero indices fall within this rank's row partition.
- Returns a `Vec{T}` managed collectively; by default vectors are returned to a reuse pool when released, not immediately destroyed. Use `ENABLE_VEC_POOL[] = false` or `clear_vec_pool!()` to force destruction.

Uses `VecSetValues` with `ADD_VALUES` to sum contributions across ranks.
"""
function Vec_sum(v::SparseVector{T};
                 row_partition::Vector{Int}=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)),
                 prefix::String="",
                 own_rank_only::Bool=false) where T
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank   = MPI.Comm_rank(MPI.COMM_WORLD)

    # Preconditions - coalesced into single MPI synchronization
    partition_valid = length(row_partition) == nranks + 1 &&
                      row_partition[1] == 1 &&
                      row_partition[end] == length(v) + 1 &&
                      all(r -> row_partition[r] <= row_partition[r+1], 1:nranks)

    # If own_rank_only, validate all nonzero indices fall within this rank's partition
    own_rank_ok = true
    if own_rank_only
        lo = row_partition[rank+1]
        hi = row_partition[rank+2] - 1
        nz_indices, _ = findnz(v)
        @inbounds for idx in nz_indices
            if !(lo <= idx <= hi)
                own_rank_ok = false
                break
            end
        end
    end

    @mpiassert partition_valid && own_rank_ok "row_partition must have length nranks+1, start at 1, end at N+1, and be strictly increasing; if own_rank_only=true, all nonzeros must fall within this rank's partition"

    # Local sizes
    lo = row_partition[rank+1]
    hi = row_partition[rank+2] - 1
    nlocal = hi - lo + 1
    nglobal = length(v)

    # Create distributed PETSc Vec (no finalizer; collective destroy via DRef)
    petsc_vec = _vec_create_mpi_for_T(T, nlocal, nglobal, prefix, row_partition)

    # Extract nonzero indices and values from sparse vector
    nz_indices, nz_values = findnz(v)

    # Set values using ADD_VALUES mode to sum across ranks
    if !isempty(nz_indices)
        _vec_setvalues!(petsc_vec, nz_indices, nz_values, PETSc.ADD_VALUES)
    end

    # Assemble the vector (required after VecSetValues)
    PETSc.assemble(petsc_vec)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Vec{T}(petsc_vec, row_partition, prefix)
    return SafeMPI.DRef(obj)
end

# Create a distributed PETSc Vec for a given element type T by dispatching to the
# underlying PETSc scalar variant via PETSc.@for_libpetsc
# This function checks the pool first before creating a new vector
function _vec_create_mpi_for_T(::Type{T}, nlocal::Integer, nglobal::Integer, prefix::String="", row_partition::Vector{Int}=Int[]) where {T}
    return _vec_create_mpi_impl(T, nlocal, nglobal, prefix, row_partition)
end

# Return a vector to the pool for reuse
function _return_vec_to_pool!(v::PETSc.Vec{T}, row_partition::Vector{Int}, prefix::String) where {T}
    return _return_vec_to_pool_impl!(v, row_partition, prefix)
end

PETSc.@for_libpetsc begin
    function _vec_create_mpi_impl(::Type{$PetscScalar}, nlocal::Integer, nglobal::Integer, prefix::String="", row_partition::Vector{Int}=Int[])
        # Try to get from pool first (only if row_partition is provided for matching)
        if ENABLE_VEC_POOL[] && !isempty(row_partition)
            pool = $(Symbol(:VEC_POOL_, PetscScalar))
            pool_key = (Int(nglobal), prefix)
            if haskey(pool, pool_key)
                pool_list = pool[pool_key]
                # Scan for matching row_partition
                for (i, pooled) in enumerate(pool_list)
                    if pooled.row_partition == row_partition
                        # Found match - remove from pool and return
                        deleteat!(pool_list, i)
                        if isempty(pool_list)
                            delete!(pool, pool_key)
                        end
                        return pooled.vec
                    end
                end
            end
        end

        # Pool miss or pooling disabled - create new vector
        vec = PETSc.Vec{$PetscScalar}(C_NULL)
        PETSc.@chk ccall((:VecCreate, $libpetsc), PETSc.PetscErrorCode,
                         (MPI.MPI_Comm, Ptr{CVec}), MPI.COMM_WORLD, vec)
        PETSc.@chk ccall((:VecSetSizes, $libpetsc), PETSc.PetscErrorCode,
                         (CVec, $PetscInt, $PetscInt), vec, $PetscInt(nlocal), $PetscInt(nglobal))
        if !isempty(prefix)
            PETSc.@chk ccall((:VecSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CVec, Cstring), vec, prefix)
        end
        PETSc.@chk ccall((:VecSetFromOptions, $libpetsc), PETSc.PetscErrorCode,
                         (CVec,), vec)
        return vec
    end

    function _destroy_petsc_vec!(v::PETSc.AbstractVec{$PetscScalar})
        PETSc.finalized($petsclib) || begin
            PETSc.@chk ccall((:VecDestroy, $libpetsc), PETSc.PetscErrorCode,
                             (Ptr{CVec},), v)
            v.ptr = C_NULL
        end
        return nothing
    end

    function _return_vec_to_pool_impl!(v::PETSc.Vec{$PetscScalar}, row_partition::Vector{Int}, prefix::String)
        # Don't pool if PETSc is finalizing
        if PETSc.finalized($petsclib)
            return nothing
        end

        # Zero out vector contents before returning to pool
        PETSc.@chk ccall((:VecZeroEntries, $libpetsc), PETSc.PetscErrorCode,
                         (CVec,), v)

        # Get global size
        nglobal = Ref{$PetscInt}(0)
        PETSc.@chk ccall((:VecGetSize, $libpetsc), PETSc.PetscErrorCode,
                         (CVec, Ptr{$PetscInt}), v, nglobal)

        # Add to pool
        pool = $(Symbol(:VEC_POOL_, PetscScalar))
        pool_key = (Int(nglobal[]), prefix)
        if !haskey(pool, pool_key)
            pool[pool_key] = PooledVec{$PetscScalar}[]
        end
        push!(pool[pool_key], PooledVec{$PetscScalar}(v, row_partition))

        return nothing
    end

    function _vec_setvalues_impl!(vec::PETSc.Vec{$PetscScalar}, indices::Vector{Int},
                                   values::Vector{$PetscScalar}, mode::PETSc.InsertMode)
        # Convert 1-based Julia indices to 0-based PETSc indices
        indices_c = $PetscInt.(indices .- 1)
        n = length(indices)

        PETSc.@chk ccall((:VecSetValues, $libpetsc), PETSc.PetscErrorCode,
                         (CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}, PETSc.InsertMode),
                         vec, $PetscInt(n), indices_c, values, mode)
        return nothing
    end

    # Get ownership range [lo, hi) in 0-based PETSc indexing; return 1-based inclusive
    function _vec_get_ownership_range(vec::PETSc.Vec{$PetscScalar})
        lo = Ref{$PetscInt}(0)
        hi = Ref{$PetscInt}(0)
        PETSc.@chk ccall((:VecGetOwnershipRange, $libpetsc), PETSc.PetscErrorCode,
                         (CVec, Ptr{$PetscInt}, Ptr{$PetscInt}), vec, lo, hi)
        # Convert: 0-based [lo, hi) -> 1-based inclusive [lo+1, hi]
        return Int(lo[] + 1), Int(hi[])
    end
end

# Generic wrapper for _vec_setvalues!
function _vec_setvalues!(vec::PETSc.Vec{T}, indices::Vector{Int}, values::Vector{T}, mode::PETSc.InsertMode) where {T}
    return _vec_setvalues_impl!(vec, indices, values, mode)
end

# Default row partition: equal blocks across ranks
"""
    default_row_partition(n::Int, nranks::Int) -> Vector{Int}

Create a default row partition that divides n rows equally among nranks.

Returns a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
"""
function default_row_partition(n::Int, nranks::Int)
    partition = Vector{Int}(undef, nranks + 1)
    base_size = div(n, nranks)
    remainder = mod(n, nranks)
    partition[1] = 1
    for r in 0:(nranks-1)
        local_size = base_size + (r < remainder ? 1 : 0)
        partition[r+2] = partition[r+1] + local_size
    end
    return partition
end

# -----------------------------------------------------------------------------
# In-place broadcasting support (Option A)
# y .= f.(x, ...)
# - Only supports mixing SafePETSc.Vec operands and scalars
# - All Vec operands must share the same row_partition as the destination
# -----------------------------------------------------------------------------

# Custom broadcast style for SafePETSc.Vec
struct VecBroadcastStyle <: Base.Broadcast.BroadcastStyle end
# Prefer DRef{_Vec} as the user-visible broadcast participant
Base.BroadcastStyle(::Type{SafeMPI.DRef{<:_Vec}}) = VecBroadcastStyle()
Base.BroadcastStyle(::SafeMPI.DRef{<:_Vec}) = VecBroadcastStyle()
Base.BroadcastStyle(::Type{T}) where {U<:_Vec,T<:SafeMPI.DRef{U}} = VecBroadcastStyle()
Base.BroadcastStyle(::VecBroadcastStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = VecBroadcastStyle()
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{0}, ::VecBroadcastStyle) = VecBroadcastStyle()
Base.BroadcastStyle(::VecBroadcastStyle, ::Base.Broadcast.DefaultArrayStyle{1}) = nothing
Base.BroadcastStyle(::Base.Broadcast.DefaultArrayStyle{1}, ::VecBroadcastStyle) = nothing

# Broadcastable participants
Base.broadcastable(r::SafeMPI.DRef{<:_Vec}) = r
Base.broadcastable(v::_Vec) = throw(ArgumentError("SafePETSc._Vec is not broadcastable; use Vec (a DRef{_Vec}) e.g., dr .= ..., z = dr .+ 4"))

# In-place materialization on a distributed PETSc Vec
function Base.materialize!(dest::_Vec{T}, bc::Base.Broadcast.Broadcasted{VecBroadcastStyle}) where {T}
    # Collect all Vec operands to validate partition compatibility
    vec_args = _Vec{T}[]
    _collect_vec_args!(vec_args, bc)

    # Check partitions match destination - single @mpiassert (pulled out of loop)
    all_partitions_match = all(v.row_partition == dest.row_partition for v in vec_args)
    @mpiassert all_partitions_match "broadcast: all Vec operands must share the same row_partition as destination"

    # Prepare destination local view
    dest_arr = PETSc.unsafe_localarray(dest.v; read=true, write=true)
    finals = Function[]  # finalizers for source local arrays

    # Map Broadcasted tree to use local arrays for Vec operands
    local_bc = _map_to_local(bc, dest, dest_arr, finals)

    try
        Base.materialize!(dest_arr, local_bc)
    finally
        # Restore all source local views first
        for fin in finals
            fin()
        end
        # Then restore destination local view
        Base.finalize(dest_arr)
    end

    return dest
end

# Allow destination to be DRef{Vec}: delegate to underlying Vec
function Base.materialize!(dest::SafeMPI.DRef{<:_Vec{T}}, bc::Base.Broadcast.Broadcasted{VecBroadcastStyle}) where {T}
    Base.materialize!(dest.obj, bc)
    return dest
end

# Recursively collect Vec operands from a Broadcasted expression
_collect_vec_args!(acc::Vector{<:_Vec}, bc::Base.Broadcast.Broadcasted) = (_collect_vec_args!(acc, bc.args); acc)
_collect_vec_args!(acc::Vector{<:_Vec}, args::Tuple) = (foreach(a -> _collect_vec_args!(acc, a), args); acc)
_collect_vec_args!(acc::Vector{<:_Vec}, v::_Vec) = (push!(acc, v); acc)
_collect_vec_args!(acc::Vector{<:_Vec}, ::Any) = acc

# Recursively map Broadcasted tree replacing Vec operands with local arrays
function _map_to_local(bc::Base.Broadcast.Broadcasted{VecBroadcastStyle}, dest::_Vec, dest_arr, finals::Vector{Function})
    mapped_args = map(a -> _map_arg_to_local(a, dest, dest_arr, finals), bc.args)
    return Base.Broadcast.Broadcasted(bc.f, mapped_args, bc.axes)
end

function _map_arg_to_local(a, dest::_Vec, dest_arr, finals)
    if a isa Base.Broadcast.Broadcasted{VecBroadcastStyle}
        return _map_to_local(a, dest, dest_arr, finals)
    elseif a isa SafeMPI.DRef
        return _map_arg_to_local(a.obj, dest, dest_arr, finals)
    elseif a isa _Vec
        if a === dest
            # Use the destination local array as read source
            return @view dest_arr[:]
        else
            arr = PETSc.unsafe_localarray(a.v; read=true, write=false)
            push!(finals, () -> Base.finalize(arr))
            return @view arr[:]
        end
    elseif isa(a, Number) || isa(a, Base.RefValue)
        return a
    else
        error("broadcast with argument of type $(typeof(a)) is not supported; use only scalars and SafePETSc.Vec")
    end
end

# Element type and shape for internal _Vec and DRef-wrapped _Vec
Base.eltype(::_Vec{T}) where {T} = T
Base.size(v::_Vec) = (v.row_partition[end] - 1,)
Base.size(v::_Vec, d::Integer) = d == 1 ? (v.row_partition[end] - 1) : 1
Base.axes(v::_Vec) = (Base.OneTo(v.row_partition[end] - 1),)
Base.length(v::_Vec) = v.row_partition[end] - 1

Base.eltype(r::SafeMPI.DRef{<:_Vec}) = Base.eltype(r.obj)
Base.size(r::SafeMPI.DRef{<:_Vec}) = Base.size(r.obj)
Base.size(r::SafeMPI.DRef{<:_Vec}, d::Integer) = Base.size(r.obj, d)
Base.axes(r::SafeMPI.DRef{<:_Vec}) = Base.axes(r.obj)
Base.length(r::SafeMPI.DRef{<:_Vec}) = Base.length(r.obj)

# Out-of-place broadcast: allocate a new Vec (wrapped in DRef) and compute into it.
function Base.copy(bc::Base.Broadcast.Broadcasted{VecBroadcastStyle})
    # Find representative Vec for partition/prefix
    vrep = _first__vec(bc)
    vrep === nothing && error("broadcast requires at least one SafePETSc.Vec operand")
    rowp = vrep.row_partition
    N = rowp[end] - 1

    # Determine result eltype
    Tres = Base.Broadcast.combine_eltypes(bc.f, bc.args)

    # Allocate distributed Vec with same partition/prefix
    dr = Vec_uniform(zeros(Tres, N); row_partition=rowp, prefix=vrep.prefix)
    y = dr.obj
    Base.materialize!(y, bc)
    return dr
end

# Helper: locate first _Vec in Broadcasted tree
_first__vec(bc::Base.Broadcast.Broadcasted{VecBroadcastStyle}) = _first__vec(bc.args)
_first__vec(args::Tuple) = (_first__vec_iter(args))
_first__vec_iter(args) = begin
    for a in args
        v = _first__vec(a)
        v !== nothing && return v
    end
    return nothing
end
_first__vec(v::_Vec) = v
_first__vec(r::SafeMPI.DRef{<:_Vec}) = r.obj
_first__vec(::Any) = nothing

# -----------------------------------------------------------------------------
# Helper constructors (user-facing): zeros_like, ones_like, fill_like
# -----------------------------------------------------------------------------

"""
    zeros_like(x::Vec{T}; T2::Type{S}=T, prefix::String=x.obj.prefix) -> Vec{S}

Create a new distributed vector with the same size and partition as `x`, filled with zeros.

# Arguments
- `x`: Template vector to match size and partition
- `T2`: Element type of the result (defaults to same type as `x`)
- `prefix`: PETSc options prefix (defaults to same prefix as `x`)

See also: [`ones_like`](@ref), [`fill_like`](@ref), [`Vec_uniform`](@ref)
"""
function zeros_like(x::Vec{T}; T2::Type{S}=T, prefix::String=x.obj.prefix) where {T,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    return Vec_uniform(zeros(S, N); row_partition=rowp, prefix=prefix)
end

"""
    ones_like(x::Vec{T}; T2::Type{S}=T, prefix::String=x.obj.prefix) -> Vec{S}

Create a new distributed vector with the same size and partition as `x`, filled with ones.

# Arguments
- `x`: Template vector to match size and partition
- `T2`: Element type of the result (defaults to same type as `x`)
- `prefix`: PETSc options prefix (defaults to same prefix as `x`)

See also: [`zeros_like`](@ref), [`fill_like`](@ref), [`Vec_uniform`](@ref)
"""
function ones_like(x::Vec{T}; T2::Type{S}=T, prefix::String=x.obj.prefix) where {T,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    return Vec_uniform(ones(S, N); row_partition=rowp, prefix=prefix)
end

"""
    fill_like(x::Vec{T}, val; T2::Type{S}=typeof(val), prefix::String=x.obj.prefix) -> Vec{S}

Create a new distributed vector with the same size and partition as `x`, filled with `val`.

# Arguments
- `x`: Template vector to match size and partition
- `val`: Value to fill the vector with
- `T2`: Element type of the result (defaults to type of `val`)
- `prefix`: PETSc options prefix (defaults to same prefix as `x`)

# Example
```julia
y = fill_like(x, 3.14)  # Create a vector like x, filled with 3.14
```

See also: [`zeros_like`](@ref), [`ones_like`](@ref), [`Vec_uniform`](@ref)
"""
function fill_like(x::Vec{T}, val; T2::Type{S}=typeof(val), prefix::String=x.obj.prefix) where {T,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    v = fill(S(val), N)
    return Vec_uniform(v; row_partition=rowp, prefix=prefix)
end

# -----------------------------------------------------------------------------
# Sugar: + and - forward to broadcasting
# -----------------------------------------------------------------------------
Base.:+(x::Vec, y::Vec) = x .+ y
Base.:-(x::Vec, y::Vec) = x .- y
Base.:-(x::Vec) = (-one(eltype(x))) .* x
Base.:+(x::Vec) = x

# Mixed scalar-Vec
Base.:+(x::Vec, y::Number) = x .+ y
Base.:+(x::Number, y::Vec) = x .+ y
Base.:-(x::Vec, y::Number) = x .- y
Base.:-(x::Number, y::Vec) = x .- y

# Adjoint support for Vec
Base.adjoint(v::Vec{T}) where {T} = LinearAlgebra.Adjoint(v)

# Adjoint-vector times matrix: w' = v' * A
function Base.:*(vt::LinearAlgebra.Adjoint{T, <:Vec{T}}, A::Mat{T}) where {T}
    v = parent(vt)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    vec_length = size(v)[1]
    m, n = size(A)
    @mpiassert vec_length == m && v.obj.row_partition == A.obj.row_partition && v.obj.prefix == A.obj.prefix "Vector length must match matrix rows (v: $(vec_length), A: $(m)×$(n)), row partitions must match, and v and A must have the same prefix"

    # Create result vector with A's column partition (transpose result)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal = col_hi - col_lo + 1

    w_petsc = _vec_create_mpi_for_T(T, nlocal, n, v.obj.prefix, A.obj.col_partition)

    # Perform w = A^T * v using PETSc
    _mat_mult_transpose_vec!(w_petsc, A.obj.A, v.obj.v)

    PETSc.assemble(w_petsc)

    # Wrap in DRef and return as adjoint
    obj = _Vec{T}(w_petsc, A.obj.col_partition, v.obj.prefix)
    w = SafeMPI.DRef(obj)
    return LinearAlgebra.Adjoint(w)
end

# In-place adjoint vector-matrix multiplication: w = v' * A (reuses pre-allocated w)
# Note: Computes w = A^T * v where v is the parent of the adjoint
function LinearAlgebra.mul!(w::Vec{T}, vt::LinearAlgebra.Adjoint{T, <:Vec{T}}, A::Mat{T}) where {T}
    v = parent(vt)

    # Check dimensions and partitioning - single @mpiassert for efficiency
    v_length = size(v)[1]
    w_length = size(w)[1]
    m, n = size(A)
    @mpiassert (v_length == m && w_length == n &&
                v.obj.row_partition == A.obj.row_partition &&
                w.obj.row_partition == A.obj.col_partition &&
                v.obj.prefix == A.obj.prefix == w.obj.prefix) "Input vector v must have length matching matrix rows (v: $(v_length), A: $(m)×$(n)), output vector w must have length matching matrix columns (w: $(w_length)), input vector partition must match matrix row partition, output vector partition must match matrix column partition, and all objects must have the same prefix"

    # Perform w = A^T * v using PETSc (reuses w)
    _mat_mult_transpose_vec!(w.obj.v, A.obj.A, v.obj.v)

    PETSc.assemble(w.obj.v)

    return w
end

# PETSc matrix-transpose-vector multiplication wrapper
PETSc.@for_libpetsc begin
    function _mat_mult_transpose_vec!(w::PETSc.Vec{$PetscScalar}, A::PETSc.Mat{$PetscScalar}, v::PETSc.Vec{$PetscScalar})
        PETSc.@chk ccall((:MatMultTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, CVec, CVec),
                         A, v, w)
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Vector Pool Utility Functions
# -----------------------------------------------------------------------------

"""
    clear_vec_pool!()

Clear all vectors from the pool, destroying them immediately.
Useful for testing or explicit memory management.
"""
function clear_vec_pool!()
    # Clear all pools by iterating over each PetscScalar type
    PETSc.@for_libpetsc begin
        pool = $(Symbol(:VEC_POOL_, PetscScalar))
        for (key, vec_list) in pool
            for pooled in vec_list
                _destroy_petsc_vec!(pooled.vec)
            end
        end
        empty!(pool)
    end
    return nothing
end

"""
    get_vec_pool_stats() -> Dict

Return statistics about the current vector pool state.
Returns a dictionary with keys (nglobal, prefix, type) => count.
"""
function get_vec_pool_stats()
    stats = Dict{Tuple{Int,String,Type}, Int}()
    # Gather stats from all pools using eval to access the pools
    # We need to do this carefully because @for_libpetsc creates separate scopes
    for petsc_scalar in [Float64, ComplexF64]  # Common PETSc scalar types
        pool_name = Symbol(:VEC_POOL_, petsc_scalar)
        if isdefined(@__MODULE__, pool_name)
            pool = getfield(@__MODULE__, pool_name)
            for (key, vec_list) in pool
                stats[(key[1], key[2], petsc_scalar)] = length(vec_list)
            end
        end
    end
    return stats
end
