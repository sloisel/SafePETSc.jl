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

# Vector pool: Dict{(nglobal, PrefixType) => Vector{PooledVec{T}}}
# Separate pool per PetscScalar type, initialized below
PETSc.@for_libpetsc begin
    const $(Symbol(:VEC_POOL_, PetscScalar)) = Dict{Tuple{Int,Type}, Vector{PooledVec{$PetscScalar}}}()
end

# -----------------------------------------------------------------------------
# Vector Construction
# -----------------------------------------------------------------------------

"""
    Vec_uniform(v::Vector{T}; row_partition=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)), Prefix::Type=MPIAIJ) -> Vec{T,Prefix}

**MPI Collective**

Create a distributed PETSc vector from a Julia vector, asserting uniform distribution across ranks (on MPI.COMM_WORLD).

- `v::Vector{T}` must be identical on all ranks (`mpi_uniform`).
- `row_partition` is a Vector{Int} of length `nranks+1` with 1-based inclusive starts.
- `Prefix` is a type parameter for `VecSetOptionsPrefix` for PETSc options (default: MPIAIJ).
- Returns a `Vec{T,Prefix}` (aka `DRef{_Vec{T,Prefix}}`) managed collectively; by default vectors are returned to a reuse pool when released, not immediately destroyed. Use `ENABLE_VEC_POOL[] = false` or `clear_vec_pool!()` to force destruction.
"""
function Vec_uniform(v::Vector{T};
                               row_partition::Vector{Int}=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)),
                               Prefix::Type=MPIAIJ) where T
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
    petsc_vec = _vec_create_mpi_for_T(T, nlocal, nglobal, Prefix, row_partition)

    # Fill local portion from v
    local_view = PETSc.unsafe_localarray(petsc_vec; read=true, write=true)
    try
        @inbounds local_view[:] = @view v[lo:hi]
    finally
        Base.finalize(local_view)  # restores array view in PETSc
    end
    PETSc.assemble(petsc_vec)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Vec{T,Prefix}(petsc_vec, row_partition)
    return SafeMPI.DRef(obj)
end

"""
    Vec_sum(v::SparseVector{T}; row_partition=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)), Prefix::Type=MPIAIJ, own_rank_only=false) -> Vec{T,Prefix}

**MPI Collective**

Create a distributed PETSc vector by summing sparse vectors across ranks (on MPI.COMM_WORLD).

- `v::SparseVector{T}` can differ across ranks; nonzeros are summed across all ranks.
- `row_partition` is a Vector{Int} of length `nranks+1` with 1-based inclusive starts.
- `Prefix` is a type parameter for `VecSetOptionsPrefix` for PETSc options (default: MPIAIJ).
- `own_rank_only::Bool` (default=false): if true, asserts that all nonzero indices fall within this rank's row partition.
- Returns a `Vec{T,Prefix}` managed collectively; by default vectors are returned to a reuse pool when released, not immediately destroyed. Use `ENABLE_VEC_POOL[] = false` or `clear_vec_pool!()` to force destruction.

Uses `VecSetValues` with `ADD_VALUES` to sum contributions across ranks.
"""
function Vec_sum(v::SparseVector{T};
                 row_partition::Vector{Int}=default_row_partition(length(v), MPI.Comm_size(MPI.COMM_WORLD)),
                 Prefix::Type=MPIAIJ,
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
    petsc_vec = _vec_create_mpi_for_T(T, nlocal, nglobal, Prefix, row_partition)

    # Extract nonzero indices and values from sparse vector
    nz_indices, nz_values = findnz(v)

    # Set values using ADD_VALUES mode to sum across ranks
    if !isempty(nz_indices)
        _vec_setvalues!(petsc_vec, nz_indices, nz_values, PETSc.ADD_VALUES)
    end

    # Assemble the vector (required after VecSetValues)
    PETSc.assemble(petsc_vec)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Vec{T,Prefix}(petsc_vec, row_partition)
    return SafeMPI.DRef(obj)
end

# Create a distributed PETSc Vec for a given element type T by dispatching to the
# underlying PETSc scalar variant via PETSc.@for_libpetsc
# This function checks the pool first before creating a new vector
function _vec_create_mpi_for_T(::Type{T}, nlocal::Integer, nglobal::Integer, Prefix::Type, row_partition::Vector{Int}=Int[]) where {T}
    return _vec_create_mpi_impl(T, nlocal, nglobal, Prefix, row_partition)
end

# Return a vector to the pool for reuse
function _return_vec_to_pool!(v::PETSc.Vec{T}, row_partition::Vector{Int}, Prefix::Type) where {T}
    return _return_vec_to_pool_impl!(v, row_partition, Prefix)
end

PETSc.@for_libpetsc begin
    function _vec_create_mpi_impl(::Type{$PetscScalar}, nlocal::Integer, nglobal::Integer, Prefix::Type, row_partition::Vector{Int}=Int[])
        # Try to get from pool first (only if row_partition is provided for matching)
        if ENABLE_VEC_POOL[] && !isempty(row_partition)
            pool = $(Symbol(:VEC_POOL_, PetscScalar))
            pool_key = (Int(nglobal), Prefix)
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

        # Set prefix and let PETSc options determine the type
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:VecSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CVec, Cstring), vec, prefix_str)
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

    function _return_vec_to_pool_impl!(v::PETSc.Vec{$PetscScalar}, row_partition::Vector{Int}, Prefix::Type)
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
        pool_key = (Int(nglobal[]), Prefix)
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

**MPI Non-Collective**

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

# Adjoint of Vec behaves as a row vector (1 × n matrix)
Base.size(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix} = (1, length(parent(vt)))
Base.size(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}, d::Integer) where {T,Prefix} = d == 1 ? 1 : (d == 2 ? length(parent(vt)) : 1)
Base.length(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix} = length(parent(vt))
Base.axes(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix} = (Base.OneTo(1), Base.OneTo(length(parent(vt))))

# Note: getindex is intentionally not defined for Vec adjoints
# They should only be used in matrix operations like v' * w or v' * A
# BlockProduct iteration uses explicit 2D indexing on the block matrix itself,
# not on the Vec adjoint elements

# Scalar multiplication with vectors
Base.:*(α::Number, v::Vec{T,Prefix}) where {T,Prefix} = α .* v
Base.:*(v::Vec{T,Prefix}, α::Number) where {T,Prefix} = α .* v

# Scalar multiplication with adjoint vectors
Base.:*(α::Number, vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix} = (α * parent(vt))'
Base.:*(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}, α::Number) where {T,Prefix} = (α * parent(vt))'

# Addition of adjoint vectors (row vectors)
Base.:+(vt1::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}, vt2::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix} = (parent(vt1) + parent(vt2))'

# Outer product: v * w' (returns Mat)
function Base.:*(v::Vec{T,Prefix}, wt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}) where {T,Prefix}
    w = parent(wt)

    # Check dimensions and partitioning
    m = length(v)
    n = length(w)

    @mpiassert v.obj.row_partition == w.obj.row_partition "Vectors must have the same row partition"

    # Create an m × n matrix
    # Each rank owns a subset of rows and columns
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Use v's row partition as both row and column partition (for v and w)
    # Result has: rows partitioned like v, columns partitioned like w
    row_partition = v.obj.row_partition
    col_partition = w.obj.row_partition  # Use w's row partition as column partition

    row_lo = row_partition[rank+1]
    row_hi = row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    col_lo = col_partition[rank+1]
    col_hi = col_partition[rank+2] - 1
    nlocal_cols = col_hi - col_lo + 1

    # Create distributed PETSc matrix
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, m, n, Prefix)

    # Get local portions of v and w
    v_local = PETSc.unsafe_localarray(v.obj.v; read=true)
    w_local = PETSc.unsafe_localarray(w.obj.v; read=true)

    # Set values: result[i,j] = v[i] * w[j]
    # Each rank computes its local rows and local columns only
    for i_local in 1:nlocal_rows
        i_global = row_lo + i_local - 1
        row_values = Vector{T}(undef, nlocal_cols)
        col_indices = Vector{Int}(undef, nlocal_cols)

        for j_local in 1:nlocal_cols
            j_global = col_lo + j_local - 1
            row_values[j_local] = v_local[i_local] * w_local[j_local]
            col_indices[j_local] = j_global
        end

        _mat_setvalues!(petsc_mat, [i_global], col_indices, row_values, PETSc.INSERT_VALUES)
    end

    # Assemble the matrix
    PETSc.assemble(petsc_mat)

    # Wrap in DRef
    obj = _Mat{T,Prefix}(petsc_mat, row_partition, col_partition)
    return SafeMPI.DRef(obj)
end

# Out-of-place broadcast: allocate a new Vec (wrapped in DRef) and compute into it.
function Base.copy(bc::Base.Broadcast.Broadcasted{VecBroadcastStyle})
    # Find representative Vec for partition/prefix
    vrep = _first__vec(bc)
    vrep === nothing && error("broadcast requires at least one SafePETSc.Vec operand")
    rowp = vrep.row_partition
    N = rowp[end] - 1

    # Determine result eltype
    Tres = Base.Broadcast.combine_eltypes(bc.f, bc.args)

    # Extract Prefix type parameter from vrep
    Prefix = typeof(vrep).parameters[2]

    # Allocate distributed Vec with same partition/prefix
    dr = Vec_uniform(zeros(Tres, N); row_partition=rowp, Prefix=Prefix)
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
    zeros_like(x::Vec{T,Prefix}; T2::Type{S}=T, Prefix2::Type=Prefix) -> Vec{S,Prefix2}

**MPI Collective**

Create a new distributed vector with the same size and partition as `x`, filled with zeros.

# Arguments
- `x`: Template vector to match size and partition
- `T2`: Element type of the result (defaults to same type as `x`)
- `Prefix2`: Prefix type (defaults to same prefix as `x`)

See also: [`ones_like`](@ref), [`fill_like`](@ref), [`Vec_uniform`](@ref)
"""
function zeros_like(x::Vec{T,Prefix}; T2::Type{S}=T, Prefix2::Type=Prefix) where {T,Prefix,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    return Vec_uniform(zeros(S, N); row_partition=rowp, Prefix=Prefix2)
end

"""
    ones_like(x::Vec{T,Prefix}; T2::Type{S}=T, Prefix2::Type=Prefix) -> Vec{S,Prefix2}

**MPI Collective**

Create a new distributed vector with the same size and partition as `x`, filled with ones.

# Arguments
- `x`: Template vector to match size and partition
- `T2`: Element type of the result (defaults to same type as `x`)
- `Prefix2`: Prefix type (defaults to same prefix as `x`)

See also: [`zeros_like`](@ref), [`fill_like`](@ref), [`Vec_uniform`](@ref)
"""
function ones_like(x::Vec{T,Prefix}; T2::Type{S}=T, Prefix2::Type=Prefix) where {T,Prefix,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    return Vec_uniform(ones(S, N); row_partition=rowp, Prefix=Prefix2)
end

"""
    fill_like(x::Vec{T,Prefix}, val; T2::Type{S}=typeof(val), Prefix2::Type=Prefix) -> Vec{S,Prefix2}

**MPI Collective**

Create a new distributed vector with the same size and partition as `x`, filled with `val`.

# Arguments
- `x`: Template vector to match size and partition
- `val`: Value to fill the vector with
- `T2`: Element type of the result (defaults to type of `val`)
- `Prefix2`: Prefix type (defaults to same prefix as `x`)

# Example
```julia
y = fill_like(x, 3.14)  # Create a vector like x, filled with 3.14
```

See also: [`zeros_like`](@ref), [`ones_like`](@ref), [`Vec_uniform`](@ref)
"""
function fill_like(x::Vec{T,Prefix}, val; T2::Type{S}=typeof(val), Prefix2::Type=Prefix) where {T,Prefix,S}
    rowp = x.obj.row_partition
    N = rowp[end] - 1
    v = fill(S(val), N)
    return Vec_uniform(v; row_partition=rowp, Prefix=Prefix2)
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
function Base.:*(vt::LinearAlgebra.Adjoint{T, <:Vec{T,Prefix}}, A::Mat{T,Prefix}) where {T,Prefix}
    v = parent(vt)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    vec_length = size(v)[1]
    m, n = size(A)
    @mpiassert vec_length == m && v.obj.row_partition == A.obj.row_partition "Vector length must match matrix rows (v: $(vec_length), A: $(m)×$(n)) and row partitions must match"

    # Create result vector with A's column partition (transpose result)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal = col_hi - col_lo + 1

    w_petsc = _vec_create_mpi_for_T(T, nlocal, n, Prefix, A.obj.col_partition)

    # Perform w = A^T * v using PETSc
    _mat_mult_transpose_vec!(w_petsc, A.obj.A, v.obj.v)

    PETSc.assemble(w_petsc)

    # Wrap in DRef and return as adjoint
    obj = _Vec{T,Prefix}(w_petsc, A.obj.col_partition)
    w = SafeMPI.DRef(obj)
    return LinearAlgebra.Adjoint(w)
end

# Inner product: v' * w (returns scalar)
function Base.:*(vt::LinearAlgebra.Adjoint{T, <:Vec{T}}, w::Vec{T}) where {T}
    v = parent(vt)

    # Check dimensions and partitioning
    v_length = size(v)[1]
    w_length = size(w)[1]
    @mpiassert v_length == w_length && v.obj.row_partition == w.obj.row_partition "Vector lengths must match (v: $(v_length), w: $(w_length)) and row partitions must match"

    # Compute inner product using PETSc VecDot
    return _vec_dot(v.obj.v, w.obj.v)
end

# Row vector times transposed matrix: v' * A' (returns row vector)
function Base.:*(vt::LinearAlgebra.Adjoint{T, <:Vec{T}}, At::LinearAlgebra.Adjoint{T, <:Mat{T}}) where {T}
    # v' * A' = (A * v)'
    v = parent(vt)
    A = parent(At)

    # A * v gives a column vector
    result_vec = A * v

    # Return as row vector (adjoint)
    return result_vec'
end

# PETSc vector dot product wrapper
PETSc.@for_libpetsc begin
    function _vec_dot(v::PETSc.Vec{$PetscScalar}, w::PETSc.Vec{$PetscScalar})
        result = Ref{$PetscScalar}()
        PETSc.@chk ccall((:VecDot, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CVec, PETSc.CVec, Ptr{$PetscScalar}),
                         v, w, result)
        return result[]
    end
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
                w.obj.row_partition == A.obj.col_partition) "Input vector v must have length matching matrix rows (v: $(v_length), A: $(m)×$(n)), output vector w must have length matching matrix columns (w: $(w_length)), input vector partition must match matrix row partition, and output vector partition must match matrix column partition"

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

**MPI Non-Collective**

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

**MPI Non-Collective**

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
                prefix_str = prefix(key[2])  # Convert Prefix type to string
                stats[(key[1], prefix_str, petsc_scalar)] = length(vec_list)
            end
        end
    end
    return stats
end

# -----------------------------------------------------------------------------
# Conversion to Julia Vector
# -----------------------------------------------------------------------------

"""
    Vector(x::Vec{T}) -> Vector{T}

**MPI Collective**

Convert a distributed PETSc Vec to a Julia Vector by gathering all data to all ranks.
This is a collective operation - all ranks must call it and will receive the complete vector.

This is primarily used for display purposes or small vectors. For large vectors, this
operation can be expensive as it gathers all data to all ranks.
"""
function Base.Vector(x::Vec{T}) where T
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    # Get local portion
    row_lo = x.obj.row_partition[rank+1]
    row_hi = x.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1
    nglobal = length(x)

    # Extract local data
    local_view = PETSc.unsafe_localarray(x.obj.v; read=true)
    local_data = try
        copy(local_view)
    finally
        Base.finalize(local_view)
    end

    # Prepare for Allgatherv
    counts = [x.obj.row_partition[i+2] - x.obj.row_partition[i+1] for i in 0:nranks-1]
    displs = [x.obj.row_partition[i+1] - 1 for i in 0:nranks-1]  # 0-based displacements

    # Gather to all ranks
    result = Vector{T}(undef, nglobal)
    MPI.Allgatherv!(local_data, MPI.VBuffer(result, counts, displs), comm)

    return result
end

# -----------------------------------------------------------------------------
# Vector Indexing
# -----------------------------------------------------------------------------

"""
    own_row(v::Vec{T}) -> UnitRange{Int}

**MPI Non-Collective**

Return the range of indices owned by the current rank for vector v.

# Example
```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
range = own_row(v)  # e.g., 1:2 on rank 0
```
"""
own_row(v::DRef{_Vec{T,Prefix}}) where {T,Prefix} = v.obj.row_partition[MPI.Comm_rank(MPI.COMM_WORLD)+1]:(v.obj.row_partition[MPI.Comm_rank(MPI.COMM_WORLD)+2]-1)

"""
    Base.getindex(v::Vec{T}, i::Int) -> T

**MPI Non-Collective**

Get the value at index i from a distributed vector.

The index i must be wholly contained in the current rank's ownership range.
If not, the function will abort with an error message and stack trace.

# Example
```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
# On rank that owns index 2:
val = v[2]  # Returns 2.0
```
"""
function Base.getindex(v::DRef{_Vec{T,Prefix}}, i::Int) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Get local range
    lo = v.obj.row_partition[rank+1]
    hi = v.obj.row_partition[rank+2] - 1

    # Check that index is in local range (non-collective check)
    if !(lo <= i <= hi)
        SafeMPI.mpierror("Index $i must be in rank $rank's ownership range [$lo, $hi]", true)
    end

    # Use unsafe_localarray to get read access to local data
    local_view = PETSc.unsafe_localarray(v.obj.v; read=true)
    try
        # Convert global index to local index
        local_idx = i - lo + 1
        return local_view[local_idx]
    finally
        Base.finalize(local_view)
    end
end

"""
    Base.getindex(v::Vec{T}, range::UnitRange{Int}) -> Vector{T}

**MPI Non-Collective**

Extract a contiguous range of values from a distributed vector.

The range must be wholly contained in the current rank's ownership range.
If not, the function will abort with an error message and stack trace.

# Example
```julia
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
# On rank that owns indices 2:3:
vals = v[2:3]  # Returns [2.0, 3.0]
```
"""
function Base.getindex(v::DRef{_Vec{T,Prefix}}, range::UnitRange{Int}) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Get local range
    lo = v.obj.row_partition[rank+1]
    hi = v.obj.row_partition[rank+2] - 1

    # Check that range is in local range (non-collective check)
    range_lo = first(range)
    range_hi = last(range)
    if !(lo <= range_lo && range_hi <= hi)
        SafeMPI.mpierror("Range $range must be wholly contained in rank $rank's ownership range [$lo, $hi]", true)
    end

    # Use unsafe_localarray to get read access to local data
    local_view = PETSc.unsafe_localarray(v.obj.v; read=true)
    try
        # Convert global indices to local indices
        local_start = range_lo - lo + 1
        local_end = range_hi - lo + 1
        return copy(local_view[local_start:local_end])
    finally
        Base.finalize(local_view)
    end
end

"""
    Base.show(io::IO, x::Vec{T})

**MPI Collective**

Display a distributed PETSc Vec by converting to Julia Vector and showing it.

All ranks must call it and will display the same vector.
To print only on rank 0, use: `println(io0(), v)`
"""
Base.show(io::IO, x::Vec{T}) where T = show(io, Vector(x))

"""
    Base.show(io::IO, mime::MIME, x::Vec{T})

**MPI Collective**

Display a distributed PETSc Vec with a specific MIME type by converting to Julia Vector.

All ranks must call it and will display the same vector.
To print only on rank 0, use: `show(io0(), mime, v)`
"""
Base.show(io::IO, mime::MIME, x::Vec{T}) where T = show(io, mime, Vector(x))
