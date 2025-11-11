# -----------------------------------------------------------------------------
# Matrix Pooling Infrastructure
# -----------------------------------------------------------------------------

"""
    PooledMat{T}

Stores a pooled PETSc matrix along with its partition information for reuse.
"""
struct PooledMat{T}
    mat::PETSc.Mat{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    fingerprint::Vector{UInt8}
end

"""
    ENABLE_MAT_POOL

Global flag to enable/disable matrix pooling. Set to `false` to disable pooling.
"""
const ENABLE_MAT_POOL = Ref{Bool}(true)

# Matrix pools: separate pools per PetscScalar type
# Non-product pool: Dict{(M, N, prefix) => Vector{PooledMat{T}}}
# Product pool: Dict{(product_type, hash1, hash2) => Vector{PooledMat{T}}}
PETSc.@for_libpetsc begin
    const $(Symbol(:MAT_POOL_NONPRODUCT_, PetscScalar)) = Dict{Tuple{Int,Int,String}, Vector{PooledMat{$PetscScalar}}}()
    const $(Symbol(:MAT_POOL_PRODUCT_, PetscScalar)) = Dict{Tuple{Cint,Vector{UInt8},Vector{UInt8}}, Vector{PooledMat{$PetscScalar}}}()
end

# -----------------------------------------------------------------------------
# Matrix Construction
# -----------------------------------------------------------------------------

"""
    Mat_uniform(A::Matrix{T}; row_partition=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)), col_partition=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)), prefix="") -> DRef{Mat{T}}

Create a distributed PETSc matrix from a Julia matrix, asserting uniform distribution across ranks (on MPI.COMM_WORLD).

- `A::Matrix{T}` must be identical on all ranks (`mpi_uniform`).
- `row_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
- `col_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start column (1-indexed) for rank i-1.
- `prefix` is an optional string prefix for MatSetOptionsPrefix() to set matrix-specific command-line options.
- Returns a DRef that will destroy the PETSc Mat collectively when all ranks release their reference.
"""
function Mat_uniform(A::Matrix{T};
                     row_partition::Vector{Int}=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)),
                     col_partition::Vector{Int}=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)),
                     prefix::String="") where T
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank   = MPI.Comm_rank(MPI.COMM_WORLD)

    # Preconditions - coalesced into single MPI synchronization
    row_partition_valid = length(row_partition) == nranks + 1 &&
                          row_partition[1] == 1 &&
                          row_partition[end] == size(A, 1) + 1 &&
                          all(r -> row_partition[r] <= row_partition[r+1], 1:nranks)
    col_partition_valid = length(col_partition) == nranks + 1 &&
                          col_partition[1] == 1 &&
                          col_partition[end] == size(A, 2) + 1 &&
                          all(r -> col_partition[r] <= col_partition[r+1], 1:nranks)
    @mpiassert SafeMPI.mpi_uniform(A) && row_partition_valid && col_partition_valid "Mat_uniform requires A to be mpi_uniform across all ranks; row_partition and col_partition must each have length nranks+1, start at 1, end at M+1/N+1 respectively, and be strictly increasing"

    # Local sizes
    row_lo = row_partition[rank+1]
    row_hi = row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1
    nglobal_rows = size(A, 1)

    col_lo = col_partition[rank+1]
    col_hi = col_partition[rank+2] - 1
    nlocal_cols = col_hi - col_lo + 1
    nglobal_cols = size(A, 2)

    # Create distributed PETSc Mat (no finalizer; collective destroy via DRef)
    # Pass partitions to enable pool lookup
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, prefix, row_partition, col_partition)

    # Set values from the local portion of A
    # For each row in the local row range, set all column values
    for i in row_lo:row_hi
        row_idx = i
        col_indices = collect(1:nglobal_cols)
        row_values = Vector{T}(A[i, :])
        _mat_setvalues!(petsc_mat, [row_idx], col_indices, row_values, PETSc.INSERT_VALUES)
    end

    # Assemble the matrix
    PETSc.assemble(petsc_mat)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Mat{T}(petsc_mat, row_partition, col_partition, prefix)
    return SafeMPI.DRef(obj)
end

"""
    Mat_sum(A::SparseMatrixCSC{T}; row_partition=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)), col_partition=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)), prefix="", own_rank_only=false) -> DRef{Mat{T}}

Create a distributed PETSc matrix by summing sparse matrices across ranks (on MPI.COMM_WORLD).

- `A::SparseMatrixCSC{T}` can differ across ranks; nonzero entries are summed across all ranks.
- `row_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
- `col_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start column (1-indexed) for rank i-1.
- `prefix` is an optional string prefix for MatSetOptionsPrefix() to set matrix-specific command-line options.
- `own_rank_only::Bool` (default=false): if true, asserts that all nonzero entries fall within this rank's row partition.
- Returns a DRef that will destroy the PETSc Mat collectively when all ranks release their reference.

Uses MatSetValues with ADD_VALUES mode to sum contributions from all ranks.
"""
function Mat_sum(A::SparseMatrixCSC{T};
                 row_partition::Vector{Int}=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)),
                 col_partition::Vector{Int}=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)),
                 prefix::String="",
                 own_rank_only::Bool=false) where T
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank   = MPI.Comm_rank(MPI.COMM_WORLD)

    # Preconditions - coalesced into single MPI synchronization
    row_partition_valid = length(row_partition) == nranks + 1 &&
                          row_partition[1] == 1 &&
                          row_partition[end] == size(A, 1) + 1 &&
                          all(r -> row_partition[r] <= row_partition[r+1], 1:nranks)
    col_partition_valid = length(col_partition) == nranks + 1 &&
                          col_partition[1] == 1 &&
                          col_partition[end] == size(A, 2) + 1 &&
                          all(r -> col_partition[r] <= col_partition[r+1], 1:nranks)

    # If own_rank_only, validate all nonzero entries fall within this rank's row partition
    own_rank_ok = true
    if own_rank_only
        row_lo = row_partition[rank+1]
        row_hi = row_partition[rank+2] - 1
        rows = rowvals(A)
        @inbounds for col in 1:size(A, 2)
            for idx in nzrange(A, col)
                row = rows[idx]
                if !(row_lo <= row <= row_hi)
                    own_rank_ok = false
                    break
                end
            end
            if !own_rank_ok; break; end
        end
    end

    @mpiassert row_partition_valid && col_partition_valid && own_rank_ok "row_partition and col_partition must each have length nranks+1, start at 1, end at M+1/N+1 respectively, and be strictly increasing; if own_rank_only=true, all nonzeros must fall within this rank's row partition"

    # Local sizes
    row_lo = row_partition[rank+1]
    row_hi = row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1
    nglobal_rows = size(A, 1)

    col_lo = col_partition[rank+1]
    col_hi = col_partition[rank+2] - 1
    nlocal_cols = col_hi - col_lo + 1
    nglobal_cols = size(A, 2)

    # Create distributed PETSc Mat (no finalizer; collective destroy via DRef)
    # Pass partitions to enable pool lookup
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, prefix, row_partition, col_partition)

    # Set values from sparse matrix using ADD_VALUES mode
    rows_csc = rowvals(A)
    vals_csc = nonzeros(A)

    for col in 1:size(A, 2)
        for idx in nzrange(A, col)
            row = rows_csc[idx]
            val = vals_csc[idx]
            _mat_setvalues!(petsc_mat, [row], [col], [val], PETSc.ADD_VALUES)
        end
    end

    # Assemble the matrix
    PETSc.assemble(petsc_mat)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Mat{T}(petsc_mat, row_partition, col_partition, prefix)
    return SafeMPI.DRef(obj)
end

# Create a distributed PETSc Mat for a given element type T by dispatching to the
# underlying PETSc scalar variant via PETSc.@for_libpetsc
# This function checks the pool first before creating a new matrix
function _mat_create_mpi_for_T(::Type{T}, nlocal_rows::Integer, nlocal_cols::Integer,
                               nglobal_rows::Integer, nglobal_cols::Integer, prefix::String="",
                               row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[]) where {T}
    return _mat_create_mpi_impl(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, prefix, row_partition, col_partition)
end

PETSc.@for_libpetsc begin
    function _mat_create_mpi_impl(::Type{$PetscScalar}, nlocal_rows::Integer, nlocal_cols::Integer,
                                  nglobal_rows::Integer, nglobal_cols::Integer, prefix::String="",
                                  row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[])
        # Try to get from pool first (only if partitions are provided for matching)
        if ENABLE_MAT_POOL[] && !isempty(row_partition) && !isempty(col_partition)
            pool = $(Symbol(:MAT_POOL_NONPRODUCT_, PetscScalar))
            pool_key = (Int(nglobal_rows), Int(nglobal_cols), prefix)
            if haskey(pool, pool_key)
                pool_list = pool[pool_key]
                # Scan for matching row_partition and col_partition
                for (i, pooled) in enumerate(pool_list)
                    if pooled.row_partition == row_partition && pooled.col_partition == col_partition
                        # Found match - remove from pool and return
                        deleteat!(pool_list, i)
                        if isempty(pool_list)
                            delete!(pool, pool_key)
                        end
                        return pooled.mat
                    end
                end
            end
        end

        # Pool miss or pooling disabled - create new matrix
        mat = PETSc.Mat{$PetscScalar}(C_NULL)
        PETSc.@chk ccall((:MatCreate, $libpetsc), PETSc.PetscErrorCode,
                         (MPI.MPI_Comm, Ptr{CMat}), MPI.COMM_WORLD, mat)
        PETSc.@chk ccall((:MatSetSizes, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, $PetscInt, $PetscInt, $PetscInt, $PetscInt),
                         mat, $PetscInt(nlocal_rows), $PetscInt(nlocal_cols),
                         $PetscInt(nglobal_rows), $PetscInt(nglobal_cols))
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CMat, Cstring), mat, prefix)
        end
        PETSc.@chk ccall((:MatSetFromOptions, $libpetsc), PETSc.PetscErrorCode,
                         (CMat,), mat)
        PETSc.@chk ccall((:MatSetUp, $libpetsc), PETSc.PetscErrorCode,
                         (CMat,), mat)
        return mat
    end

    function _destroy_petsc_mat!(m::PETSc.AbstractMat{$PetscScalar})
        PETSc.finalized($petsclib) || begin
            PETSc.@chk ccall((:MatDestroy, $libpetsc), PETSc.PetscErrorCode,
                             (Ptr{CMat},), m)
            m.ptr = C_NULL
        end
        return nothing
    end

    function _mat_setvalues_impl!(mat::PETSc.Mat{$PetscScalar}, row_indices::Vector{Int},
                                  col_indices::Vector{Int}, values::Vector{$PetscScalar},
                                  mode::PETSc.InsertMode)
        # Convert 1-based Julia indices to 0-based PETSc indices
        rows_c = $PetscInt.(row_indices .- 1)
        cols_c = $PetscInt.(col_indices .- 1)
        nrows = length(row_indices)
        ncols = length(col_indices)

        PETSc.@chk ccall((:MatSetValues, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, $PetscInt, Ptr{$PetscInt}, $PetscInt, Ptr{$PetscInt},
                          Ptr{$PetscScalar}, PETSc.InsertMode),
                         mat, $PetscInt(nrows), rows_c, $PetscInt(ncols), cols_c,
                         values, mode)
        return nothing
    end
end

# Generic wrapper for _mat_setvalues!
function _mat_setvalues!(mat::PETSc.Mat{T}, row_indices::Vector{Int}, col_indices::Vector{Int},
                        values::Vector{T}, mode::PETSc.InsertMode) where {T}
    return _mat_setvalues_impl!(mat, row_indices, col_indices, values, mode)
end

# -----------------------------------------------------------------------------
# Matrix Pool Helper Functions
# -----------------------------------------------------------------------------

# Return a matrix to the pool for reuse
function _return_mat_to_pool!(m::PETSc.Mat{T}, row_partition::Vector{Int},
                              col_partition::Vector{Int}, prefix::String,
                              product_type::Cint, product_args::Vector{Vector{UInt8}}) where {T}
    return _return_mat_to_pool_impl!(m, row_partition, col_partition, prefix, product_type, product_args)
end

# Generic wrapper for product pool lookup
function _try_get_from_product_pool(product_type::Cint, hash_A::Vector{UInt8}, hash_B::Vector{UInt8},
                                    row_partition::Vector{Int}, col_partition::Vector{Int}, ::Type{T}) where {T}
    return _try_get_from_product_pool_impl(product_type, hash_A, hash_B, row_partition, col_partition, T)
end

"""
    _try_get_from_nonproduct_pool_by_fingerprint(row_partition, col_partition, prefix, fingerprint, ::Type{T})

Try to retrieve a non-product pooled Mat matching the given partitions, prefix, and structure fingerprint.
Returns a pooled PETSc.Mat{T} or nothing on miss.
"""
function _try_get_from_nonproduct_pool_by_fingerprint(row_partition::Vector{Int},
                                                      col_partition::Vector{Int},
                                                      prefix::String,
                                                      fingerprint::Vector{UInt8},
                                                      ::Type{T}) where {T}
    return _try_get_from_nonproduct_pool_by_fingerprint_impl(row_partition, col_partition, prefix, fingerprint, T)
end

PETSc.@for_libpetsc begin
    # Implementation for specific PetscScalar type
    function _try_get_from_product_pool_impl(product_type::Cint, hash_A::Vector{UInt8}, hash_B::Vector{UInt8},
                                             row_partition::Vector{Int}, col_partition::Vector{Int}, ::Type{$PetscScalar})
        if !ENABLE_MAT_POOL[] || isempty(row_partition) || isempty(col_partition) ||
           isempty(hash_A) || isempty(hash_B)
            return nothing
        end

        pool = $(Symbol(:MAT_POOL_PRODUCT_, PetscScalar))
        pool_key = (product_type, hash_A, hash_B)
        if !haskey(pool, pool_key)
            return nothing
        end

        pool_list = pool[pool_key]
        for (i, pooled) in enumerate(pool_list)
            if pooled.row_partition == row_partition && pooled.col_partition == col_partition
                # Found match - remove from pool and return
                deleteat!(pool_list, i)
                if isempty(pool_list)
                    delete!(pool, pool_key)
                end
                return pooled.mat
            end
        end
        return nothing
    end

    function _try_get_from_nonproduct_pool_by_fingerprint_impl(row_partition::Vector{Int},
                                                               col_partition::Vector{Int},
                                                               prefix::String,
                                                               fingerprint::Vector{UInt8},
                                                               ::Type{$PetscScalar})
        if !ENABLE_MAT_POOL[] || isempty(row_partition) || isempty(col_partition)
            return nothing
        end

        M = row_partition[end] - 1
        N = col_partition[end] - 1
        pool = $(Symbol(:MAT_POOL_NONPRODUCT_, PetscScalar))
        pool_key = (Int(M), Int(N), prefix)
        if !haskey(pool, pool_key)
            return nothing
        end

        pool_list = pool[pool_key]
        for (i, pooled) in enumerate(pool_list)
            if pooled.row_partition == row_partition && pooled.col_partition == col_partition && pooled.fingerprint == fingerprint
                deleteat!(pool_list, i)
                if isempty(pool_list)
                    delete!(pool, pool_key)
                end
                return pooled.mat
            end
        end
        return nothing
    end

    function _return_mat_to_pool_impl!(m::PETSc.Mat{$PetscScalar}, row_partition::Vector{Int},
                                       col_partition::Vector{Int}, prefix::String,
                                       product_type::Cint, product_args::Vector{Vector{UInt8}})
        # Don't pool if PETSc is finalizing
        if PETSc.finalized($petsclib)
            return nothing
        end

        # Zero out matrix contents before returning to pool
        PETSc.@chk ccall((:MatZeroEntries, $libpetsc), PETSc.PetscErrorCode,
                         (CMat,), m)

        # Compute fingerprint for this matrix (structure + prefix + partitions)
        fp = _matrix_fingerprint(m, row_partition, col_partition, prefix)

        # Get matrix dimensions
        M_ref = Ref{$PetscInt}(0)
        N_ref = Ref{$PetscInt}(0)
        PETSc.@chk ccall((:MatGetSize, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, Ptr{$PetscInt}, Ptr{$PetscInt}), m, M_ref, N_ref)
        M = Int(M_ref[])
        N = Int(N_ref[])

        # Determine which pool to use based on product_type
        if product_type == MATPRODUCT_UNSPECIFIED
            # Non-product matrix pool
            pool = $(Symbol(:MAT_POOL_NONPRODUCT_, PetscScalar))
            pool_key = (M, N, prefix)
            if !haskey(pool, pool_key)
                pool[pool_key] = PooledMat{$PetscScalar}[]
            end
            push!(pool[pool_key], PooledMat{$PetscScalar}(m, row_partition, col_partition, fp))
        else
            # Product matrix pool
            if length(product_args) >= 2
                pool = $(Symbol(:MAT_POOL_PRODUCT_, PetscScalar))
                hash1 = product_args[1]
                hash2 = product_args[2]
                pool_key = (product_type, hash1, hash2)
                if !haskey(pool, pool_key)
                    pool[pool_key] = PooledMat{$PetscScalar}[]
                end
                push!(pool[pool_key], PooledMat{$PetscScalar}(m, row_partition, col_partition, fp))
            end
        end

        return nothing
    end

    function _mat_zero_entries!(m::PETSc.Mat{$PetscScalar})
        PETSc.@chk ccall((:MatZeroEntries, $libpetsc), PETSc.PetscErrorCode,
                         (CMat,), m)
        return nothing
    end
end

# Element type and shape for internal _Mat and DRef-wrapped _Mat
Base.eltype(::_Mat{T}) where {T} = T
Base.size(m::_Mat) = (m.row_partition[end] - 1, m.col_partition[end] - 1)
Base.size(m::_Mat, d::Integer) = d == 1 ? (m.row_partition[end] - 1) : (d == 2 ? (m.col_partition[end] - 1) : 1)

Base.eltype(r::SafeMPI.DRef{<:_Mat}) = Base.eltype(r.obj)
Base.size(r::SafeMPI.DRef{<:_Mat}) = Base.size(r.obj)
Base.size(r::SafeMPI.DRef{<:_Mat}, d::Integer) = Base.size(r.obj, d)

# Adjoint support for Mat
Base.adjoint(A::Mat{T}) where {T} = LinearAlgebra.Adjoint(A)

# Materialize an Adjoint{Mat} into a new Mat (creates A^T)
function Mat(adj::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) where {T}
    A = parent(adj)::Mat{T}
    # Create B = A^T via PETSc (MAT_INITIAL_MATRIX) and preserve prefix
    # Pass swapped partitions to enable pool lookup
    B_petsc = _mat_transpose(A.obj.A, A.obj.prefix, A.obj.col_partition, A.obj.row_partition)
    # Swap row/col partitions for the transpose
    obj = _Mat{T}(B_petsc, A.obj.col_partition, A.obj.row_partition, A.obj.prefix)
    return SafeMPI.DRef(obj)
end

# In-place transpose: B = A^T (reuses pre-allocated B)
function LinearAlgebra.transpose!(B::Mat{T}, A::Mat{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 1) == size(B, 2) && size(A, 2) == size(B, 1) &&
                A.obj.row_partition == B.obj.col_partition &&
                A.obj.col_partition == B.obj.row_partition) "Matrices must have the same prefix, transpose dimensions must match (A is $(size(A)), B must be $(size(A, 2))×$(size(A, 1)), got $(size(B))), A's row partition must match B's column partition, and A's column partition must match B's row partition"

    # Perform in-place transpose using PETSc reuse path. The caller must
    # ensure B was originally created by MatTranspose(A, MAT_INITIAL_MATRIX)
    # (or has a valid precursor) with matching partitions.
    _mat_transpose!(B.obj.A, A.obj.A)

    return B
end

# Matrix-vector multiplication: y = A * x
function Base.:*(A::Mat{T}, x::Vec{T}) where {T}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(x)[1]
    @mpiassert n == vec_length && A.obj.col_partition == x.obj.row_partition && A.obj.prefix == x.obj.prefix "Matrix columns must match vector length (A: $(m)×$(n), x: $(vec_length)), column partition of A must match row partition of x, and A and x must have the same prefix"

    # Create result vector with A's row partition
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    y_petsc = _vec_create_mpi_for_T(T, nlocal, m, x.obj.prefix, A.obj.row_partition)

    # Perform matrix-vector multiplication using PETSc
    _mat_mult_vec!(y_petsc, A.obj.A, x.obj.v)

    PETSc.assemble(y_petsc)

    # Wrap in DRef
    obj = _Vec{T}(y_petsc, A.obj.row_partition, x.obj.prefix)
    return SafeMPI.DRef(obj)
end

# In-place matrix-vector multiplication: y = A * x (reuses pre-allocated y)
function LinearAlgebra.mul!(y::Vec{T}, A::Mat{T}, x::Vec{T}) where {T}
    # Check dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(A)
    y_length = size(y)[1]
    x_length = size(x)[1]
    @mpiassert (m == y_length && n == x_length &&
                A.obj.row_partition == y.obj.row_partition &&
                A.obj.col_partition == x.obj.row_partition &&
                A.obj.prefix == x.obj.prefix == y.obj.prefix) "Output vector y must have length matching matrix rows (A: $(m)×$(n), y: $(y_length)), input vector x must have length matching matrix columns (x: $(x_length)), matrix row partition must match output vector partition, matrix column partition must match input vector partition, and all objects must have the same prefix"

    # Perform matrix-vector multiplication using PETSc (reuses y)
    _mat_mult_vec!(y.obj.v, A.obj.A, x.obj.v)

    PETSc.assemble(y.obj.v)

    return y
end

# PETSc matrix-vector multiplication wrapper
PETSc.@for_libpetsc begin
    function _mat_mult_vec!(y::PETSc.Vec{$PetscScalar}, A::PETSc.Mat{$PetscScalar}, x::PETSc.Vec{$PetscScalar})
        PETSc.@chk ccall((:MatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CVec, PETSc.CVec),
                         A, x, y)
        return nothing
    end
end

# PETSc MatAXPY: Y = a*X + Y with MatStructure indicating relationship of X's pattern to Y's
PETSc.@for_libpetsc begin
    function _mat_axpy!(Y::PETSc.Mat{$PetscScalar}, a::$PetscScalar, X::PETSc.Mat{$PetscScalar}, structure::Cint)
        PETSc.@chk ccall((:MatAXPY, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, $PetscScalar, PETSc.CMat, Cint),
                         Y, a, X, structure)
        return nothing
    end
end

# -----------------------------------------------------------------------------
# Matrix addition and subtraction (non-product ops)
# - Compute union sparsity, try to reuse from non-product pool by fingerprint
# - Fall back to dynamic pattern with MatAXPY if no pooled match
# -----------------------------------------------------------------------------

"""
    Base.:+(A::Mat{T}, B::Mat{T}) -> Mat{T}

Add two distributed PETSc matrices. Requires identical sizes, partitions, and prefix.
Attempts to reuse a pooled matrix with matching union structure via fingerprint.
"""
function Base.:+(A::Mat{T}, B::Mat{T}) where {T}
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) &&
                A.obj.row_partition == B.obj.row_partition &&
                A.obj.col_partition == B.obj.col_partition) "Matrix addition requires same prefix, identical sizes, and identical partitions"

    # Compute union local structure and final fingerprint
    nloc, iaU, jaU = _local_union_structure(A.obj.A, B.obj.A)
    local_hash = _local_structure_hash(nloc, iaU, jaU)
    fp = _structure_fingerprint(local_hash, A.obj.row_partition, A.obj.col_partition, A.obj.prefix)

    # Try to reuse from pool by fingerprint
    Cmat = _try_get_from_nonproduct_pool_by_fingerprint(A.obj.row_partition, A.obj.col_partition,
                                                        A.obj.prefix, fp, T)
    # Create if miss
    if Cmat === nothing
        nr = MPI.Comm_rank(MPI.COMM_WORLD)
        nlocal_rows = A.obj.row_partition[nr+2] - A.obj.row_partition[nr+1]
        nlocal_cols = A.obj.col_partition[nr+2] - A.obj.col_partition[nr+1]
        Cmat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols,
                                     size(A,1), size(A,2), A.obj.prefix,
                                     A.obj.row_partition, A.obj.col_partition)
        _mat_zero_entries!(Cmat)
        structure = DIFFERENT_NONZERO_PATTERN
    else
        structure = SUBSET_NONZERO_PATTERN
    end

    # Accumulate C = A + B
    _mat_axpy!(Cmat, one(T), A.obj.A, structure)
    _mat_axpy!(Cmat, one(T), B.obj.A, structure)
    PETSc.assemble(Cmat)

    obj = _Mat{T}(Cmat, A.obj.row_partition, A.obj.col_partition, A.obj.prefix)
    return SafeMPI.DRef(obj)
end

"""
    Base.:-(A::Mat{T}, B::Mat{T}) -> Mat{T}

Subtract two distributed PETSc matrices. Requires identical sizes, partitions, and prefix.
Attempts to reuse a pooled matrix with matching union structure via fingerprint.
"""
function Base.:-(A::Mat{T}, B::Mat{T}) where {T}
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) &&
                A.obj.row_partition == B.obj.row_partition &&
                A.obj.col_partition == B.obj.col_partition) "Matrix subtraction requires same prefix, identical sizes, and identical partitions"

    # Compute union local structure and final fingerprint
    nloc, iaU, jaU = _local_union_structure(A.obj.A, B.obj.A)
    local_hash = _local_structure_hash(nloc, iaU, jaU)
    fp = _structure_fingerprint(local_hash, A.obj.row_partition, A.obj.col_partition, A.obj.prefix)

    Cmat = _try_get_from_nonproduct_pool_by_fingerprint(A.obj.row_partition, A.obj.col_partition,
                                                        A.obj.prefix, fp, T)
    if Cmat === nothing
        nr = MPI.Comm_rank(MPI.COMM_WORLD)
        nlocal_rows = A.obj.row_partition[nr+2] - A.obj.row_partition[nr+1]
        nlocal_cols = A.obj.col_partition[nr+2] - A.obj.col_partition[nr+1]
        Cmat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols,
                                     size(A,1), size(A,2), A.obj.prefix,
                                     A.obj.row_partition, A.obj.col_partition)
        _mat_zero_entries!(Cmat)
        structure = DIFFERENT_NONZERO_PATTERN
    else
        structure = SUBSET_NONZERO_PATTERN
    end

    _mat_axpy!(Cmat, one(T), A.obj.A, structure)
    _mat_axpy!(Cmat, -one(T), B.obj.A, structure)
    PETSc.assemble(Cmat)

    obj = _Mat{T}(Cmat, A.obj.row_partition, A.obj.col_partition, A.obj.prefix)
    return SafeMPI.DRef(obj)
end

"""
    _local_union_structure(A::PETSc.Mat{T}, B::PETSc.Mat{T}) -> (nloc, iaU, jaU)

Compute the union CSR structure for the local rows of A and B on this rank.
Returns the number of local rows, row-pointer iaU (length nloc+1) and 0-based column
indices jaU for hashing compatibility.
If structure cannot be queried (e.g. dense types), returns zero-length arrays with nloc=0.
"""
function _local_union_structure(A::PETSc.Mat{T}, B::PETSc.Mat{T}) where {T}
    try
        # Query A
        nA, iaA, jaA, iaA_ptr, jaA_ptr = _mat_get_rowij(A)
        # Query B
        nB, iaB, jaB, iaB_ptr, jaB_ptr = _mat_get_rowij(B)
        @assert nA == nB
        nloc = Int(nA)
        iaU = Vector{Int64}(undef, nloc + 1)
        iaU[1] = 0
        jaU = Int64[]
        @inbounds for i in 1:nloc
            # Convert C-style offsets to Julia ranges
            a_s = Int(iaA[i]) + 1; a_e = Int(iaA[i+1])
            b_s = Int(iaB[i]) + 1; b_e = Int(iaB[i+1])
            colsA = a_s <= a_e ? jaA[a_s:a_e] : Int[]
            colsB = b_s <= b_e ? jaB[b_s:b_e] : Int[]
            # Merge union of sorted arrays of 0-based columns
            u = Vector{Int64}()
            sizehint!(u, length(colsA) + length(colsB))
            iax = 1; ibx = 1
            while iax <= length(colsA) && ibx <= length(colsB)
                a = Int64(colsA[iax]); b = Int64(colsB[ibx])
                if a == b
                    push!(u, a); iax += 1; ibx += 1
                elseif a < b
                    push!(u, a); iax += 1
                else
                    push!(u, b); ibx += 1
                end
            end
            while iax <= length(colsA); push!(u, Int64(colsA[iax])); iax += 1; end
            while ibx <= length(colsB); push!(u, Int64(colsB[ibx])); ibx += 1; end
            append!(jaU, u)
            iaU[i+1] = length(jaU)
        end

        # Restore
        _mat_restore_rowij!(A, nA, iaA_ptr, jaA_ptr)
        _mat_restore_rowij!(B, nB, iaB_ptr, jaB_ptr)
        return (length(iaU) - 1, iaU, jaU)
    catch
        return (0, Int64[0], Int64[])
    end
end


# Helper function to extract owned rows from a PETSc Mat to Julia SparseMatrixCSC
function _mat_to_local_sparse(A::Mat{T}) where {T}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    m, n = size(A)
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    # Initialize arrays for CSC format
    I = Int[]
    J = Int[]
    V = T[]

    # Extract owned rows
    for local_row in 1:nlocal_rows
        global_row = row_lo + local_row - 1
        # Get values for this row
        for j in 1:n
            # Use MatGetValues to get the entry
            val = _mat_getvalue(A.obj.A, global_row, j)
            if abs(val) > 0  # Only store nonzeros
                push!(I, global_row)
                push!(J, j)
                push!(V, val)
            end
        end
    end

    # Create sparse matrix of full global size
    return sparse(I, J, V, m, n)
end

# Helper to get a single matrix value
PETSc.@for_libpetsc begin
    function _mat_getvalue(mat::PETSc.Mat{$PetscScalar}, row::Int, col::Int)
        row_c = $PetscInt(row - 1)  # Convert to 0-based
        col_c = $PetscInt(col - 1)
        val = Ref{$PetscScalar}(0)
        PETSc.@chk ccall((:MatGetValues, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, $PetscInt, Ptr{$PetscInt}, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}),
                         mat, $PetscInt(1), Ref(row_c), $PetscInt(1), Ref(col_c), val)
        return val[]
    end
end

"""
    Base.cat(As::Mat{T}...; dims) -> Mat{T}

Concatenate distributed PETSc matrices along dimension `dims`.

All input matrices must:
- Have the same element type `T`
- Have the same prefix
- Have compatible sizes and partitions for the concatenation dimension

The concatenation is performed by:
1. Each rank extracts its owned rows from each input matrix as a Julia sparse matrix
2. Standard Julia `cat` is applied locally on each rank
3. The results are summed across ranks using `Mat_sum`

# Examples
```julia
C = cat(A, B; dims=1)  # Vertical concatenation (vcat)
D = cat(A, B; dims=2)  # Horizontal concatenation (hcat)
```
"""
function Base.cat(As::Mat{T}...; dims) where {T}
    n = length(As)
    if n == 0
        throw(ArgumentError("cat requires at least one matrix"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate all matrices have same type T and prefix - collect conditions first
    first_prefix = As[1].obj.prefix
    all_same_prefix = all(As[k].obj.prefix == first_prefix for k in 2:n)

    # Validate partition compatibility based on dims - collect conditions first
    if dims == 1  # Vertical concatenation
        # All matrices must have same column partition and number of columns
        first_col_partition = As[1].obj.col_partition
        first_ncols = size(As[1], 2)
        all_same_col_partition = all(As[k].obj.col_partition == first_col_partition for k in 2:n)
        all_same_ncols = all(size(As[k], 2) == first_ncols for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        @mpiassert (all_same_prefix && all_same_col_partition && all_same_ncols) "For dims=1: all matrices must have the same prefix, column partition, and number of columns"
    elseif dims == 2  # Horizontal concatenation
        # All matrices must have same row partition and number of rows
        first_row_partition = As[1].obj.row_partition
        first_nrows = size(As[1], 1)
        all_same_row_partition = all(As[k].obj.row_partition == first_row_partition for k in 2:n)
        all_same_nrows = all(size(As[k], 1) == first_nrows for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        @mpiassert (all_same_prefix && all_same_row_partition && all_same_nrows) "For dims=2: all matrices must have the same prefix, row partition, and number of rows"
    else
        throw(ArgumentError("dims must be 1 or 2 for matrix concatenation"))
    end

    # Extract local sparse matrices (owned rows only)
    local_sparse = [_mat_to_local_sparse(A) for A in As]

    # Perform local cat operation
    local_result = cat(local_sparse...; dims=dims)

    # Determine result partitions
    if dims == 1  # Vertical: rows increase, cols stay same
        # New row partition: concatenate the individual row counts
        total_rows = sum(size(A, 1) for A in As)
        result_row_partition = zeros(Int, nranks + 1)
        result_row_partition[1] = 1
        for r in 1:nranks
            rows_in_rank = sum(As[k].obj.row_partition[r+1] - As[k].obj.row_partition[r] for k in 1:n)
            result_row_partition[r+1] = result_row_partition[r] + rows_in_rank
        end
        result_col_partition = As[1].obj.col_partition
    else  # dims == 2, Horizontal: cols increase, rows stay same
        # New column partition: concatenate the individual column counts
        total_cols = sum(size(A, 2) for A in As)
        result_col_partition = zeros(Int, nranks + 1)
        result_col_partition[1] = 1
        for r in 1:nranks
            cols_in_rank = sum(As[k].obj.col_partition[r+1] - As[k].obj.col_partition[r] for k in 1:n)
            result_col_partition[r+1] = result_col_partition[r] + cols_in_rank
        end
        result_row_partition = As[1].obj.row_partition
    end

    # Use Mat_sum to combine results across ranks
    return Mat_sum(local_result;
                   row_partition=result_row_partition,
                   col_partition=result_col_partition,
                   prefix=first_prefix,
                   own_rank_only=false)
end

"""
    Base.vcat(As::Mat{T}...) -> Mat{T}

Vertically concatenate distributed PETSc matrices.

Equivalent to `cat(As...; dims=1)`. All matrices must have the same number of columns
and the same column partition.
"""
Base.vcat(As::Mat{T}...) where {T} = cat(As...; dims=1)

"""
    Base.hcat(As::Mat{T}...) -> Mat{T}

Horizontally concatenate distributed PETSc matrices.

Equivalent to `cat(As...; dims=2)`. All matrices must have the same number of rows
and the same row partition.
"""
Base.hcat(As::Mat{T}...) where {T} = cat(As...; dims=2)

# Import blockdiag from SparseArrays
import SparseArrays: blockdiag

"""
    blockdiag(As::Mat{T}...) -> Mat{T}

Create a block diagonal matrix from distributed PETSc matrices.

The result is a matrix with the input matrices along the diagonal and zeros elsewhere.
All matrices must have the same prefix and element type.

# Example
```julia
# If A is m×n and B is p×q, then blockdiag(A, B) is (m+p)×(n+q)
C = blockdiag(A, B)
```
"""
function blockdiag(As::Mat{T}...) where {T}
    n = length(As)
    if n == 0
        throw(ArgumentError("blockdiag requires at least one matrix"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate all matrices have same type T and prefix - single @mpiassert (pulled out of loop)
    first_prefix = As[1].obj.prefix
    all_same_prefix = all(As[k].obj.prefix == first_prefix for k in 2:n)
    @mpiassert all_same_prefix "All matrices must have the same prefix"

    # Extract local sparse matrices (owned rows only)
    local_sparse = [_mat_to_local_sparse(A) for A in As]

    # Perform local blockdiag operation
    local_result = blockdiag(local_sparse...)

    # Compute result partitions
    # Row partition: sum of all row partitions
    total_rows = sum(size(A, 1) for A in As)
    result_row_partition = zeros(Int, nranks + 1)
    result_row_partition[1] = 1
    for r in 1:nranks
        rows_in_rank = sum(As[k].obj.row_partition[r+1] - As[k].obj.row_partition[r] for k in 1:n)
        result_row_partition[r+1] = result_row_partition[r] + rows_in_rank
    end

    # Column partition: sum of all column partitions
    total_cols = sum(size(A, 2) for A in As)
    result_col_partition = zeros(Int, nranks + 1)
    result_col_partition[1] = 1
    for r in 1:nranks
        cols_in_rank = sum(As[k].obj.col_partition[r+1] - As[k].obj.col_partition[r] for k in 1:n)
        result_col_partition[r+1] = result_col_partition[r] + cols_in_rank
    end

    # Use Mat_sum to combine results across ranks
    return Mat_sum(local_result;
                   row_partition=result_row_partition,
                   col_partition=result_col_partition,
                   prefix=first_prefix,
                   own_rank_only=false)
end

"""
    Base.:*(A::Mat{T}, B::Mat{T}) -> Mat{T}

Multiply two distributed PETSc matrices using PETSc's MatMatMult.

Both matrices must have the same element type `T` and the same prefix.
The number of columns in A must match the number of rows in B.

# Example
```julia
C = A * B  # Matrix multiplication
```
"""
function Base.:*(A::Mat{T}, B::Mat{T}) where {T}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs - single @mpiassert for efficiency
    # Check: same prefix, inner dimensions match, and inner partitions match
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 2) == size(B, 1) &&
                A.obj.col_partition == B.obj.row_partition) "Matrix multiplication requires same prefix, compatible dimensions (A cols must equal B rows), and matching inner partitions (A.col_partition must equal B.row_partition)"

    # Determine result partitions
    # Result has same row partition as A and same column partition as B
    result_row_partition = A.obj.row_partition
    result_col_partition = B.obj.col_partition

    # Use PETSc's MatMatMult via a small wrapper, passing fingerprints for pooling
    C_mat = _mat_mat_mult(A.obj.A, B.obj.A, A.obj.prefix,
                          result_row_partition, result_col_partition,
                          A.obj.fingerprint, B.obj.fingerprint)

    # Wrap in our _Mat type and return as DRef
    # Track that this is a product matrix with fingerprints of source matrices
    obj = _Mat{T}(C_mat, result_row_partition, result_col_partition, A.obj.prefix;
                  product_type=MATPRODUCT_AB,
                  product_args=[A.obj.fingerprint, B.obj.fingerprint])
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, B::Mat{T}) -> Mat{T}

Transpose-matrix multiplication using PETSc's MatTransposeMatMult.

Computes C = A' * B where A' is the transpose (adjoint for real matrices) of A.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, B::Mat{T}) where {T}
    A = parent(At)::Mat{T}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B is m×p, result is n×p
    # A' has row partition = A's col partition, col partition = A's row partition
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 1) == size(B, 1) &&
                A.obj.row_partition == B.obj.row_partition) "Transpose-matrix multiplication A'*B requires same prefix, compatible dimensions (A rows must equal B rows for A'*B), and matching partitions (A.row_partition must equal B.row_partition)"

    # Result partitions: rows from A's columns, columns from B's columns
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.col_partition

    # Use PETSc's MatTransposeMatMult, passing fingerprints for pooling
    C_mat = _mat_transpose_mat_mult(A.obj.A, B.obj.A, A.obj.prefix,
                                     result_row_partition, result_col_partition,
                                     A.obj.fingerprint, B.obj.fingerprint)

    # Wrap in our _Mat type and return as DRef
    obj = _Mat{T}(C_mat, result_row_partition, result_col_partition, A.obj.prefix;
                  product_type=MATPRODUCT_AtB,
                  product_args=[A.obj.fingerprint, B.obj.fingerprint])
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(A::Mat{T}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) -> Mat{T}

Matrix-transpose multiplication using PETSc's MatMatTransposeMult.

Computes C = A * B' where B' is the transpose (adjoint for real matrices) of B.
"""
function Base.:*(A::Mat{T}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) where {T}
    B = parent(Bt)::Mat{T}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A is m×n, B' is n×p (B is p×n), result is m×p
    # B' has row partition = B's col partition, col partition = B's row partition
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 2) == size(B, 2) &&
                A.obj.col_partition == B.obj.col_partition) "Matrix-transpose multiplication A*B' requires same prefix, compatible dimensions (A cols must equal B cols for A*B'), and matching partitions (A.col_partition must equal B.col_partition)"

    # Result partitions: rows from A's rows, columns from B's rows
    result_row_partition = A.obj.row_partition
    result_col_partition = B.obj.row_partition

    # Use PETSc's MatMatTransposeMult, passing fingerprints for pooling
    C_mat = _mat_mat_transpose_mult(A.obj.A, B.obj.A, A.obj.prefix,
                                     result_row_partition, result_col_partition,
                                     A.obj.fingerprint, B.obj.fingerprint)

    # Wrap in our _Mat type and return as DRef
    obj = _Mat{T}(C_mat, result_row_partition, result_col_partition, A.obj.prefix;
                  product_type=MATPRODUCT_ABt,
                  product_args=[A.obj.fingerprint, B.obj.fingerprint])
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) -> Mat{T}

Transpose-transpose multiplication: C = A' * B'.

Since PETSc does not have a direct AtBt product type, this is computed as
C = (B * A)' by materializing the transpose of the result.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) where {T}
    A = parent(At)::Mat{T}
    B = parent(Bt)::Mat{T}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B' is m×p (B is p×m), result is n×p
    @mpiassert (A.obj.prefix == B.obj.prefix &&
                size(A, 1) == size(B, 2) &&
                A.obj.row_partition == B.obj.col_partition) "Transpose-transpose multiplication A'*B' requires same prefix, compatible dimensions (A rows must equal B cols for A'*B'), and matching partitions (A.row_partition must equal B.col_partition)"

    # Compute as (B * A)' since PETSc doesn't have AtBt product type
    # First compute BA
    BA_mat = _mat_mat_mult(B.obj.A, A.obj.A, A.obj.prefix)

    # Then transpose it to get (BA)' = A'B'
    # BA has partitions (B.row_partition, A.col_partition), so transpose has swapped
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.row_partition
    C_mat = _mat_transpose(BA_mat, A.obj.prefix, result_row_partition, result_col_partition)

    # Result partitions: rows from A's columns, columns from B's rows

    # Wrap in our _Mat type - mark as UNSPECIFIED since no direct PETSc support
    # Store both fingerprints to indicate this depends on A and B
    obj = _Mat{T}(C_mat, result_row_partition, result_col_partition, A.obj.prefix;
                  product_type=MATPRODUCT_UNSPECIFIED,  # No AtBt in PETSc
                  product_args=[A.obj.fingerprint, B.obj.fingerprint])
    return SafeMPI.DRef(obj)
end

# In-place matrix-matrix multiplication: C = A * B (reuses pre-allocated C)
function LinearAlgebra.mul!(C::Mat{T}, A::Mat{T}, B::Mat{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    @mpiassert (A.obj.prefix == B.obj.prefix == C.obj.prefix &&
                size(A, 2) == size(B, 1) &&
                size(A, 1) == size(C, 1) && size(B, 2) == size(C, 2) &&
                A.obj.col_partition == B.obj.row_partition &&
                A.obj.row_partition == C.obj.row_partition &&
                B.obj.col_partition == C.obj.col_partition) "All matrices must have the same prefix, inner dimensions must match (A cols must equal B rows): A is $(size(A)), B is $(size(B)), output matrix C must have dimensions $(size(A, 1))×$(size(B, 2)) (got $(size(C))), matrix inner partitions must match (A.col_partition must equal B.row_partition), result row partition must match A's row partition, and result column partition must match B's column partition"

    # Perform in-place matrix-matrix multiplication using PETSc
    _mat_mat_mult!(C.obj.A, A.obj.A, B.obj.A)

    return C
end

# PETSc matrix-matrix multiplication wrapper
PETSc.@for_libpetsc begin
    function _mat_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, prefix::String="",
                          row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[],
                          hash_A::Vector{UInt8}=UInt8[], hash_B::Vector{UInt8}=UInt8[])
        # Try to get from product pool first
        pooled = _try_get_from_product_pool(MATPRODUCT_AB, hash_A, hash_B, row_partition, col_partition, $PetscScalar)
        if pooled !== nothing
            # Recompute using MAT_REUSE_MATRIX
            C_ref = Ref{PETSc.CMat}(pooled.ptr)
            PETSC_DEFAULT_REAL = $PetscReal(-2.0)
            PETSc.@chk ccall((:MatMatMult, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                             A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, C_ref)
            return pooled
        end

        # Pool miss or pooling disabled - create new matrix
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix)
        end
        return Cmat
    end

    # In-place version using MAT_REUSE_MATRIX
    function _mat_mat_mult!(C::PETSc.Mat{$PetscScalar}, A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        C_ref = Ref{PETSc.CMat}(C.ptr)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, C_ref)
        return nothing
    end

    # Transpose-matrix multiplication: C = A' * B
    function _mat_transpose_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, prefix::String="",
                                     row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[],
                                     hash_A::Vector{UInt8}=UInt8[], hash_B::Vector{UInt8}=UInt8[])
        # Try to get from product pool first
        pooled = _try_get_from_product_pool(MATPRODUCT_AtB, hash_A, hash_B, row_partition, col_partition, $PetscScalar)
        if pooled !== nothing
            # Recompute using MAT_REUSE_MATRIX
            C_ref = Ref{PETSc.CMat}(pooled.ptr)
            PETSC_DEFAULT_REAL = $PetscReal(-2.0)
            PETSc.@chk ccall((:MatTransposeMatMult, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                             A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, C_ref)
            return pooled
        end

        # Pool miss or pooling disabled - create new matrix
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatTransposeMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix)
        end
        return Cmat
    end

    # Matrix-transpose multiplication: C = A * B'
    function _mat_mat_transpose_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, prefix::String="",
                                     row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[],
                                     hash_A::Vector{UInt8}=UInt8[], hash_B::Vector{UInt8}=UInt8[])
        # Try to get from product pool first
        pooled = _try_get_from_product_pool(MATPRODUCT_ABt, hash_A, hash_B, row_partition, col_partition, $PetscScalar)
        if pooled !== nothing
            # Recompute using MAT_REUSE_MATRIX
            C_ref = Ref{PETSc.CMat}(pooled.ptr)
            PETSC_DEFAULT_REAL = $PetscReal(-2.0)
            PETSc.@chk ccall((:MatMatTransposeMult, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                             A, B, MAT_REUSE_MATRIX, PETSC_DEFAULT_REAL, C_ref)
            return pooled
        end

        # Pool miss or pooling disabled - create new matrix
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatTransposeMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix)
        end
        return Cmat
    end
end

# Import spdiagm from SparseArrays
import SparseArrays: spdiagm

"""
    spdiagm(kv::Pair{<:Integer, <:Vec{T}}...) -> Mat{T}
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T}}...) -> Mat{T}

Create a sparse diagonal matrix from distributed PETSc vectors.

Each pair `k => v` places the vector `v` on the `k`-th diagonal:
- `k = 0`: main diagonal
- `k > 0`: superdiagonal
- `k < 0`: subdiagonal

All vectors must have the same element type `T` and prefix. The matrix dimensions
are inferred from the diagonal positions and vector lengths, or can be specified explicitly.

# Examples
```julia
# Create a tridiagonal matrix
A = spdiagm(-1 => lower, 0 => diag, 1 => upper)

# Create a 100×100 matrix with specified vectors on diagonals
B = spdiagm(100, 100, 0 => v1, 1 => v2)
```
"""
function spdiagm(kv::Pair{<:Integer, <:Vec{T}}...) where {T}
    if length(kv) == 0
        throw(ArgumentError("spdiagm requires at least one diagonal"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate all vectors have same prefix - single @mpiassert (pulled out of loop)
    first_prefix = kv[1].second.obj.prefix
    all_same_prefix = all(v.obj.prefix == first_prefix for (k, v) in kv)
    @mpiassert all_same_prefix "All vectors must have the same prefix"

    # Infer matrix dimensions from diagonals without querying PETSc values
    # Avoid generic length(v) in case it triggers unintended array conversions
    m = 0; n = 0
    for (k, v) in kv
        len = v.obj.row_partition[end] - 1
        if k >= 0
            m = max(m, len)
            n = max(n, len + k)
        else
            m = max(m, len - k)
            n = max(n, len)
        end
    end

    return spdiagm(m, n, kv...)
end

function spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T}}...) where {T}
    if length(kv) == 0
        throw(ArgumentError("spdiagm requires at least one diagonal"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate all vectors have same prefix and correct lengths - single @mpiassert (pulled out of loop)
    first_prefix = kv[1].second.obj.prefix
    all_same_prefix = all(v.obj.prefix == first_prefix for (k, v) in kv)
    all_correct_lengths = all(begin
        required_len = k >= 0 ? min(m, n - k) : min(m + k, n)
        length(v) <= required_len  # allow shorter vectors; remaining entries are treated as zeros
    end for (k, v) in kv)
    @mpiassert (all_same_prefix && all_correct_lengths) "All vectors must have the same prefix and lengths not exceeding the allowed diagonal span"

    # Build local sparse contributions explicitly (I, J, V) using only locally-owned
    # entries from each distributed diagonal vector. This avoids global VecGetValues
    # (which only allows local indices on MPIVec).
    I = Int[]
    J = Int[]
    V = T[]
    for (k, v) in kv
        # Determine locally owned diagonal index range according to PETSc (not our partition)
        di_lo, di_hi = _vec_get_ownership_range(v.obj.v)
        nlocal = di_hi - di_lo + 1
        if nlocal <= 0
            continue
        end

        # Read only the local values using VecGetValues on local indices
        local_vals = zeros(T, nlocal)
        # Debug print ownership window per rank once per diagonal
        # (kept concise to reduce noise in normal runs)
        r = MPI.Comm_rank(MPI.COMM_WORLD)
        @static if false
            println("[DEBUG] rank=$(r) k=$(k) lo=$(di_lo) hi=$(di_hi) nlocal=$(nlocal)")
            flush(stdout)
        end
        _vec_getvalues_local!(local_vals, v.obj.v, di_lo, di_hi)
        @inbounds for t in 1:nlocal
            di = di_lo + t - 1
            if k >= 0
                row = di
                col = di + k
            else
                row = di - k  # shift downwards for subdiagonal
                col = di
            end
            if 1 <= row <= m && 1 <= col <= n
                push!(I, row)
                push!(J, col)
                push!(V, local_vals[t])
            end
        end
    end

    # Add structural zeros for all diagonal positions in this rank's row partition
    result_row_partition = default_row_partition(m, nranks)
    result_col_partition = default_row_partition(n, nranks)
    row_lo = result_row_partition[rank+1]
    row_hi = result_row_partition[rank+2] - 1

    # Build per-row column sets for this rank's rows (for fingerprint and structure allocation)
    # Also insert explicit structural zeros into (I,J,V)
    # Compute set of columns on each diagonal for each local row
    @inbounds for row in row_lo:row_hi
        for (k, v) in kv
            if k >= 0
                col = row + k
            else
                col = row
                # For subdiagonal k<0, the nonzero is at (row, col) only if row-k is within 1..m
                # which is equivalent to row + (-k) <= m
            end
            if k >= 0
                if 1 <= col <= n && row <= m
                    push!(I, row); push!(J, col); push!(V, zero(T))
                end
            else
                # k < 0: entry exists if (row, col) within bounds and also row - k <= m
                if 1 <= col <= n && 1 <= row <= m && (row - k) <= m
                    push!(I, row); push!(J, col); push!(V, zero(T))
                end
            end
        end
    end

    # Local sparse matrix with global dimensions
    local_result = sparse(I, J, V, m, n)

    # Compute structure fingerprint for the final assembled matrix
    # Derive local CSR for this rank's rows from the known diagonal structure
    # Create CSR directly from (I,J) for local rows [row_lo, row_hi]
    cols_by_row = Dict{Int, Vector{Int64}}()
    @inbounds for r in row_lo:row_hi
        cols_by_row[r] = Int64[]
    end
    @inbounds for idx in 1:length(I)
        r = I[idx]; c = J[idx]
        if row_lo <= r <= row_hi
            push!(cols_by_row[r], Int64(c-1))  # 0-based columns for hashing
        end
    end
    # Sort and unique per-row
    nloc = row_hi - row_lo + 1
    ia = Vector{Int64}(undef, nloc + 1)
    ia[1] = 0
    ja = Int64[]
    @inbounds for (i, r) in enumerate(row_lo:row_hi)
        cs = cols_by_row[r]
        sort!(cs); unique!(cs)
        append!(ja, cs)
        ia[i+1] = length(ja)
    end
    local_hash = _local_structure_hash(nloc, ia, ja)
    fp = _structure_fingerprint(local_hash, result_row_partition, result_col_partition, first_prefix)

    # Try reuse from non-product pool by fingerprint
    pooled = _try_get_from_nonproduct_pool_by_fingerprint(result_row_partition, result_col_partition,
                                                          first_prefix, fp, T)
    if pooled === nothing
        # Assemble via Mat_sum
        return Mat_sum(local_result;
                       row_partition=result_row_partition,
                       col_partition=result_col_partition,
                       prefix=first_prefix,
                       own_rank_only=false)
    else
        # Assemble directly into pooled matrix using ADD_VALUES
        rows_csc = rowvals(local_result)
        vals_csc = nonzeros(local_result)
        for col in 1:size(local_result, 2)
            for idx in nzrange(local_result, col)
                row = rows_csc[idx]
                val = vals_csc[idx]
                _mat_setvalues!(pooled, [row], [col], [val], PETSc.ADD_VALUES)
            end
        end
        PETSc.assemble(pooled)
        obj = _Mat{T}(pooled, result_row_partition, result_col_partition, first_prefix)
        return SafeMPI.DRef(obj)
    end
end

PETSc.@for_libpetsc begin
    # Fetch values for a contiguous local range [lo, hi] from an MPIVec.
    function _vec_getvalues_local!(vals::Vector{$PetscScalar}, vec::PETSc.Vec{$PetscScalar}, lo::Int, hi::Int)
        n = hi - lo + 1
        @assert n == length(vals)
        # PETSc uses 0-based indices
        idx = collect(lo-1:hi-1)
        idx_c = $PetscInt.(idx)
        PETSc.@chk ccall((:VecGetValues, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}),
                         vec, $PetscInt(n), idx_c, vals)
        return nothing
    end
end

# -----------------------------------------------------------------------------
# eachrow for distributed MPIDENSE matrices
# - Assumes A is MATMPIDENSE; iterates local rows and yields SubArray views
# - Performs exactly one MatDenseGetArrayRead per iterator and restores on finish
# -----------------------------------------------------------------------------

mutable struct _EachRowDense{T}
    aref::SafeMPI.DRef{_Mat{T}}           # keep the matrix alive
    petscA::PETSc.Mat{T}                  # underlying PETSc Mat
    row_lo::Int                           # global start row (1-based)
    nloc::Int                             # number of local rows
    ncols::Int                            # global number of columns
    data::Union{Matrix{T}, Nothing}       # local dense block view
    ptr::Ptr{T}                           # raw pointer from PETSc
end

PETSc.@for_libpetsc begin
    function _matdense_get_array_read(A::PETSc.Mat{$PetscScalar})
        parr = Ref{Ptr{$PetscScalar}}(C_NULL)
        PETSc.@chk ccall((:MatDenseGetArrayRead, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Ptr{Ptr{$PetscScalar}}), A, parr)
        return parr[]
    end
    function _matdense_restore_array_read(A::PETSc.Mat{$PetscScalar}, p::Ptr{$PetscScalar})
        parr = Ref{Ptr{$PetscScalar}}(p)
        PETSc.@chk ccall((:MatDenseRestoreArrayRead, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Ptr{Ptr{$PetscScalar}}), A, parr)
        return nothing
    end
end

function _eachrow_dense(A::Mat{T}) where {T}
    # Determine local row range from stored partition
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nloc = row_hi - row_lo + 1
    ncols = size(A, 2)

    # Acquire read-only dense local array from PETSc and wrap as a Julia Matrix
    p = _matdense_get_array_read(A.obj.A)
    data = unsafe_wrap(Array, p, (nloc, ncols); own=false)

    it = _EachRowDense{T}(A, A.obj.A, row_lo, nloc, ncols, data, p)
    # Ensure restore on GC if user exits early
    finalizer(it) do x
        if x.ptr != C_NULL
            _matdense_restore_array_read(x.petscA, x.ptr)
            x.ptr = C_NULL
            x.data = nothing
        end
    end
    return it
end

Base.IteratorEltype(::Type{_EachRowDense{T}}) where {T} = Base.HasEltype()
Base.eltype(::Type{_EachRowDense{T}}) where {T} = SubArray{T,1,Matrix{T},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true}
Base.IteratorSize(::Type{_EachRowDense{T}}) where {T} = Base.HasLength()
Base.length(it::_EachRowDense) = it.nloc

function Base.iterate(it::_EachRowDense{T}) where {T}
    it.nloc == 0 && return nothing
    row = @view it.data[1, :]
    return (row, 1)
end

function Base.iterate(it::_EachRowDense{T}, st::Int) where {T}
    i = st + 1
    if i > it.nloc
        # Restore PETSc array now that iteration is complete
        if it.ptr != C_NULL
            _matdense_restore_array_read(it.petscA, it.ptr)
            it.ptr = C_NULL
            it.data = nothing
        end
        return nothing
    end
    row = @view it.data[i, :]
    return (row, i)
end

function Base.eachrow(A::Mat{T}) where {T}
    return _eachrow_dense(A)
end

# -----------------------------------------------------------------------------
# Matrix structural fingerprinting
# - Query CSR structure (row pointers, column indices) without values
# - Hash the structure to create fingerprints for matrix reuse
# - New helper _structure_fingerprint computes a final fingerprint from a
#   per-rank local structure hash, aggregating with prefix and partitions.
#   _matrix_fingerprint becomes a thin wrapper over it.
# -----------------------------------------------------------------------------

PETSc.@for_libpetsc begin
    """
        _mat_get_rowij(mat::PETSc.Mat{T}) -> (nrows, ia, ja)

    Query the local CSR structure of a PETSc matrix.

    Returns:
    - `nrows`: Number of local rows
    - `ia`: Row pointer array (ia[i] = start index in ja for row i), read-only view
    - `ja`: Column indices array, read-only view

    IMPORTANT: Must call `_mat_restore_rowij!` when done to avoid memory leaks.
    Do not modify the returned arrays as they are read-only views into PETSc memory.
    """
    function _mat_get_rowij(mat::PETSc.Mat{$PetscScalar})
        n = Ref{$PetscInt}(0)
        ia_ptr = Ref{Ptr{$PetscInt}}(C_NULL)
        ja_ptr = Ref{Ptr{$PetscInt}}(C_NULL)
        done = Ref{PETSc.PetscBool}(PETSC_FALSE)

        PETSc.@chk ccall((:MatGetRowIJ, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, $PetscInt, PETSc.PetscBool, PETSc.PetscBool,
                          Ptr{$PetscInt}, Ptr{Ptr{$PetscInt}}, Ptr{Ptr{$PetscInt}},
                          Ptr{PETSc.PetscBool}),
                         mat, $PetscInt(0), PETSC_FALSE, PETSC_TRUE,
                         n, ia_ptr, ja_ptr, done)

        if done[] == PETSC_FALSE
            error("MatGetRowIJ failed for this matrix type (some matrix types like ELEMENTAL or SCALAPACK do not support this operation)")
        end

        # Convert C arrays to Julia arrays (views, no copy)
        nrows = Int(n[])
        ia = unsafe_wrap(Array, ia_ptr[], nrows + 1; own=false)
        nnz = ia[end]
        ja = unsafe_wrap(Array, ja_ptr[], nnz; own=false)

        return (nrows, ia, ja, ia_ptr[], ja_ptr[])
    end

    """
        _mat_restore_rowij!(mat::PETSc.Mat{T}, nrows, ia_ptr, ja_ptr)

    Restore CSR structure access obtained from `_mat_get_rowij`.

    MUST be called after `_mat_get_rowij` to avoid memory leaks.
    """
    function _mat_restore_rowij!(mat::PETSc.Mat{$PetscScalar}, nrows::Int, ia_ptr::Ptr{$PetscInt}, ja_ptr::Ptr{$PetscInt})
        n_ref = Ref{$PetscInt}($PetscInt(nrows))
        ia_ref = Ref{Ptr{$PetscInt}}(ia_ptr)
        ja_ref = Ref{Ptr{$PetscInt}}(ja_ptr)
        done = Ref{PETSc.PetscBool}(PETSC_TRUE)

        PETSc.@chk ccall((:MatRestoreRowIJ, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, $PetscInt, PETSc.PetscBool, PETSc.PetscBool,
                          Ptr{$PetscInt}, Ptr{Ptr{$PetscInt}}, Ptr{Ptr{$PetscInt}},
                          Ptr{PETSc.PetscBool}),
                         mat, $PetscInt(0), PETSC_FALSE, PETSC_TRUE,
                         n_ref, ia_ref, ja_ref, done)
        return nothing
    end

    """
        _mat_hash_local_structure(mat::PETSc.Mat{T}) -> Vector{UInt8}

    Compute SHA-1 hash of the local CSR structure (row pointers and column indices only).

    Returns a 20-byte hash vector representing the sparsity pattern of local rows.
    Does NOT include matrix values, only the structural information.

    For matrix types that don't support MatGetRowIJ (like MPIDENSE, ELEMENTAL, SCALAPACK),
    returns a zero hash to indicate fingerprinting is not supported.
    """
    function _mat_hash_local_structure(mat::PETSc.Mat{$PetscScalar})
        # Try to get CSR structure - some matrix types don't support this
        try
            nrows, ia, ja, ia_ptr, ja_ptr = _mat_get_rowij(mat)

            # Hash the structure (not values!)
            # We need to convert PetscInt arrays to a standard format for hashing
            io = IOBuffer()
            write(io, Int64(nrows))  # Number of local rows (standardized to Int64)

            # Write row pointers (ia array)
            for i in 1:(nrows + 1)
                write(io, Int64(ia[i]))  # Standardize to Int64 for consistent hashing
            end

            # Write column indices (ja array)
            nnz = ia[end]
            for i in 1:nnz
                write(io, Int64(ja[i]))  # Standardize to Int64 for consistent hashing
            end

            local_hash = sha1(take!(io))

            # Restore (important!)
            _mat_restore_rowij!(mat, nrows, ia_ptr, ja_ptr)

            return local_hash  # Vector{UInt8}, length 20
        catch e
            # Matrix type doesn't support MatGetRowIJ (e.g., MPIDENSE, ELEMENTAL)
            # Return a zero hash to indicate fingerprinting is not supported
            return zeros(UInt8, 20)
        end
    end
end

"""
    _local_structure_hash(nrows::Integer, ia::AbstractVector{<:Integer}, ja::AbstractVector{<:Integer})

Hash per-rank local CSR structure (row pointers ia of length nrows+1, column indices ja of length ia[end]).
The ia/ja values are expected in PETSc's 0-based convention for consistency.
Returns a 20-byte SHA-1 Vector{UInt8}.
"""
function _local_structure_hash(nrows::Integer, ia::AbstractVector{<:Integer}, ja::AbstractVector{<:Integer})
    io = IOBuffer()
    write(io, Int64(nrows))
    @inbounds for i in 1:(Int(nrows) + 1)
        write(io, Int64(ia[i]))
    end
    nnz = Int(ia[Int(nrows)+1])
    @inbounds for i in 1:nnz
        write(io, Int64(ja[i]))
    end
    return sha1(take!(io))
end

"""
    _structure_fingerprint(local_hash::Vector{UInt8}, row_partition::Vector{Int}, col_partition::Vector{Int}, prefix::String)

Compute a final 20-byte fingerprint from a per-rank local structure hash by Allgather'ing
across ranks and combining with prefix and partitions, mirroring _matrix_fingerprint.
"""
function _structure_fingerprint(local_hash::Vector{UInt8},
                                row_partition::Vector{Int},
                                col_partition::Vector{Int},
                                prefix::String)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    # Allgather all hashes (each 20 bytes)
    all_hashes = zeros(UInt8, 20 * nranks)
    MPI.Allgather!(local_hash, MPI.UBuffer(all_hashes, 20), MPI.COMM_WORLD)

    # Create final fingerprint
    io = IOBuffer()
    write(io, all_hashes)
    write(io, prefix)
    for x in row_partition
        write(io, Int64(x))
    end
    for x in col_partition
        write(io, Int64(x))
    end
    return sha1(take!(io))
end

"""
    _matrix_fingerprint(A::PETSc.Mat{T}, row_partition::Vector{Int}, col_partition::Vector{Int}, prefix::String) -> Vector{UInt8}

Create a structural fingerprint of a distributed PETSc matrix.

The fingerprint is computed by:
1. Hashing the local CSR structure (row pointers, column indices) on each rank
2. Gathering all local hashes via MPI.Allgather
3. Combining with prefix and partitions
4. Computing final SHA-1 hash

The result is a 20-byte hash that uniquely identifies the matrix structure
(but not its values). Two matrices with identical structure but different
values will produce the same fingerprint.

This is useful for determining whether matrices can reuse the same
preconditioner setup, symbolic factorization, or other structure-dependent
operations.

Note: Global matrix dimensions are not hashed separately since they are
already encoded in the partition vectors (row_partition[end]-1 = nrows,
col_partition[end]-1 = ncols).
"""
function _matrix_fingerprint(A::PETSc.Mat{T}, row_partition::Vector{Int},
                              col_partition::Vector{Int}, prefix::String) where T
    # Compute a local structure hash, then aggregate via _structure_fingerprint
    local_hash = _mat_hash_local_structure(A)
    return _structure_fingerprint(local_hash, row_partition, col_partition, prefix)
end

# -----------------------------------------------------------------------------
# Matrix Pool Utility Functions
# -----------------------------------------------------------------------------

"""
    clear_mat_pool!()

Clear all matrices from the pool, destroying them immediately.
Useful for testing or explicit memory management.
"""
function clear_mat_pool!()
    # Clear all pools by iterating over each PetscScalar type
    PETSc.@for_libpetsc begin
        # Clear non-product pool
        pool_nonproduct = $(Symbol(:MAT_POOL_NONPRODUCT_, PetscScalar))
        for (key, mat_list) in pool_nonproduct
            for pooled in mat_list
                _destroy_petsc_mat!(pooled.mat)
            end
        end
        empty!(pool_nonproduct)

        # Clear product pool
        pool_product = $(Symbol(:MAT_POOL_PRODUCT_, PetscScalar))
        for (key, mat_list) in pool_product
            for pooled in mat_list
                _destroy_petsc_mat!(pooled.mat)
            end
        end
        empty!(pool_product)
    end
    return nothing
end

"""
    get_mat_pool_stats() -> Dict

Return statistics about the current matrix pool state.
Returns a dictionary with keys (M, N, prefix, type, pool_type) => count for non-product matrices
and (product_type, type, pool_type) => count for product matrices.
"""
function get_mat_pool_stats()
    stats = Dict{Tuple, Int}()
    # Gather stats from all pools
    for petsc_scalar in [Float64, ComplexF64]  # Common PETSc scalar types
        # Non-product pool stats
        pool_nonproduct_name = Symbol(:MAT_POOL_NONPRODUCT_, petsc_scalar)
        if isdefined(@__MODULE__, pool_nonproduct_name)
            pool = getfield(@__MODULE__, pool_nonproduct_name)
            for (key, mat_list) in pool
                stats[(key[1], key[2], key[3], petsc_scalar, :nonproduct)] = length(mat_list)
            end
        end

        # Product pool stats
        pool_product_name = Symbol(:MAT_POOL_PRODUCT_, petsc_scalar)
        if isdefined(@__MODULE__, pool_product_name)
            pool = getfield(@__MODULE__, pool_product_name)
            for (key, mat_list) in pool
                stats[(key[1], petsc_scalar, :product)] = length(mat_list)
            end
        end
    end
    return stats
end
