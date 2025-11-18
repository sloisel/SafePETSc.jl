# -----------------------------------------------------------------------------
# Matrix Construction
# -----------------------------------------------------------------------------

"""
    Mat_uniform(A::Matrix{T}; row_partition=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)), col_partition=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)), Prefix::Type=MPIAIJ) -> DRef{Mat{T,Prefix}}

**MPI Collective**

Create a distributed PETSc matrix from a Julia matrix, asserting uniform distribution across ranks (on MPI.COMM_WORLD).

- `A::Matrix{T}` must be identical on all ranks (`mpi_uniform`).
- `row_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
- `col_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start column (1-indexed) for rank i-1.
- `Prefix` is a type parameter for MatSetOptionsPrefix() to set matrix-specific command-line options (default: MPIAIJ).
- Returns a DRef that will destroy the PETSc Mat collectively when all ranks release their reference.
"""
function Mat_uniform(A::Matrix{T};
                     row_partition::Vector{Int}=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)),
                     col_partition::Vector{Int}=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)),
                     Prefix::Type=MPIAIJ) where T
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
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, Prefix)

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
    obj = _Mat{T,Prefix}(petsc_mat, row_partition, col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Mat_uniform(A::SparseMatrixCSC{T}; row_partition=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)), col_partition=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)), Prefix::Type=MPIAIJ) -> DRef{Mat{T,Prefix}}

**MPI Collective**

Create a distributed PETSc matrix from a sparse Julia matrix, asserting uniform distribution across ranks (on MPI.COMM_WORLD).

- `A::SparseMatrixCSC{T}` must be identical on all ranks (`mpi_uniform`).
- `row_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
- `col_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start column (1-indexed) for rank i-1.
- `Prefix` is a type parameter for MatSetOptionsPrefix() to set matrix-specific command-line options (default: MPIAIJ).
- Returns a DRef that will destroy the PETSc Mat collectively when all ranks release their reference.

Each rank inserts only the values from its assigned row partition using INSERT_VALUES mode.
"""
function Mat_uniform(A::SparseMatrixCSC{T};
                     row_partition::Vector{Int}=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)),
                     col_partition::Vector{Int}=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)),
                     Prefix::Type=MPIAIJ) where T
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

    # Debug output for troubleshooting

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
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, Prefix)

    # Set values from the local row partition only, using INSERT_VALUES mode
    # Iterate through columns and extract nonzero entries
    rows_csc = rowvals(A)
    vals_csc = nonzeros(A)

    for col in 1:size(A, 2)
        for idx in nzrange(A, col)
            row = rows_csc[idx]
            # Only insert values that belong to this rank's row partition
            if row_lo <= row <= row_hi
                val = vals_csc[idx]
                _mat_setvalues!(petsc_mat, [row], [col], [val], PETSc.INSERT_VALUES)
            end
        end
    end

    # Assemble the matrix
    PETSc.assemble(petsc_mat)

    # Wrap and DRef-manage with a manager bound to this communicator
    obj = _Mat{T,Prefix}(petsc_mat, row_partition, col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Mat_sum(A::SparseMatrixCSC{T}; row_partition=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)), col_partition=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)), Prefix::Type=MPIAIJ, own_rank_only=false) -> DRef{Mat{T,Prefix}}

**MPI Collective**

Create a distributed PETSc matrix by summing sparse matrices across ranks (on MPI.COMM_WORLD).

- `A::SparseMatrixCSC{T}` can differ across ranks; nonzero entries are summed across all ranks.
- `row_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start row (1-indexed) for rank i-1.
- `col_partition` is a Vector{Int} of length nranks+1 where partition[i] is the start column (1-indexed) for rank i-1.
- `Prefix` is a type parameter for MatSetOptionsPrefix() to set matrix-specific command-line options (default: MPIAIJ).
- `own_rank_only::Bool` (default=false): if true, asserts that all nonzero entries fall within this rank's row partition.
- Returns a DRef that will destroy the PETSc Mat collectively when all ranks release their reference.

Uses MatSetValues with ADD_VALUES mode to sum contributions from all ranks.
"""
function Mat_sum(A::SparseMatrixCSC{T};
                 row_partition::Vector{Int}=default_row_partition(size(A, 1), MPI.Comm_size(MPI.COMM_WORLD)),
                 col_partition::Vector{Int}=default_row_partition(size(A, 2), MPI.Comm_size(MPI.COMM_WORLD)),
                 Prefix::Type=MPIAIJ,
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
    petsc_mat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, Prefix)

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
    obj = _Mat{T,Prefix}(petsc_mat, row_partition, col_partition)
    return SafeMPI.DRef(obj)
end

# Create a distributed PETSc Mat for a given element type T by dispatching to the
# underlying PETSc scalar variant via PETSc.@for_libpetsc
function _mat_create_mpi_for_T(::Type{T}, nlocal_rows::Integer, nlocal_cols::Integer,
                               nglobal_rows::Integer, nglobal_cols::Integer, Prefix::Type) where {T}
    return _mat_create_mpi_impl(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, Prefix)
end

PETSc.@for_libpetsc begin
    function _mat_create_mpi_impl(::Type{$PetscScalar}, nlocal_rows::Integer, nlocal_cols::Integer,
                                  nglobal_rows::Integer, nglobal_cols::Integer, Prefix::Type)
        mat = PETSc.Mat{$PetscScalar}(C_NULL)
        PETSc.@chk ccall((:MatCreate, $libpetsc), PETSc.PetscErrorCode,
                         (MPI.MPI_Comm, Ptr{CMat}), MPI.COMM_WORLD, mat)
        PETSc.@chk ccall((:MatSetSizes, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, $PetscInt, $PetscInt, $PetscInt, $PetscInt),
                         mat, $PetscInt(nlocal_rows), $PetscInt(nlocal_cols),
                         $PetscInt(nglobal_rows), $PetscInt(nglobal_cols))

        # Set prefix and let PETSc options determine the type
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CMat, Cstring), mat, prefix_str)
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

# Element type and shape for internal _Mat and DRef-wrapped _Mat
Base.eltype(::_Mat{T}) where {T} = T
Base.size(m::_Mat) = (m.row_partition[end] - 1, m.col_partition[end] - 1)
Base.size(m::_Mat, d::Integer) = d == 1 ? (m.row_partition[end] - 1) : (d == 2 ? (m.col_partition[end] - 1) : 1)

Base.eltype(r::SafeMPI.DRef{<:_Mat}) = Base.eltype(r.obj)
Base.size(r::SafeMPI.DRef{<:_Mat}) = Base.size(r.obj)
Base.size(r::SafeMPI.DRef{<:_Mat}, d::Integer) = Base.size(r.obj, d)

# Broadcast support - Mat objects are already array-like and can be broadcast over
Base.Broadcast.broadcastable(m::Mat) = m
Base.Broadcast.BroadcastStyle(::Type{<:Mat}) = Base.Broadcast.DefaultArrayStyle{2}()

# Adjoint support for Mat
Base.adjoint(A::Mat{T}) where {T} = LinearAlgebra.Adjoint(A)

"""
    Mat(adj::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) -> Mat{T,PrefixResult}

**MPI Collective**

Materialize an adjoint matrix into a new transposed Mat.

Creates a new matrix B = A^T using PETSc's transpose operations.
The result prefix is determined by querying what PETSc creates.
"""
function Mat(adj::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) where {T,Prefix}
    A = parent(adj)::Mat{T,Prefix}
    # Create B = A^T via PETSc (MAT_INITIAL_MATRIX) - returns both matrix and prefix
    B_petsc, ResultPrefix = _mat_transpose(A.obj.A)
    # Swap row/col partitions for the transpose
    obj = _Mat{T,ResultPrefix}(B_petsc, A.obj.col_partition, A.obj.row_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.transpose!(B::Mat{T,Prefix}, A::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

In-place transpose: B = A^T.

Reuses the pre-allocated matrix B. Dimensions and partitions must match appropriately.
"""
function LinearAlgebra.transpose!(B::Mat{T,Prefix}, A::Mat{T,Prefix}) where {T,Prefix}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    @mpiassert (size(A, 1) == size(B, 2) && size(A, 2) == size(B, 1) &&
                A.obj.row_partition == B.obj.col_partition &&
                A.obj.col_partition == B.obj.row_partition) "Transpose dimensions must match (A is $(size(A)), B must be $(size(A, 2))×$(size(A, 1)), got $(size(B))), A's row partition must match B's column partition, and A's column partition must match B's row partition"

    # Perform in-place transpose using PETSc reuse path. The caller must
    # ensure B was originally created by MatTranspose(A, MAT_INITIAL_MATRIX)
    # (or has a valid precursor) with matching partitions.
    _mat_transpose!(B.obj.A, A.obj.A)

    return B
end

"""
    Base.:*(A::Mat{T,Prefix}, x::Vec{T,Prefix}) -> Vec{T,Prefix}

**MPI Collective**

Matrix-vector multiplication: y = A * x.

Returns a new distributed vector with the result.
"""
function Base.:*(A::Mat{T,PrefixA}, x::Vec{T,PrefixX}) where {T,PrefixA,PrefixX}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(x)[1]
    @mpiassert n == vec_length && A.obj.col_partition == x.obj.row_partition "Matrix columns must match vector length (A: $(m)×$(n), x: $(vec_length)) and A.col_partition=$(A.obj.col_partition) must match x.row_partition=$(x.obj.row_partition)"

    # Create result vector with A's row partition
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    y_petsc = _vec_create_mpi_for_T(T, nlocal, m, PrefixX, A.obj.row_partition)

    # Perform matrix-vector multiplication using PETSc
    _mat_mult_vec!(y_petsc, A.obj.A, x.obj.v)

    PETSc.assemble(y_petsc)

    # Wrap in DRef
    obj = _Vec{T,PrefixX}(y_petsc, A.obj.row_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.mul!(y::Vec{T,Prefix}, A::Mat{T,Prefix}, x::Vec{T,Prefix}) -> Vec{T,Prefix}

**MPI Collective**

In-place matrix-vector multiplication: y = A * x.

Reuses the pre-allocated vector y. Dimensions and partitions must match appropriately.
"""
function LinearAlgebra.mul!(y::Vec{T,Prefix}, A::Mat{T,Prefix}, x::Vec{T,Prefix}) where {T,Prefix}
    # Check dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(A)
    y_length = size(y)[1]
    x_length = size(x)[1]
    @mpiassert (m == y_length && n == x_length &&
                A.obj.row_partition == y.obj.row_partition &&
                A.obj.col_partition == x.obj.row_partition) "Output vector y must have length matching matrix rows (A: $(m)×$(n), y: $(y_length)), input vector x must have length matching matrix columns (x: $(x_length)), matrix row partition must match output vector partition, and matrix column partition must match input vector partition"

    # Perform matrix-vector multiplication using PETSc (reuses y)
    _mat_mult_vec!(y.obj.v, A.obj.A, x.obj.v)

    PETSc.assemble(y.obj.v)

    return y
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, x::Vec{T,PrefixX}) -> Vec{T,PrefixX}

**MPI Collective**

Transpose-matrix-vector multiplication: y = A' * x.

Computes the product of a transposed matrix and a vector using PETSc's MatMultTranspose.
Matrices and vectors can have different prefixes.

Returns a new distributed vector with the result.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, x::Vec{T,PrefixX}) where {T,PrefixA,PrefixX}
    A = parent(At)::Mat{T,PrefixA}

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(x)[1]
    @mpiassert m == vec_length && A.obj.row_partition == x.obj.row_partition "Matrix rows must match vector length (A: $(m)×$(n), x: $(vec_length)) and A.row_partition=$(A.obj.row_partition) must match x.row_partition=$(x.obj.row_partition)"

    # Create result vector with A's column partition (since A' has dimensions n×m)
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal = col_hi - col_lo + 1

    y_petsc = _vec_create_mpi_for_T(T, nlocal, n, PrefixX, A.obj.col_partition)

    # Perform y = A^T * x using PETSc
    _mat_mult_transpose_vec!(y_petsc, A.obj.A, x.obj.v)

    PETSc.assemble(y_petsc)

    # Wrap in DRef
    obj = _Vec{T,PrefixX}(y_petsc, A.obj.col_partition)
    return SafeMPI.DRef(obj)
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
    Base.:+(A::Mat{T,Prefix}, B::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

Add two distributed PETSc matrices. Requires identical sizes, partitions, and prefix.
Note: Does not use pooling due to PETSc MatAXPY internal state management issues.
"""
function Base.:+(A::Mat{T,Prefix}, B::Mat{T,Prefix}) where {T,Prefix}
    @mpiassert (size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) &&
                A.obj.row_partition == B.obj.row_partition &&
                A.obj.col_partition == B.obj.col_partition) "Matrix addition requires identical sizes and identical partitions"

    # Create fresh result matrix
    nr = MPI.Comm_rank(MPI.COMM_WORLD)
    nlocal_rows = A.obj.row_partition[nr+2] - A.obj.row_partition[nr+1]
    nlocal_cols = A.obj.col_partition[nr+2] - A.obj.col_partition[nr+1]
    Cmat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols,
                                 size(A,1), size(A,2), Prefix)
    PETSc.assemble(Cmat)  # Must assemble before MatAXPY

    # Accumulate C = A + B using MatAXPY
    _mat_axpy!(Cmat, one(T), A.obj.A, DIFFERENT_NONZERO_PATTERN)
    _mat_axpy!(Cmat, one(T), B.obj.A, DIFFERENT_NONZERO_PATTERN)
    PETSc.assemble(Cmat)

    obj = _Mat{T,Prefix}(Cmat, A.obj.row_partition, A.obj.col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:-(A::Mat{T,Prefix}, B::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

Subtract two distributed PETSc matrices. Requires identical sizes, partitions, and prefix.
Note: Does not use pooling due to PETSc MatAXPY internal state management issues.
"""
function Base.:-(A::Mat{T,Prefix}, B::Mat{T,Prefix}) where {T,Prefix}
    @mpiassert (size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) &&
                A.obj.row_partition == B.obj.row_partition &&
                A.obj.col_partition == B.obj.col_partition) "Matrix subtraction requires identical sizes and identical partitions"

    # Create fresh result matrix
    nr = MPI.Comm_rank(MPI.COMM_WORLD)
    nlocal_rows = A.obj.row_partition[nr+2] - A.obj.row_partition[nr+1]
    nlocal_cols = A.obj.col_partition[nr+2] - A.obj.col_partition[nr+1]
    Cmat = _mat_create_mpi_for_T(T, nlocal_rows, nlocal_cols,
                                 size(A,1), size(A,2), Prefix)
    PETSc.assemble(Cmat)  # Must assemble before MatAXPY

    # Accumulate C = A - B using MatAXPY
    _mat_axpy!(Cmat, one(T), A.obj.A, DIFFERENT_NONZERO_PATTERN)
    _mat_axpy!(Cmat, -one(T), B.obj.A, DIFFERENT_NONZERO_PATTERN)
    PETSc.assemble(Cmat)

    obj = _Mat{T,Prefix}(Cmat, A.obj.row_partition, A.obj.col_partition)
    return SafeMPI.DRef(obj)
end

# Helper function to extract owned rows from a PETSc Mat to Julia SparseMatrixCSC
# Uses efficient bulk operations: MatDenseGetArrayRead for dense matrices,
# MatGetRow for sparse matrices (avoids element-by-element extraction)
function _mat_to_local_sparse(A::Mat{T,Prefix}) where {T,Prefix}
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

    # Use different extraction methods based on matrix type
    if is_dense(A)
        # Dense matrix: use MatDenseGetArrayRead to get entire local block at once
        p = _matdense_get_array_read(A.obj.A)
        local_data = unsafe_wrap(Array, p, (nlocal_rows, n); own=false)

        try
            # Extract nonzeros from dense local block
            for local_row in 1:nlocal_rows
                global_row = row_lo + local_row - 1
                for j in 1:n
                    val = local_data[local_row, j]
                    if abs(val) > 0
                        push!(I, global_row)
                        push!(J, j)
                        push!(V, val)
                    end
                end
            end
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
    else
        # Sparse matrix: use MatGetRow to extract one row at a time
        for local_row in 1:nlocal_rows
            global_row = row_lo + local_row - 1

            # Get sparse row structure from PETSc
            ncols_row, cols_ptr, vals_ptr = _mat_get_row(A.obj.A, global_row)

            if ncols_row > 0
                # Wrap PETSc arrays (must copy since we'll restore them)
                cols = unsafe_wrap(Array, cols_ptr, ncols_row; own=false)
                vals = unsafe_wrap(Array, vals_ptr, ncols_row; own=false)

                # Copy data and convert to 1-based indexing
                for k in 1:ncols_row
                    push!(I, global_row)
                    push!(J, Int(cols[k]) + 1)  # Convert to 1-based
                    push!(V, vals[k])
                end
            end

            # Must restore row after use
            _mat_restore_row(A.obj.A, global_row, ncols_row, cols_ptr, vals_ptr)
        end
    end

    # Create sparse matrix of full global size
    return sparse(I, J, V, m, n)
end

# Helper function to extract owned elements from a PETSc Vec to Julia SparseMatrixCSC (as column vector)
# Treats the Vec as a single-column matrix for concatenation operations
function _mat_to_local_sparse(v::Vec{T,Prefix}) where {T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    m = length(v)
    row_lo = v.obj.row_partition[rank+1]
    row_hi = v.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    # Get local vector data using unsafe_localarray
    local_view = PETSc.unsafe_localarray(v.obj.v; read=true)

    try
        # Build sparse matrix arrays (single column)
        I = Int[]
        J = Int[]
        V = T[]

        for local_row in 1:nlocal_rows
            global_row = row_lo + local_row - 1
            val = local_view[local_row]
            if val != 0  # Only store nonzeros
                push!(I, global_row)
                push!(J, 1)  # Single column
                push!(V, val)
            end
        end

        return sparse(I, J, V, m, 1)
    finally
        Base.finalize(local_view)
    end
end

# Helper to extract Prefix type parameter from Vec or Mat
_get_prefix(::Union{Vec{T,P}, Mat{T,P}}) where {T,P} = P

# Helper to get row partition (works for both Vec and Mat)
_get_row_partition(v::Vec) = v.obj.row_partition
_get_row_partition(m::Mat) = m.obj.row_partition

# Helper to get column partition (Mat has it, Vec represents 1 column)
function _get_col_partition(m::Mat)
    return m.obj.col_partition
end
function _get_col_partition(v::Vec)
    # Vec represents a single column - return default partition for 1 column
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    return default_row_partition(1, nranks)  # Partition 1 column across ranks
end

# Helper to compute output width (number of columns) after concatenation
function _compute_output_width(As, dims)
    if dims == 1  # vcat: width stays same
        return size(As[1], 2)
    elseif dims == 2  # hcat: width is sum of all widths
        return sum(size(A, 2) for A in As)
    else
        throw(ArgumentError("dims must be 1 or 2 for matrix concatenation"))
    end
end

"""
    Base.cat(As::Union{Vec{T,Prefix},Mat{T,Prefix}}...; dims) -> Union{Vec{T,Prefix}, Mat{T,Prefix}}

**MPI Collective**

Concatenate distributed PETSc vectors and/or matrices along dimension `dims`.

# Arguments
- `As::Union{Vec{T,Prefix},Mat{T,Prefix}}...`: One or more vectors or matrices with the same element type `T`
- `dims`: Concatenation dimension (1 for vertical/vcat, 2 for horizontal/hcat)

# Return Type
- Returns `Vec{T,Prefix}` when `dims=1` and result has a single column (vertical stacking of vectors)
- Returns `Mat{T,Prefix}` otherwise (horizontal concatenation or matrix inputs)

# Requirements
All inputs must:
- Have the same element type `T`
- Have compatible sizes and partitions for the concatenation dimension
  - For `dims=1` (vcat): same number of columns and column partition
  - For `dims=2` (hcat): same number of rows and row partition

# Automatic Prefix Selection
The output `Prefix` type is automatically determined to ensure correctness:
- **MPIDENSE** if any input has `Prefix=MPIDENSE` (dense format required)
- **MPIDENSE** if concatenating vectors horizontally with width > 1 (e.g., `hcat(x, y)`)
- Otherwise, preserves the first input's `Prefix`

This ensures that operations like `hcat(vec1, vec2)` produce dense matrices as expected,
since vectors are inherently dense and horizontal concatenation creates a dense result.

# Implementation
The concatenation is performed by:
1. Each rank extracts its owned rows from each input as a Julia sparse matrix
2. Standard Julia `cat` is applied locally on each rank
3. The results are combined across ranks using `Vec_sum` (for single-column results) or `Mat_sum` (otherwise)

# Examples
```julia
# Vertical concatenation (stacking) - returns Vec
x = Vec_uniform([1.0, 2.0, 3.0])
y = Vec_uniform([4.0, 5.0, 6.0])
v = vcat(x, y)  # Returns Vec{Float64,MPIAIJ} with 6 elements

# Horizontal concatenation - returns Mat
M = hcat(x, y)  # Returns Mat{Float64,MPIDENSE} of size 3×2

# Matrix concatenation - returns Mat
A = Mat_uniform(sparse([1 2; 3 4]))
B = Mat_uniform(sparse([5 6; 7 8]))
C = vcat(A, B)  # Returns Mat{Float64,MPIAIJ} of size 4×2
```

See also: [`vcat`](@ref), [`hcat`](@ref), [`Vec_sum`](@ref), [`Mat_sum`](@ref)
"""
function Base.cat(As::Union{Vec{T},Mat{T}}...; dims, row_partition=nothing, col_partition=nothing) where {T}
    n = length(As)
    if n == 0
        throw(ArgumentError("cat requires at least one matrix or vector"))
    end

    # Determine output Prefix: upgrade to MPIDENSE if needed
    prefixes = [_get_prefix(a) for a in As]
    width = _compute_output_width(As, dims)
    Prefix = (MPIDENSE in prefixes || (As[1] isa Vec && width > 1)) ? MPIDENSE : prefixes[1]

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate partition compatibility based on dims - collect conditions first
    if dims == 1  # Vertical concatenation
        # All matrices/vectors must have same column partition and number of columns
        first_col_partition = _get_col_partition(As[1])
        first_ncols = size(As[1], 2)
        all_same_col_partition = all(_get_col_partition(As[k]) == first_col_partition for k in 2:n)
        all_same_ncols = all(size(As[k], 2) == first_ncols for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        col_partitions = [_get_col_partition(As[k]) for k in 1:n]
        @mpiassert (all_same_col_partition && all_same_ncols) "For dims=1 (vcat): all inputs must have the same column partition and number of columns. Got col_partitions=$(col_partitions), ncols=$([size(As[k], 2) for k in 1:n])"
    elseif dims == 2  # Horizontal concatenation
        # All matrices/vectors must have same row partition and number of rows
        first_row_partition = _get_row_partition(As[1])
        first_nrows = size(As[1], 1)
        all_same_row_partition = all(_get_row_partition(As[k]) == first_row_partition for k in 2:n)
        all_same_nrows = all(size(As[k], 1) == first_nrows for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        row_partitions = [_get_row_partition(As[k]) for k in 1:n]
        @mpiassert (all_same_row_partition && all_same_nrows) "For dims=2 (hcat): all inputs must have the same row partition and number of rows. Got row_partitions=$(row_partitions), nrows=$([size(As[k], 1) for k in 1:n])"
    else
        throw(ArgumentError("dims must be 1 or 2 for matrix concatenation"))
    end

    # Extract local sparse matrices (owned rows only)
    local_sparse = [_mat_to_local_sparse(A) for A in As]

    # Perform local cat operation
    local_result = cat(local_sparse...; dims=dims)

    # Determine result partitions
    if dims == 1  # Vertical: rows increase, cols stay same
        # New row partition: use provided or default partitioning for total rows
        total_rows = sum(size(A, 1) for A in As)
        result_row_partition = row_partition === nothing ? default_row_partition(total_rows, nranks) : row_partition
        result_col_partition = col_partition === nothing ? _get_col_partition(As[1]) : col_partition
    else  # dims == 2, Horizontal: cols increase, rows stay same
        # New column partition: use provided or default partitioning for total columns
        total_cols = sum(size(A, 2) for A in As)
        result_col_partition = col_partition === nothing ? default_row_partition(total_cols, nranks) : col_partition
        result_row_partition = row_partition === nothing ? _get_row_partition(As[1]) : row_partition
    end

    # Return Vec for single-column vertical concatenation, Mat otherwise
    if dims == 1 && size(local_result, 2) == 1
        # Extract single column as sparse vector
        local_vec = sparsevec(local_result[:, 1])
        return Vec_sum(local_vec;
                       row_partition=result_row_partition,
                       Prefix=Prefix,
                       own_rank_only=false)
    else
        # Use Mat_sum to combine results across ranks
        return Mat_sum(local_result;
                       row_partition=result_row_partition,
                       col_partition=result_col_partition,
                       Prefix=Prefix,
                       own_rank_only=false)
    end
end

"""
    Base.vcat(As::Union{Vec{T,Prefix},Mat{T,Prefix}}...) -> Union{Vec{T,Prefix}, Mat{T,Prefix}}

**MPI Collective**

Vertically concatenate (stack) distributed PETSc vectors and/or matrices.

Equivalent to `cat(As...; dims=1)`. Stacks inputs vertically, increasing the number of rows
while keeping the number of columns constant.

# Return Type
- Returns `Vec{T,Prefix}` when concatenating only vectors (single-column result)
- Returns `Mat{T,Prefix}` when concatenating matrices or when result has multiple columns

# Requirements
All inputs must have the same number of columns and the same column partition.

# Prefix Selection
- Typically preserves the input `Prefix` (e.g., `MPIAIJ` for vectors)
- Upgrades to `MPIDENSE` if any input is `MPIDENSE`

# Examples
```julia
# Concatenating vectors returns a Vec
x = Vec_uniform([1.0, 2.0])
y = Vec_uniform([3.0, 4.0])
v = vcat(x, y)  # Vec{Float64,MPIAIJ} with 4 elements

# Concatenating matrices returns a Mat
A = Mat_uniform(sparse([1 2; 3 4]))
B = Mat_uniform(sparse([5 6; 7 8]))
C = vcat(A, B)  # Mat{Float64,MPIAIJ} of size 4×2
```

See also: [`cat`](@ref), [`hcat`](@ref)
"""
Base.vcat(As::Union{Vec{T},Mat{T}}...) where {T} = cat(As...; dims=1)

"""
    Base.hcat(As::Union{Vec{T,Prefix},Mat{T,Prefix}}...) -> Mat{T,Prefix}

**MPI Collective**

Horizontally concatenate (place side-by-side) distributed PETSc vectors and/or matrices.

Equivalent to `cat(As...; dims=2)`. Concatenates inputs horizontally, increasing the number
of columns while keeping the number of rows constant.

# Requirements
All inputs must have the same number of rows and the same row partition.

# Prefix Selection
- **Automatically upgrades to `MPIDENSE`** when concatenating vectors (width > 1)
- Upgrades to `MPIDENSE` if any input is `MPIDENSE`
- Otherwise preserves the input `Prefix`

The automatic upgrade for vectors is important because vectors are inherently dense,
and horizontal concatenation of vectors produces a dense matrix.

# Examples
```julia
x = Vec_uniform([1.0, 2.0, 3.0])
y = Vec_uniform([4.0, 5.0, 6.0])
M = hcat(x, y)  # 3×2 Mat{Float64,MPIDENSE} - auto-upgraded!

A = Mat_uniform(sparse([1; 2; 3]))
B = Mat_uniform(sparse([4; 5; 6]))
C = hcat(A, B)  # 3×2 matrix with MPIDENSE
```

See also: [`cat`](@ref), [`vcat`](@ref)
"""
Base.hcat(As::Union{Vec{T},Mat{T}}...) where {T} = cat(As...; dims=2)

# Import blockdiag from SparseArrays
import SparseArrays: blockdiag

"""
    blockdiag(As::Mat{T,Prefix}...) -> Mat{T,Prefix}

**MPI Collective**

Create a block diagonal matrix from distributed PETSc matrices.

The result is a matrix with the input matrices along the diagonal and zeros elsewhere.
All matrices must have the same prefix and element type.

# Example
```julia
# If A is m×n and B is p×q, then blockdiag(A, B) is (m+p)×(n+q)
C = blockdiag(A, B)
```
"""
function blockdiag(As::Mat{T,Prefix}...) where {T,Prefix}
    n = length(As)
    if n == 0
        throw(ArgumentError("blockdiag requires at least one matrix"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Extract local sparse matrices (owned rows only)
    local_sparse = [_mat_to_local_sparse(A) for A in As]

    # Perform local blockdiag operation
    local_result = blockdiag(local_sparse...)

    # Compute result partitions using default partitioning
    total_rows = sum(size(A, 1) for A in As)
    total_cols = sum(size(A, 2) for A in As)
    result_row_partition = default_row_partition(total_rows, nranks)
    result_col_partition = default_row_partition(total_cols, nranks)

    # Use Mat_sum to combine results across ranks
    return Mat_sum(local_result;
                   row_partition=result_row_partition,
                   col_partition=result_col_partition,
                   Prefix=Prefix,
                   own_rank_only=false)
end

"""
    Base.:*(A::Mat{T,PrefixA}, B::Mat{T,PrefixB}) -> Mat{T,PrefixC}

**MPI Collective**

Multiply two distributed PETSc matrices using PETSc's MatMatMult.

Both matrices must have the same element type `T`, but can have different prefixes
(e.g., MPIAIJ × MPIDENSE is supported). The result prefix is determined by querying
what type PETSc actually creates.

The number of columns in A must match the number of rows in B.

# Example
```julia
C = A * B  # Matrix multiplication
```
"""
function Base.:*(A::Mat{T,PrefixA}, B::Mat{T,PrefixB}) where {T,PrefixA,PrefixB}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs - single @mpiassert for efficiency
    # Check: inner dimensions match, and inner partitions match
    @mpiassert (size(A, 2) == size(B, 1) &&
                A.obj.col_partition == B.obj.row_partition) "Matrix multiplication requires compatible dimensions (A cols must equal B rows) and matching inner partitions (A.col_partition must equal B.row_partition)"

    # Determine result partitions
    # Result has same row partition as A and same column partition as B
    result_row_partition = A.obj.row_partition
    result_col_partition = B.obj.col_partition

    # Use PETSc's MatMatMult - it returns both the matrix and the result Prefix
    C_mat, ResultPrefix = _mat_mat_mult(A.obj.A, B.obj.A)

    # Wrap in our _Mat type with the queried Prefix and return as DRef
    obj = _Mat{T,ResultPrefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, B::Mat{T,PrefixB}) -> Mat{T,PrefixC}

**MPI Collective**

Transpose-matrix multiplication using PETSc's MatTransposeMatMult.

Computes C = A' * B where A' is the transpose (adjoint for real matrices) of A.
Matrices can have different prefixes; the result prefix is determined by PETSc.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, B::Mat{T,PrefixB}) where {T,PrefixA,PrefixB}
    A = parent(At)::Mat{T,PrefixA}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B is m×p, result is n×p
    # A' has row partition = A's col partition, col partition = A's row partition
    @mpiassert (size(A, 1) == size(B, 1) &&
                A.obj.row_partition == B.obj.row_partition) "Transpose-matrix multiplication A'*B requires compatible dimensions (A rows must equal B rows for A'*B) and matching partitions (A.row_partition must equal B.row_partition)"

    # Result partitions: rows from A's columns, columns from B's columns
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.col_partition

    # Use PETSc's MatTransposeMatMult - it returns both the matrix and the result Prefix
    C_mat, ResultPrefix = _mat_transpose_mat_mult(A.obj.A, B.obj.A)

    # Wrap in our _Mat type with the queried Prefix and return as DRef
    obj = _Mat{T,ResultPrefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(A::Mat{T,PrefixA}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixB}}) -> Mat{T,PrefixC}

**MPI Collective**

Matrix-transpose multiplication using PETSc's MatMatTransposeMult.

Computes C = A * B' where B' is the transpose (adjoint for real matrices) of B.
Matrices can have different prefixes; the result prefix is determined by PETSc.
"""
function Base.:*(A::Mat{T,PrefixA}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixB}}) where {T,PrefixA,PrefixB}
    B = parent(Bt)::Mat{T,PrefixB}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A is m×n, B' is n×p (B is p×n), result is m×p
    # B' has row partition = B's col partition, col partition = B's row partition
    @mpiassert (size(A, 2) == size(B, 2) &&
                A.obj.col_partition == B.obj.col_partition) "Matrix-transpose multiplication A*B' requires compatible dimensions (A cols must equal B cols for A*B') and matching partitions (A.col_partition must equal B.col_partition)"

    # Result partitions: rows from A's rows, columns from B's rows
    result_row_partition = A.obj.row_partition
    result_col_partition = B.obj.row_partition

    # Use PETSc's MatMatTransposeMult - it returns both the matrix and the result Prefix
    C_mat, ResultPrefix = _mat_mat_transpose_mult(A.obj.A, B.obj.A)

    # Wrap in our _Mat type with the queried Prefix and return as DRef
    obj = _Mat{T,ResultPrefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixB}}) -> Mat{T,PrefixC}

**MPI Collective**

Transpose-transpose multiplication: C = A' * B'.

Since PETSc does not have a direct AtBt product type, this is computed as
C = (B * A)' by materializing the transpose of the result.
Matrices can have different prefixes; the result prefix is determined by PETSc.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixA}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,PrefixB}}) where {T,PrefixA,PrefixB}
    A = parent(At)::Mat{T,PrefixA}
    B = parent(Bt)::Mat{T,PrefixB}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B' is m×p (B is p×m), result is n×p
    @mpiassert (size(A, 1) == size(B, 2) &&
                A.obj.row_partition == B.obj.col_partition) "Transpose-transpose multiplication A'*B' requires compatible dimensions (A rows must equal B cols for A'*B') and matching partitions (A.row_partition must equal B.col_partition)"

    # Compute as (B * A)' since PETSc doesn't have AtBt product type
    # First compute BA - it returns both the matrix and its Prefix
    BA_mat, BAPrefix = _mat_mat_mult(B.obj.A, A.obj.A)

    # Then transpose it to get (BA)' = A'B'
    # BA has partitions (B.row_partition, A.col_partition), so transpose has swapped
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.row_partition
    C_mat, ResultPrefix = _mat_transpose(BA_mat)

    # Wrap in our _Mat type with the queried Prefix
    obj = _Mat{T,ResultPrefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.mul!(C::Mat{T,Prefix}, A::Mat{T,Prefix}, B::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

In-place matrix-matrix multiplication: C = A * B.

Reuses the pre-allocated matrix C. Dimensions and partitions must match appropriately.
"""
function LinearAlgebra.mul!(C::Mat{T,Prefix}, A::Mat{T,Prefix}, B::Mat{T,Prefix}) where {T,Prefix}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    @mpiassert (size(A, 2) == size(B, 1) &&
                size(A, 1) == size(C, 1) && size(B, 2) == size(C, 2) &&
                A.obj.col_partition == B.obj.row_partition &&
                A.obj.row_partition == C.obj.row_partition &&
                B.obj.col_partition == C.obj.col_partition) "Inner dimensions must match (A cols must equal B rows): A is $(size(A)), B is $(size(B)), output matrix C must have dimensions $(size(A, 1))×$(size(B, 2)) (got $(size(C))), matrix inner partitions must match (A.col_partition must equal B.row_partition), result row partition must match A's row partition, and result column partition must match B's column partition"

    # Perform in-place matrix-matrix multiplication using PETSc
    _mat_mat_mult!(C.obj.A, A.obj.A, B.obj.A)

    return C
end

# Scalar-matrix multiplication implemented using PETSc.@for_libpetsc
PETSc.@for_libpetsc begin
    """
        Base.:*(α::Number, A::Mat{T,Prefix}) -> Mat{T,Prefix}

    **MPI Collective**

    Scalar-matrix multiplication: α * A.

    Creates a new matrix with all entries scaled by α.
    """
    function Base.:*(α::Number, A::Mat{$PetscScalar,Prefix}) where {Prefix}
        # Duplicate the matrix
        result_mat_ref = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall(
            (:MatDuplicate, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
            A.obj.A,
            Cint(1),  # MAT_COPY_VALUES
            result_mat_ref
        )
        result_mat = PETSc.Mat{$PetscScalar}(result_mat_ref[])

        # Scale the duplicate
        PETSc.@chk ccall(
            (:MatScale, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, $PetscScalar),
            result_mat,
            $PetscScalar(α)
        )

        # Wrap in our _Mat type
        obj = _Mat{$PetscScalar,Prefix}(result_mat, copy(A.obj.row_partition), copy(A.obj.col_partition))
        return SafeMPI.DRef(obj)
    end

    """
        Base.:*(A::Mat{T,Prefix}, α::Number) -> Mat{T,Prefix}

    **MPI Collective**

    Matrix-scalar multiplication: A * α.

    Creates a new matrix with all entries scaled by α.
    """
    Base.:*(A::Mat{$PetscScalar,Prefix}, α::Number) where {Prefix} = α * A

    """
        Base.:+(α::Number, A::Mat{T,Prefix}) -> Mat{T,Prefix}

    **MPI Collective**

    Scalar-matrix addition: α + A.

    Adds scalar α to the diagonal of matrix A (equivalent to A + α*I).
    Requires A to be square.
    """
    function Base.:+(α::Number, A::Mat{$PetscScalar,Prefix}) where {Prefix}
        # For a square matrix, A + α*I means shift all diagonal elements by α
        # For non-square matrices, this operation is not well-defined mathematically
        # We'll implement it by creating a copy and using MatShift
        @mpiassert size(A, 1) == size(A, 2) "Scalar-matrix addition requires square matrix"

        # Duplicate the matrix
        result_mat_ref = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall(
            (:MatDuplicate, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
            A.obj.A,
            Cint(1),  # MAT_COPY_VALUES
            result_mat_ref
        )
        result_mat = PETSc.Mat{$PetscScalar}(result_mat_ref[])

        # Shift the diagonal (A := A + α*I)
        PETSc.@chk ccall(
            (:MatShift, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, $PetscScalar),
            result_mat,
            $PetscScalar(α)
        )

        # Wrap in our _Mat type
        obj = _Mat{$PetscScalar,Prefix}(result_mat, copy(A.obj.row_partition), copy(A.obj.col_partition))
        return SafeMPI.DRef(obj)
    end

    """
        Base.:+(A::Mat{T,Prefix}, α::Number) -> Mat{T,Prefix}

    **MPI Collective**

    Matrix-scalar addition: A + α.

    Adds scalar α to the diagonal of matrix A (equivalent to A + α*I).
    Requires A to be square.
    """
    Base.:+(A::Mat{$PetscScalar,Prefix}, α::Number) where {Prefix} = α + A

    """
        Base.:-(A::Mat{T,Prefix}, α::Number) -> Mat{T,Prefix}

    **MPI Collective**

    Matrix-scalar subtraction: A - α.

    Subtracts scalar α from the diagonal of matrix A (equivalent to A - α*I).
    Requires A to be square.
    """
    function Base.:-(A::Mat{$PetscScalar,Prefix}, α::Number) where {Prefix}
        @mpiassert size(A, 1) == size(A, 2) "Scalar-matrix subtraction requires square matrix"

        # Duplicate the matrix
        result_mat_ref = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall(
            (:MatDuplicate, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
            A.obj.A,
            Cint(1),  # MAT_COPY_VALUES
            result_mat_ref
        )
        result_mat = PETSc.Mat{$PetscScalar}(result_mat_ref[])

        # Shift the diagonal by -α (A := A - α*I)
        PETSc.@chk ccall(
            (:MatShift, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, $PetscScalar),
            result_mat,
            $PetscScalar(-α)
        )

        # Wrap in our _Mat type
        obj = _Mat{$PetscScalar,Prefix}(result_mat, copy(A.obj.row_partition), copy(A.obj.col_partition))
        return SafeMPI.DRef(obj)
    end

    """
        Base.:-(α::Number, A::Mat{T,Prefix}) -> Mat{T,Prefix}

    **MPI Collective**

    Scalar-matrix subtraction: α - A.

    Computes α*I - A (negates A and adds α to the diagonal).
    Requires A to be square.
    """
    function Base.:-(α::Number, A::Mat{$PetscScalar,Prefix}) where {Prefix}
        @mpiassert size(A, 1) == size(A, 2) "Scalar-matrix subtraction requires square matrix"

        # Duplicate the matrix
        result_mat_ref = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall(
            (:MatDuplicate, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
            A.obj.A,
            Cint(1),  # MAT_COPY_VALUES
            result_mat_ref
        )
        result_mat = PETSc.Mat{$PetscScalar}(result_mat_ref[])

        # Scale by -1 (negate the matrix)
        PETSc.@chk ccall(
            (:MatScale, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, $PetscScalar),
            result_mat,
            $PetscScalar(-1)
        )

        # Shift the diagonal by α (A := -A + α*I)
        PETSc.@chk ccall(
            (:MatShift, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, $PetscScalar),
            result_mat,
            $PetscScalar(α)
        )

        # Wrap in our _Mat type
        obj = _Mat{$PetscScalar,Prefix}(result_mat, copy(A.obj.row_partition), copy(A.obj.col_partition))
        return SafeMPI.DRef(obj)
    end
end

# UniformScaling operations (for working with LinearAlgebra.I)
using LinearAlgebra: UniformScaling

PETSc.@for_libpetsc begin
    """
        Base.:+(A::Mat{T,Prefix}, J::UniformScaling) -> Mat{T,Prefix}

    **MPI Collective**

    Matrix-UniformScaling addition: A + J.

    Adds J.λ to the diagonal of matrix A (equivalent to A + J.λ*I).
    Requires A to be square.
    """
    function Base.:+(A::Mat{$PetscScalar,Prefix}, J::UniformScaling) where {Prefix}
        return A + $PetscScalar(J.λ)
    end

    """
        Base.:+(J::UniformScaling, A::Mat{T,Prefix}) -> Mat{T,Prefix}

    **MPI Collective**

    UniformScaling-matrix addition: J + A.

    Adds J.λ to the diagonal of matrix A (equivalent to J.λ*I + A).
    Requires A to be square.
    """
    function Base.:+(J::UniformScaling, A::Mat{$PetscScalar,Prefix}) where {Prefix}
        return A + $PetscScalar(J.λ)
    end

    """
        Base.:-(A::Mat{T,Prefix}, J::UniformScaling) -> Mat{T,Prefix}

    **MPI Collective**

    Matrix-UniformScaling subtraction: A - J.

    Subtracts J.λ from the diagonal of matrix A (equivalent to A - J.λ*I).
    Requires A to be square.
    """
    function Base.:-(A::Mat{$PetscScalar,Prefix}, J::UniformScaling) where {Prefix}
        return A - $PetscScalar(J.λ)
    end

    """
        Base.:-(J::UniformScaling, A::Mat{T,Prefix}) -> Mat{T,Prefix}

    **MPI Collective**

    UniformScaling-matrix subtraction: J - A.

    Computes J.λ*I - A (negates A and adds J.λ to the diagonal).
    Requires A to be square.
    """
    function Base.:-(J::UniformScaling, A::Mat{$PetscScalar,Prefix}) where {Prefix}
        return $PetscScalar(J.λ) - A
    end
end

# Matrix norms using PETSc's MatNorm
using LinearAlgebra: norm

PETSc.@for_libpetsc begin
    # PETSc NormType constants (from petscsystypes.h)
    const NORM_1 = Cint(0)
    const NORM_FROBENIUS = Cint(2)
    const NORM_INFINITY = Cint(3)

    """
        LinearAlgebra.norm(A::Mat{T,Prefix}) -> T

    **MPI Collective**

    Compute the Frobenius norm of a distributed PETSc matrix: √(sum of squares of all entries).

    Note: This is the only matrix norm supported. For induced operator norms (max column/row sum),
    use `opnorm(A, p)` instead.

    # Examples
    ```julia
    A = Mat_uniform(sprand(10, 10, 0.3))
    norm(A)        # Frobenius norm
    ```
    """
    function LinearAlgebra.norm(A::Mat{$PetscScalar,Prefix}) where {Prefix}
        # Call PETSc's MatNorm with Frobenius norm
        nrm = Ref{$PetscReal}(0)
        PETSc.@chk ccall(
            (:MatNorm, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{$PetscReal}),
            A.obj.A,
            NORM_FROBENIUS,
            nrm
        )

        return $PetscScalar(nrm[])
    end

    # Catch-all to provide helpful error message for norm(A, p)
    function LinearAlgebra.norm(A::Mat{$PetscScalar,Prefix}, p::Real) where {Prefix}
        error("norm(A, p) is not supported for PETSc matrices. Use norm(A) for Frobenius norm or opnorm(A, p) for induced operator norms.")
    end

    """
        LinearAlgebra.opnorm(A::Mat{T,Prefix}, p::Real=2) -> T

    **MPI Collective**

    Compute the induced operator norm of a distributed PETSc matrix.

    Supported norms:
    - `p = 1`: 1-norm (maximum absolute column sum)
    - `p = 2`: Spectral norm (largest singular value) - not supported, use Frobenius as approximation
    - `p = Inf`: Infinity norm (maximum absolute row sum)

    # Examples
    ```julia
    A = Mat_uniform(sprand(10, 10, 0.3))
    opnorm(A, 1)     # Maximum absolute column sum
    opnorm(A, Inf)   # Maximum absolute row sum
    ```
    """
    function LinearAlgebra.opnorm(A::Mat{$PetscScalar,Prefix}, p::Real=2) where {Prefix}
        # Determine which PETSc norm type to use
        norm_type = if p == 1
            NORM_1
        elseif p == 2
            # Spectral norm (2-norm) requires expensive SVD computation
            # Fall back to Frobenius norm as an upper bound approximation
            NORM_FROBENIUS
        elseif isinf(p)
            NORM_INFINITY
        else
            error("Only p = 1, 2, or Inf are supported for operator norms")
        end

        # Call PETSc's MatNorm
        nrm = Ref{$PetscReal}(0)
        PETSc.@chk ccall(
            (:MatNorm, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, Cint, Ptr{$PetscReal}),
            A.obj.A,
            norm_type,
            nrm
        )

        return $PetscScalar(nrm[])
    end
end

# Helper function to determine Prefix type from PETSc MatType string
function _petsc_type_to_prefix(mat_type_str::String)
    # Map PETSc type strings to our Prefix types
    lowercase_type = lowercase(mat_type_str)
    if occursin("dense", lowercase_type)
        return MPIDENSE
    else
        # Default to MPIAIJ for sparse types (mpiaij, seqaij, etc.)
        return MPIAIJ
    end
end

# PETSc matrix transpose wrapper
PETSc.@for_libpetsc begin
    # Create new transposed matrix
    function _mat_transpose(A::PETSc.Mat{$PetscScalar})
        B = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall((:MatTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
                         A, MAT_INITIAL_MATRIX, B)
        Bmat = PETSc.Mat{$PetscScalar}(B[])

        # Query what type PETSc actually created
        mat_type_str = _mat_type_string(Bmat)
        ResultPrefix = _petsc_type_to_prefix(mat_type_str)

        # Set the options prefix based on the result type
        prefix_str = SafePETSc.prefix(ResultPrefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Bmat, prefix_str)
        end

        return Bmat, ResultPrefix
    end
end

# Note: _mat_transpose! is defined in ksp.jl with MatTransposeSetPrecursor support

# PETSc matrix-matrix multiplication wrapper
# Now queries the result type and returns both the matrix and its Prefix
PETSc.@for_libpetsc begin
    function _mat_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])

        # Query what type PETSc actually created
        mat_type_str = _mat_type_string(Cmat)
        ResultPrefix = _petsc_type_to_prefix(mat_type_str)

        # Set the options prefix based on the result type
        prefix_str = SafePETSc.prefix(ResultPrefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
        end

        return Cmat, ResultPrefix
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
    function _mat_transpose_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatTransposeMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])

        # Query what type PETSc actually created
        mat_type_str = _mat_type_string(Cmat)
        ResultPrefix = _petsc_type_to_prefix(mat_type_str)

        # Set the options prefix based on the result type
        prefix_str = SafePETSc.prefix(ResultPrefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
        end

        return Cmat, ResultPrefix
    end

    # Matrix-transpose multiplication: C = A * B'
    function _mat_mat_transpose_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatTransposeMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])

        # Query what type PETSc actually created
        mat_type_str = _mat_type_string(Cmat)
        ResultPrefix = _petsc_type_to_prefix(mat_type_str)

        # Set the options prefix based on the result type
        prefix_str = SafePETSc.prefix(ResultPrefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
        end

        return Cmat, ResultPrefix
    end
end

# Import spdiagm from SparseArrays
import SparseArrays: spdiagm

"""
    spdiagm(kv::Pair{<:Integer, <:Vec{T,Prefix}}...) -> Mat{T,Prefix}
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T,Prefix}}...) -> Mat{T,Prefix}

**MPI Collective**

Create a sparse diagonal matrix from distributed PETSc vectors.

Each pair `k => v` places the vector `v` on the `k`-th diagonal:
- `k = 0`: main diagonal
- `k > 0`: superdiagonal
- `k < 0`: subdiagonal

All vectors must have the same element type `T`. The matrix dimensions
are inferred from the diagonal positions and vector lengths, or can be specified explicitly.

# Optional Keyword Arguments
- `prefix`: Matrix prefix type to use for the result. Defaults to the input vector's prefix.
  Use this to create a matrix with a different prefix than the input vectors (e.g., create
  MPIAIJ from MPIDENSE vectors, or vice versa).
- `row_partition`: Override the default equal-row partitioning (length `nranks+1`, start at 1,
  end at `m+1`, non-decreasing). Defaults to `default_row_partition(m, nranks)`.
- `col_partition`: Override the default equal-column partitioning (length `nranks+1`, start at 1,
  end at `n+1`, non-decreasing). Defaults to `default_row_partition(n, nranks)`.

# Examples
```julia
# Create a tridiagonal matrix
A = spdiagm(-1 => lower, 0 => diag, 1 => upper)

# Create a 100×100 matrix with specified vectors on diagonals
B = spdiagm(100, 100, 0 => v1, 1 => v2)

# Create MPIAIJ (sparse) matrix from MPIDENSE vector
v_dense = Vec_uniform(data; Prefix=MPIDENSE)
A_sparse = spdiagm(0 => v_dense; prefix=MPIAIJ)
```
"""
function spdiagm(kv::Pair{<:Integer, <:Vec{T,Prefix}}...; prefix=Prefix, kwargs...) where {T,Prefix}
    if length(kv) == 0
        throw(ArgumentError("spdiagm requires at least one diagonal"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

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

    return spdiagm(m, n, kv...; prefix=prefix, kwargs...)
end

function spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T,Prefix}}...;
                 row_partition::Vector{Int}=default_row_partition(m, MPI.Comm_size(MPI.COMM_WORLD)),
                 col_partition::Vector{Int}=default_row_partition(n, MPI.Comm_size(MPI.COMM_WORLD)),
                 prefix=Prefix) where {T,Prefix}
    if length(kv) == 0
        throw(ArgumentError("spdiagm requires at least one diagonal"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate partitions
    # Note: We cannot infer partitions from diagonal vectors because the diagonals
    # are shifted by their offset k, so vector partitions don't map directly to
    # matrix row/column partitions. User can override with explicit partitions.
    row_partition_valid = length(row_partition) == nranks + 1 &&
                          row_partition[1] == 1 &&
                          row_partition[end] == m + 1 &&
                          all(r -> row_partition[r] <= row_partition[r+1], 1:nranks)
    col_partition_valid = length(col_partition) == nranks + 1 &&
                          col_partition[1] == 1 &&
                          col_partition[end] == n + 1 &&
                          all(c -> col_partition[c] <= col_partition[c+1], 1:nranks)

    # Validate vector lengths and partitions
    all_correct_lengths = all(begin
        required_len = k >= 0 ? min(m, n - k) : min(m + k, n)
        length(v) <= required_len  # allow shorter vectors; remaining entries are treated as zeros
    end for (k, v) in kv)
    @mpiassert (all_correct_lengths && row_partition_valid && col_partition_valid) "All vector lengths must not exceed the allowed diagonal span, and row_partition/col_partition must each have length nranks+1, start at 1, end at m+1/n+1 respectively, and be non-decreasing"

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
    row_lo = row_partition[rank+1]
    row_hi = row_partition[rank+2] - 1

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
    # Assemble result via Mat_sum (no pooling for concat operations)
    return Mat_sum(local_result;
                   row_partition=row_partition,
                   col_partition=col_partition,
                   Prefix=prefix,
                   own_rank_only=false)
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
# eachrow for distributed matrices (both MPIDENSE and MPIAIJ)
# - For dense matrices: uses MatDenseGetArrayRead and yields SubArray views
# - For sparse matrices: uses MatGetRow and yields SparseVector efficiently preserving sparsity
# - Performs exactly one MatDenseGetArrayRead per iterator (dense) and restores on finish
# -----------------------------------------------------------------------------

# Dense matrix iterator
mutable struct _EachRowDense{T,Prefix}
    aref::SafeMPI.DRef{_Mat{T,Prefix}}           # keep the matrix alive
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

    # MatGetRow - extract a single row from a sparse matrix
    # Returns (ncols, col_indices, values) where col_indices and values are pointers
    function _mat_get_row(A::PETSc.Mat{$PetscScalar}, row::Int)
        row_c = $PetscInt(row - 1)  # Convert to 0-based
        ncols = Ref{$PetscInt}(0)
        cols_ptr = Ref{Ptr{$PetscInt}}(C_NULL)
        vals_ptr = Ref{Ptr{$PetscScalar}}(C_NULL)
        PETSc.@chk ccall((:MatGetRow, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, $PetscInt, Ptr{$PetscInt}, Ptr{Ptr{$PetscInt}}, Ptr{Ptr{$PetscScalar}}),
                         A, row_c, ncols, cols_ptr, vals_ptr)
        return (Int(ncols[]), cols_ptr[], vals_ptr[])
    end

    # MatRestoreRow - must be called after MatGetRow to free resources
    function _mat_restore_row(A::PETSc.Mat{$PetscScalar}, row::Int, ncols::Int, cols_ptr::Ptr{$PetscInt}, vals_ptr::Ptr{$PetscScalar})
        row_c = $PetscInt(row - 1)  # Convert to 0-based
        ncols_c = $PetscInt(ncols)
        cols_ref = Ref{Ptr{$PetscInt}}(cols_ptr)
        vals_ref = Ref{Ptr{$PetscScalar}}(vals_ptr)
        PETSc.@chk ccall((:MatRestoreRow, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, $PetscInt, Ptr{$PetscInt}, Ptr{Ptr{$PetscInt}}, Ptr{Ptr{$PetscScalar}}),
                         A, row_c, Ref(ncols_c), cols_ref, vals_ref)
        return nothing
    end
end

function _eachrow_dense(A::Mat{T,Prefix}) where {T,Prefix}
    # Determine local row range from stored partition
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nloc = row_hi - row_lo + 1
    ncols = size(A, 2)

    # Acquire read-only dense local array from PETSc and wrap as a Julia Matrix
    p = _matdense_get_array_read(A.obj.A)
    data = unsafe_wrap(Array, p, (nloc, ncols); own=false)

    it = _EachRowDense{T,Prefix}(A, A.obj.A, row_lo, nloc, ncols, data, p)
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

Base.IteratorEltype(::Type{_EachRowDense{T,Prefix}}) where {T,Prefix} = Base.HasEltype()
Base.eltype(::Type{_EachRowDense{T,Prefix}}) where {T,Prefix} = SubArray{T,1,Matrix{T},Tuple{Int,Base.Slice{Base.OneTo{Int}}},true}
Base.IteratorSize(::Type{_EachRowDense{T,Prefix}}) where {T,Prefix} = Base.HasLength()
Base.length(it::_EachRowDense) = it.nloc

function Base.iterate(it::_EachRowDense{T,Prefix}) where {T,Prefix}
    it.nloc == 0 && return nothing
    row = @view it.data[1, :]
    return (row, 1)
end

function Base.iterate(it::_EachRowDense{T,Prefix}, st::Int) where {T,Prefix}
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

# -----------------------------------------------------------------------------
# Sparse matrix iterator (MPIAIJ)
# -----------------------------------------------------------------------------

mutable struct _EachRowSparse{T,Prefix}
    aref::SafeMPI.DRef{_Mat{T,Prefix}}     # keep the matrix alive
    petscA::PETSc.Mat{T}                   # underlying PETSc Mat
    row_lo::Int                            # global start row (1-based)
    nloc::Int                              # number of local rows
    ncols::Int                             # global number of columns
end

function _eachrow_sparse(A::Mat{T,Prefix}) where {T,Prefix}
    # Determine local row range from stored partition
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nloc = row_hi - row_lo + 1
    ncols = size(A, 2)

    return _EachRowSparse{T,Prefix}(A, A.obj.A, row_lo, nloc, ncols)
end

Base.IteratorEltype(::Type{_EachRowSparse{T,Prefix}}) where {T,Prefix} = Base.HasEltype()
Base.eltype(::Type{_EachRowSparse{T,Prefix}}) where {T,Prefix} = SparseVector{T,Int}
Base.IteratorSize(::Type{_EachRowSparse{T,Prefix}}) where {T,Prefix} = Base.HasLength()
Base.length(it::_EachRowSparse) = it.nloc

function Base.iterate(it::_EachRowSparse{T,Prefix}) where {T,Prefix}
    it.nloc == 0 && return nothing
    return _iterate_sparse_row(it, 0)
end

function Base.iterate(it::_EachRowSparse{T,Prefix}, st::Int) where {T,Prefix}
    i = st + 1
    i > it.nloc && return nothing
    return _iterate_sparse_row(it, st)
end

function _iterate_sparse_row(it::_EachRowSparse{T,Prefix}, st::Int) where {T,Prefix}
    i = st + 1
    global_row = it.row_lo + i - 1

    # Get the sparse row from PETSc
    ncols_row, cols_ptr, vals_ptr = _mat_get_row(it.petscA, global_row)

    # Create a sparse vector directly from the non-zero entries
    if ncols_row > 0
        # Wrap the PETSc arrays (we must copy since we'll restore them)
        cols = unsafe_wrap(Array, cols_ptr, ncols_row; own=false)
        vals = unsafe_wrap(Array, vals_ptr, ncols_row; own=false)

        # Copy to Julia arrays and convert to 1-based indexing
        nzind = [Int(cols[j] + 1) for j in 1:ncols_row]
        nzval = copy(vals)

        # Restore the row data to PETSc before constructing SparseVector
        _mat_restore_row(it.petscA, global_row, ncols_row, cols_ptr, vals_ptr)

        # Construct sparse vector
        row_vec = SparseVector(it.ncols, nzind, nzval)
    else
        # Restore the row data to PETSc
        _mat_restore_row(it.petscA, global_row, ncols_row, cols_ptr, vals_ptr)

        # Empty row - create empty sparse vector
        row_vec = spzeros(T, it.ncols)
    end

    return (row_vec, i)
end

"""
    Base.eachrow(A::Mat{T,Prefix}) -> Iterator

**MPI Non-Collective**

Iterate over the rows of a distributed matrix.

Only iterates over rows owned by the current rank. Each row is returned as a view (for dense)
or a SparseVector (for sparse matrices).

For MPIDENSE matrices: returns views of the underlying dense storage
For MPIAIJ matrices: returns SparseVector{T,Int} efficiently preserving sparsity
"""
function Base.eachrow(A::Mat{T,MPIDENSE}) where {T}
    return _eachrow_dense(A)
end

function Base.eachrow(A::Mat{T,MPIAIJ}) where {T}
    return _eachrow_sparse(A)
end

# -----------------------------------------------------------------------------
# Column Extraction
# -----------------------------------------------------------------------------

"""
    Base.getindex(A::Mat{T,Prefix}, ::Colon, k::Int) -> Vec{T,Prefix}

**MPI Non-Collective**

Extract column k from matrix A, returning a distributed vector.

Each rank extracts its owned rows from column k. The resulting vector has the same
row partition as matrix A.

Uses efficient bulk operations: MatDenseGetArrayRead for dense matrices,
MatGetRow for sparse matrices.

# Example
```julia
A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0])
v = A[:, 2]  # Extract second column: [2.0, 5.0]
```
"""
function Base.getindex(A::DRef{_Mat{T,Prefix}}, ::Colon, k::Int) where {T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    m, n = size(A)

    # Check bounds
    @assert 1 <= k <= n "Column index $k out of bounds for matrix of size $(m)×$(n)"

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    # Extract local portion of column k using efficient methods
    local_data = Vector{T}(undef, nlocal_rows)

    if is_dense(A)
        # Dense matrix: get entire local block and extract column
        p = _matdense_get_array_read(A.obj.A)
        try
            local_block = unsafe_wrap(Array, p, (nlocal_rows, n); own=false)
            @inbounds local_data[:] = @view local_block[:, k]
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
    else
        # Sparse matrix: use MatGetRow for each row and extract column k
        for i in 1:nlocal_rows
            global_row = row_lo + i - 1
            ncols_row, cols_ptr, vals_ptr = _mat_get_row(A.obj.A, global_row)

            # Find column k in this row (PETSc uses 0-based indexing)
            local_data[i] = zero(T)  # Default value
            if ncols_row > 0
                cols = unsafe_wrap(Array, cols_ptr, ncols_row; own=false)
                vals = unsafe_wrap(Array, vals_ptr, ncols_row; own=false)
                for j in 1:ncols_row
                    if Int(cols[j]) + 1 == k
                        local_data[i] = vals[j]
                        break
                    end
                end
            end

            _mat_restore_row(A.obj.A, global_row, ncols_row, cols_ptr, vals_ptr)
        end
    end

    # Create distributed PETSc Vec
    petsc_vec = _vec_create_mpi_for_T(T, nlocal_rows, m, Prefix, A.obj.row_partition)

    # Fill local portion
    local_view = PETSc.unsafe_localarray(petsc_vec; read=true, write=true)
    try
        @inbounds local_view[:] = local_data
    finally
        Base.finalize(local_view)
    end
    PETSc.assemble(petsc_vec)

    # Wrap and return
    obj = _Vec{T,Prefix}(petsc_vec, A.obj.row_partition)
    return SafeMPI.DRef(obj)
end

"""
    own_row(A::Mat{T,Prefix}) -> UnitRange{Int}

**MPI Non-Collective**

Return the range of row indices owned by the current rank for matrix A.

# Example
```julia
A = Mat_uniform([1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0])
range = own_row(A)  # e.g., 1:2 on rank 0
```
"""
own_row(A::DRef{_Mat{T,Prefix}}) where {T,Prefix} = A.obj.row_partition[MPI.Comm_rank(MPI.COMM_WORLD)+1]:(A.obj.row_partition[MPI.Comm_rank(MPI.COMM_WORLD)+2]-1)

"""
    Base.getindex(A::Mat{T}, i::Int, j::Int) -> T

**MPI Non-Collective**

Get the value at position (i, j) from a distributed matrix.

The row index i must be wholly contained in the current rank's row ownership range.
If not, the function will abort with an error message and stack trace.

This is a non-collective operation.

# Example
```julia
A = Mat_uniform([1.0 2.0; 3.0 4.0])
# On rank that owns row 1:
val = A[1, 2]  # Returns 2.0
```
"""
function Base.getindex(A::DRef{_Mat{T,Prefix}}, i::Int, j::Int) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    m, n = size(A)

    # Check bounds
    @assert 1 <= i <= m && 1 <= j <= n "Indices ($i, $j) out of bounds for matrix of size $(m)×$(n)"

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # Check that row index is in local range (non-collective check)
    if !(row_lo <= i <= row_hi)
        SafeMPI.mpierror("Row index $i must be in rank $rank's ownership range [$row_lo, $row_hi]", true)
    end

    # Use different methods for dense vs sparse
    if is_dense(A)
        # Use MatDenseGetArrayRead for dense matrices
        p = _matdense_get_array_read(A.obj.A)
        try
            nlocal = row_hi - row_lo + 1
            local_data = unsafe_wrap(Array, p, (nlocal, n); own=false)
            local_row = i - row_lo + 1
            return local_data[local_row, j]
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
    else
        # For sparse matrices, use MatGetRow/MatRestoreRow
        ncols, cols, vals = _mat_get_row(A.obj.A, i)
        try
            if ncols > 0
                cols_arr = unsafe_wrap(Array, cols, ncols; own=false)
                vals_arr = unsafe_wrap(Array, vals, ncols; own=false)
                # Find column j in the row (PETSc uses 0-based indexing)
                for k in 1:ncols
                    if Int(cols_arr[k]) + 1 == j
                        return vals_arr[k]
                    end
                end
            end
            return zero(T)
        finally
            _mat_restore_row(A.obj.A, i, ncols, cols, vals)
        end
    end
end

"""
    Base.getindex(A::Mat{T}, range_i::UnitRange{Int}, j::Int) -> Vector{T}

**MPI Non-Collective**

Extract a contiguous range of rows from column j of a distributed matrix.

The row range must be wholly contained in the current rank's row ownership range.
If not, the function will abort with an error message and stack trace.

This is a non-collective operation.

# Example
```julia
A = Mat_uniform([1.0 2.0; 3.0 4.0; 5.0 6.0])
# On rank that owns rows 1:2:
vals = A[1:2, 2]  # Returns [2.0, 4.0]
```
"""
function Base.getindex(A::DRef{_Mat{T,Prefix}}, range_i::UnitRange{Int}, j::Int) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    m, n = size(A)

    # Check bounds
    @assert 1 <= first(range_i) && last(range_i) <= m && 1 <= j <= n "Indices ($range_i, $j) out of bounds for matrix of size $(m)×$(n)"

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # Check that row range is in local range (non-collective check)
    if !(row_lo <= first(range_i) && last(range_i) <= row_hi)
        SafeMPI.mpierror("Row range $range_i must be wholly contained in rank $rank's ownership range [$row_lo, $row_hi]", true)
    end

    # Extract values using local array access
    nrows = length(range_i)
    vals = Vector{T}(undef, nrows)

    if is_dense(A)
        # Use MatDenseGetArrayRead for dense matrices
        p = _matdense_get_array_read(A.obj.A)
        try
            nlocal = row_hi - row_lo + 1
            local_data = unsafe_wrap(Array, p, (nlocal, n); own=false)
            for (idx, i) in enumerate(range_i)
                local_row = i - row_lo + 1
                vals[idx] = local_data[local_row, j]
            end
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
    else
        # Use MatGetRow for sparse matrices
        for (idx, i) in enumerate(range_i)
            ncols, cols_ptr, vals_ptr = _mat_get_row(A.obj.A, i)
            try
                val = zero(T)
                if ncols > 0
                    cols = unsafe_wrap(Array, cols_ptr, ncols; own=false)
                    vals_row = unsafe_wrap(Array, vals_ptr, ncols; own=false)
                    for k in 1:ncols
                        if Int(cols[k]) + 1 == j
                            val = vals_row[k]
                            break
                        end
                    end
                end
                vals[idx] = val
            finally
                _mat_restore_row(A.obj.A, i, ncols, cols_ptr, vals_ptr)
            end
        end
    end

    return vals
end

"""
    Base.getindex(A::Mat{T}, i::Int, range_j::UnitRange{Int}) -> Vector{T}

**MPI Non-Collective**

Extract a contiguous range of columns from row i of a distributed matrix.

The row index i must be wholly contained in the current rank's row ownership range.
If not, the function will abort with an error message and stack trace.

This is a non-collective operation.

# Example
```julia
A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0])
# On rank that owns row 1:
vals = A[1, 2:3]  # Returns [2.0, 3.0]
```
"""
function Base.getindex(A::DRef{_Mat{T,Prefix}}, i::Int, range_j::UnitRange{Int}) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    m, n = size(A)

    # Check bounds
    @assert 1 <= i <= m && 1 <= first(range_j) && last(range_j) <= n "Indices ($i, $range_j) out of bounds for matrix of size $(m)×$(n)"

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # Check that row index is in local range (non-collective check)
    if !(row_lo <= i <= row_hi)
        SafeMPI.mpierror("Row index $i must be in rank $rank's ownership range [$row_lo, $row_hi]", true)
    end

    # Extract values using local array access
    ncols = length(range_j)
    vals = Vector{T}(undef, ncols)

    if is_dense(A)
        # Use MatDenseGetArrayRead for dense matrices
        p = _matdense_get_array_read(A.obj.A)
        try
            nlocal = row_hi - row_lo + 1
            local_data = unsafe_wrap(Array, p, (nlocal, n); own=false)
            local_row = i - row_lo + 1
            for (idx, j) in enumerate(range_j)
                vals[idx] = local_data[local_row, j]
            end
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
    else
        # Use MatGetRow for sparse matrices
        ncols_row, cols_ptr, vals_ptr = _mat_get_row(A.obj.A, i)
        try
            # Initialize with zeros
            vals .= zero(T)
            if ncols_row > 0
                cols = unsafe_wrap(Array, cols_ptr, ncols_row; own=false)
                vals_row = unsafe_wrap(Array, vals_ptr, ncols_row; own=false)
                # Match requested columns with row data
                for k in 1:ncols_row
                    col = Int(cols[k]) + 1  # Convert to 1-based
                    if col in range_j
                        idx = col - first(range_j) + 1
                        vals[idx] = vals_row[k]
                    end
                end
            end
        finally
            _mat_restore_row(A.obj.A, i, ncols_row, cols_ptr, vals_ptr)
        end
    end

    return vals
end

"""
    Base.getindex(A::Mat{T}, range_i::UnitRange{Int}, range_j::UnitRange{Int}) -> Union{Matrix{T}, SparseMatrixCSC{T}}

**MPI Non-Collective**

Extract a submatrix from a distributed matrix.

The row range must be wholly contained in the current rank's row ownership range.
If not, the function will abort with an error message and stack trace.

Returns a dense Matrix if A is dense, otherwise returns a SparseMatrixCSC.

This is a non-collective operation.

# Example
```julia
A = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])
# On rank that owns rows 1:2:
submat = A[1:2, 2:3]  # Returns [2.0 3.0; 5.0 6.0]
```
"""
function Base.getindex(A::DRef{_Mat{T,Prefix}}, range_i::UnitRange{Int}, range_j::UnitRange{Int}) where {T,Prefix}
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    m, n = size(A)

    # Check bounds
    @assert 1 <= first(range_i) && last(range_i) <= m && 1 <= first(range_j) && last(range_j) <= n "Indices ($range_i, $range_j) out of bounds for matrix of size $(m)×$(n)"

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # Check that row range is in local range (non-collective check)
    if !(row_lo <= first(range_i) && last(range_i) <= row_hi)
        SafeMPI.mpierror("Row range $range_i must be wholly contained in rank $rank's ownership range [$row_lo, $row_hi]", true)
    end

    # Extract values using local array access
    nrows = length(range_i)
    ncols = length(range_j)

    if is_dense(A)
        # Return dense matrix
        result = Matrix{T}(undef, nrows, ncols)
        p = _matdense_get_array_read(A.obj.A)
        try
            nlocal = row_hi - row_lo + 1
            local_data = unsafe_wrap(Array, p, (nlocal, n); own=false)
            for (row_idx, i) in enumerate(range_i)
                local_row = i - row_lo + 1
                for (col_idx, j) in enumerate(range_j)
                    result[row_idx, col_idx] = local_data[local_row, j]
                end
            end
        finally
            _matdense_restore_array_read(A.obj.A, p)
        end
        return result
    else
        # Return sparse matrix
        I = Int[]
        J = Int[]
        V = T[]
        for (row_idx, i) in enumerate(range_i)
            ncols_row, cols_ptr, vals_ptr = _mat_get_row(A.obj.A, i)
            try
                if ncols_row > 0
                    cols = unsafe_wrap(Array, cols_ptr, ncols_row; own=false)
                    vals_row = unsafe_wrap(Array, vals_ptr, ncols_row; own=false)
                    for k in 1:ncols_row
                        col = Int(cols[k]) + 1  # Convert to 1-based
                        if col in range_j
                            col_idx = col - first(range_j) + 1
                            push!(I, row_idx)
                            push!(J, col_idx)
                            push!(V, vals_row[k])
                        end
                    end
                end
            finally
                _mat_restore_row(A.obj.A, i, ncols_row, cols_ptr, vals_ptr)
            end
        end
        return sparse(I, J, V, nrows, ncols)
    end
end

# -----------------------------------------------------------------------------
# Matrix Type Query and Conversion to Julia Arrays
# -----------------------------------------------------------------------------

# Get PETSc MatType string
PETSc.@for_libpetsc begin
    function _mat_type_string(A::PETSc.Mat{$PetscScalar})
        type_ptr = Ref{Ptr{Cchar}}(C_NULL)
        PETSc.@chk ccall((:MatGetType, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, Ptr{Ptr{Cchar}}), A, type_ptr)
        return unsafe_string(type_ptr[])
    end
end

"""
    is_dense(x::Mat{T,Prefix}) -> Bool

**MPI Non-Collective**

Check if a PETSc matrix is a dense matrix type.

This checks the PETSc matrix type string and returns true if it contains "dense"
(case-insensitive). This handles various dense types like "seqdense", "mpidense",
and vendor-specific dense matrix types.
"""
function is_dense(x::Mat{T,Prefix}) where {T,Prefix}
    mat_type = _mat_type_string(x.obj.A)
    return occursin("dense", lowercase(mat_type))
end

"""
    Matrix(x::Mat{T,Prefix}) -> Matrix{T}

**MPI Collective**

Convert a distributed PETSc Mat to a Julia Matrix by gathering all data to all ranks.
This is a collective operation - all ranks must call it and will receive the complete matrix.

For dense matrices, this uses efficient MatDenseGetArrayRead. For other matrix types,
it uses MatGetRow to extract each row.

This is primarily used for display purposes or small matrices. For large matrices, this
operation can be expensive as it gathers all data to all ranks.
"""
function Base.Matrix(x::Mat{T,Prefix}) where {T,Prefix}
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    m, n = size(x)
    row_lo = x.obj.row_partition[rank+1]
    row_hi = x.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    # Extract local rows as dense matrix
    if is_dense(x)
        # Efficient dense extraction
        p = _matdense_get_array_read(x.obj.A)
        try
            # PETSc stores dense matrices column-major, same as Julia
            local_data = unsafe_wrap(Array, p, (nlocal, n); own=false)
            local_dense = copy(local_data)  # Copy to own the data
        finally
            _matdense_restore_array_read(x.obj.A, p)
        end
    else
        # General case: use MatGetRow
        local_dense = Matrix{T}(undef, nlocal, n)
        for i in 1:nlocal
            global_row = row_lo + i - 1

            # Get row from PETSc
            ncols, cols_ptr, vals_ptr = _mat_get_row(x.obj.A, global_row)
            try
                # Convert to Julia arrays (PETSc uses 0-based indexing)
                if ncols > 0
                    cols = unsafe_wrap(Array, cols_ptr, ncols; own=false)
                    vals = unsafe_wrap(Array, vals_ptr, ncols; own=false)

                    # Fill row with zeros and set nonzero values
                    local_dense[i, :] .= zero(T)
                    for j in 1:ncols
                        col_idx = Int(cols[j]) + 1  # Convert to 1-based
                        local_dense[i, col_idx] = vals[j]
                    end
                else
                    # Empty row
                    local_dense[i, :] .= zero(T)
                end
            finally
                _mat_restore_row(x.obj.A, global_row, ncols, cols_ptr, vals_ptr)
            end
        end
    end

    # Prepare for Allgatherv (gather rows)
    counts = [x.obj.row_partition[i+2] - x.obj.row_partition[i+1] for i in 0:nranks-1]
    counts_elems = counts .* n  # Total elements per rank
    displs_elems = [sum(counts_elems[1:i]) for i in 1:nranks]
    pushfirst!(displs_elems, 0)
    pop!(displs_elems)

    # Gather to all ranks
    local_flat = Vector{T}(vec(local_dense'))  # Flatten column-major, ensure plain Vector
    result_flat = Vector{T}(undef, m * n)
    MPI.Allgatherv!(local_flat, MPI.VBuffer(result_flat, counts_elems, displs_elems), comm)

    # Reshape back to matrix (column-major)
    result = reshape(result_flat, n, m)'  # Transpose because we flattened row-wise
    return Matrix{T}(result)  # Ensure it's a plain Matrix, not an adjoint
end

"""
    SparseArrays.sparse(x::Mat{T,Prefix}) -> SparseMatrixCSC{T, Int}

**MPI Collective**

Convert a distributed PETSc Mat to a Julia SparseMatrixCSC by gathering all data to all ranks.
This is a collective operation - all ranks must call it and will receive the complete sparse matrix.

Uses MatGetRow to extract the sparse structure efficiently, preserving sparsity.

This is primarily used for display purposes or small matrices. For large matrices, this
operation can be expensive as it gathers all data to all ranks.
"""
function SparseArrays.sparse(x::Mat{T,Prefix}) where {T,Prefix}
    comm = MPI.COMM_WORLD
    nranks = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    m, n = size(x)
    row_lo = x.obj.row_partition[rank+1]
    row_hi = x.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    # Extract local sparse data
    local_I = Int[]
    local_J = Int[]
    local_V = T[]

    for i in 1:nlocal
        global_row = row_lo + i - 1

        # Get row from PETSc
        ncols, cols_ptr, vals_ptr = _mat_get_row(x.obj.A, global_row)
        try
            if ncols > 0
                # Convert to Julia arrays (PETSc uses 0-based indexing)
                cols = unsafe_wrap(Array, cols_ptr, ncols; own=false)
                vals = unsafe_wrap(Array, vals_ptr, ncols; own=false)

                # Add to local sparse arrays
                for j in 1:ncols
                    col_idx = Int(cols[j]) + 1  # Convert to 1-based
                    push!(local_I, global_row)
                    push!(local_J, col_idx)
                    push!(local_V, vals[j])
                end
            end
        finally
            _mat_restore_row(x.obj.A, global_row, ncols, cols_ptr, vals_ptr)
        end
    end

    # Gather counts from all ranks
    local_nnz = length(local_I)
    all_nnz = MPI.Allgather(local_nnz, comm)

    # Prepare for Allgatherv
    total_nnz = sum(all_nnz)
    displs = [sum(all_nnz[1:i]) for i in 1:nranks]
    pushfirst!(displs, 0)
    pop!(displs)

    # Gather I, J, V arrays
    global_I = Vector{Int}(undef, total_nnz)
    global_J = Vector{Int}(undef, total_nnz)
    global_V = Vector{T}(undef, total_nnz)

    MPI.Allgatherv!(local_I, MPI.VBuffer(global_I, all_nnz, displs), comm)
    MPI.Allgatherv!(local_J, MPI.VBuffer(global_J, all_nnz, displs), comm)
    MPI.Allgatherv!(local_V, MPI.VBuffer(global_V, all_nnz, displs), comm)

    # Create sparse matrix
    return sparse(global_I, global_J, global_V, m, n)
end

"""
    Base.show(io::IO, x::Mat{T,Prefix})

**MPI Collective**

Display a distributed PETSc Mat by converting to Julia Matrix or SparseMatrixCSC and showing it.

Dense matrices are converted to Matrix, other matrices are converted to SparseMatrixCSC.
This is a collective operation - all ranks must call it and will display the same matrix.
To print only on rank 0, use: `println(io0(), A)`
"""
Base.show(io::IO, x::Mat{T,Prefix}) where {T,Prefix} = show(io, is_dense(x) ? Matrix(x) : sparse(x))

"""
    Base.show(io::IO, mime::MIME, x::Mat{T,Prefix})

**MPI Collective**

Display a distributed PETSc Mat with a specific MIME type by converting to Julia Matrix or SparseMatrixCSC.

Dense matrices are converted to Matrix, other matrices are converted to SparseMatrixCSC.
This is a collective operation - all ranks must call it and will display the same matrix.
To print only on rank 0, use: `show(io0(), mime, A)`
"""
Base.show(io::IO, mime::MIME, x::Mat{T,Prefix}) where {T,Prefix} = show(io, mime, is_dense(x) ? Matrix(x) : sparse(x))

