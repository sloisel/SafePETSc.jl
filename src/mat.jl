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

# Adjoint support for Mat
Base.adjoint(A::Mat{T}) where {T} = LinearAlgebra.Adjoint(A)

"""
    Mat(adj::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) -> Mat{T,Prefix}

**MPI Collective**

Materialize an adjoint matrix into a new transposed Mat.

Creates a new matrix B = A^T using PETSc's transpose operations.
"""
function Mat(adj::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) where {T,Prefix}
    A = parent(adj)::Mat{T,Prefix}
    # Create B = A^T via PETSc (MAT_INITIAL_MATRIX) and preserve prefix
    B_petsc = _mat_transpose(A.obj.A, Prefix)
    # Swap row/col partitions for the transpose
    obj = _Mat{T,Prefix}(B_petsc, A.obj.col_partition, A.obj.row_partition)
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
    Base.:*(A::Mat{T}, x::Vec{T}) -> Vec{T}

**MPI Collective**

Matrix-vector multiplication: y = A * x.

Returns a new distributed vector with the result.
"""
function Base.:*(A::Mat{T,PrefixA}, x::Vec{T,PrefixX}) where {T,PrefixA,PrefixX}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(x)[1]
    @mpiassert n == vec_length && A.obj.col_partition == x.obj.row_partition "Matrix columns must match vector length (A: $(m)×$(n), x: $(vec_length)) and column partition of A must match row partition of x"

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
    LinearAlgebra.mul!(y::Vec{T}, A::Mat{T}, x::Vec{T}) -> Vec{T}

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
    Base.:-(A::Mat{T}, B::Mat{T}) -> Mat{T}

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
function Base.cat(As::Mat{T,Prefix}...; dims) where {T,Prefix}
    n = length(As)
    if n == 0
        throw(ArgumentError("cat requires at least one matrix"))
    end

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate partition compatibility based on dims - collect conditions first
    if dims == 1  # Vertical concatenation
        # All matrices must have same column partition and number of columns
        first_col_partition = As[1].obj.col_partition
        first_ncols = size(As[1], 2)
        all_same_col_partition = all(As[k].obj.col_partition == first_col_partition for k in 2:n)
        all_same_ncols = all(size(As[k], 2) == first_ncols for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        @mpiassert (all_same_col_partition && all_same_ncols) "For dims=1: all matrices must have the same column partition and number of columns"
    elseif dims == 2  # Horizontal concatenation
        # All matrices must have same row partition and number of rows
        first_row_partition = As[1].obj.row_partition
        first_nrows = size(As[1], 1)
        all_same_row_partition = all(As[k].obj.row_partition == first_row_partition for k in 2:n)
        all_same_nrows = all(size(As[k], 1) == first_nrows for k in 2:n)
        # Single @mpiassert for efficiency (pulled out of loops)
        @mpiassert (all_same_row_partition && all_same_nrows) "For dims=2: all matrices must have the same row partition and number of rows"
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
                   Prefix=Prefix,
                   own_rank_only=false)
end

"""
    Base.vcat(As::Mat{T}...) -> Mat{T}

**MPI Collective**

Vertically concatenate distributed PETSc matrices.

Equivalent to `cat(As...; dims=1)`. All matrices must have the same number of columns
and the same column partition.
"""
Base.vcat(As::Mat{T,Prefix}...) where {T,Prefix} = cat(As...; dims=1)

"""
    Base.hcat(As::Mat{T}...) -> Mat{T}

**MPI Collective**

Horizontally concatenate distributed PETSc matrices.

Equivalent to `cat(As...; dims=2)`. All matrices must have the same number of rows
and the same row partition.
"""
Base.hcat(As::Mat{T,Prefix}...) where {T,Prefix} = cat(As...; dims=2)

# Import blockdiag from SparseArrays
import SparseArrays: blockdiag

"""
    blockdiag(As::Mat{T}...) -> Mat{T}

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
                   Prefix=Prefix,
                   own_rank_only=false)
end

"""
    Base.:*(A::Mat{T}, B::Mat{T}) -> Mat{T}

**MPI Collective**

Multiply two distributed PETSc matrices using PETSc's MatMatMult.

Both matrices must have the same element type `T` and the same prefix.
The number of columns in A must match the number of rows in B.

# Example
```julia
C = A * B  # Matrix multiplication
```
"""
function Base.:*(A::Mat{T,Prefix}, B::Mat{T,Prefix}) where {T,Prefix}
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

    # Use PETSc's MatMatMult
    C_mat = _mat_mat_mult(A.obj.A, B.obj.A, Prefix)

    # Wrap in our _Mat type and return as DRef
    obj = _Mat{T,Prefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, B::Mat{T}) -> Mat{T}

**MPI Collective**

Transpose-matrix multiplication using PETSc's MatTransposeMatMult.

Computes C = A' * B where A' is the transpose (adjoint for real matrices) of A.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}, B::Mat{T,Prefix}) where {T,Prefix}
    A = parent(At)::Mat{T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B is m×p, result is n×p
    # A' has row partition = A's col partition, col partition = A's row partition
    @mpiassert (size(A, 1) == size(B, 1) &&
                A.obj.row_partition == B.obj.row_partition) "Transpose-matrix multiplication A'*B requires compatible dimensions (A rows must equal B rows for A'*B) and matching partitions (A.row_partition must equal B.row_partition)"

    # Result partitions: rows from A's columns, columns from B's columns
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.col_partition

    # Use PETSc's MatTransposeMatMult
    C_mat = _mat_transpose_mat_mult(A.obj.A, B.obj.A, Prefix)

    # Wrap in our _Mat type and return as DRef
    obj = _Mat{T,Prefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(A::Mat{T}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) -> Mat{T}

**MPI Collective**

Matrix-transpose multiplication using PETSc's MatMatTransposeMult.

Computes C = A * B' where B' is the transpose (adjoint for real matrices) of B.
"""
function Base.:*(A::Mat{T,Prefix}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) where {T,Prefix}
    B = parent(Bt)::Mat{T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A is m×n, B' is n×p (B is p×n), result is m×p
    # B' has row partition = B's col partition, col partition = B's row partition
    @mpiassert (size(A, 2) == size(B, 2) &&
                A.obj.col_partition == B.obj.col_partition) "Matrix-transpose multiplication A*B' requires compatible dimensions (A cols must equal B cols for A*B') and matching partitions (A.col_partition must equal B.col_partition)"

    # Result partitions: rows from A's rows, columns from B's rows
    result_row_partition = A.obj.row_partition
    result_col_partition = B.obj.row_partition

    # Use PETSc's MatMatTransposeMult
    C_mat = _mat_mat_transpose_mult(A.obj.A, B.obj.A, Prefix)

    # Wrap in our _Mat type and return as DRef
    obj = _Mat{T,Prefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T}}) -> Mat{T}

**MPI Collective**

Transpose-transpose multiplication: C = A' * B'.

Since PETSc does not have a direct AtBt product type, this is computed as
C = (B * A)' by materializing the transpose of the result.
"""
function Base.:*(At::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}, Bt::LinearAlgebra.Adjoint{<:Any,<:Mat{T,Prefix}}) where {T,Prefix}
    A = parent(At)::Mat{T,Prefix}
    B = parent(Bt)::Mat{T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Validate inputs: A' is n×m, B' is m×p (B is p×m), result is n×p
    @mpiassert (size(A, 1) == size(B, 2) &&
                A.obj.row_partition == B.obj.col_partition) "Transpose-transpose multiplication A'*B' requires compatible dimensions (A rows must equal B cols for A'*B') and matching partitions (A.row_partition must equal B.col_partition)"

    # Compute as (B * A)' since PETSc doesn't have AtBt product type
    # First compute BA
    BA_mat = _mat_mat_mult(B.obj.A, A.obj.A, Prefix)

    # Then transpose it to get (BA)' = A'B'
    # BA has partitions (B.row_partition, A.col_partition), so transpose has swapped
    result_row_partition = A.obj.col_partition
    result_col_partition = B.obj.row_partition
    C_mat = _mat_transpose(BA_mat, Prefix)

    # Result partitions: rows from A's columns, columns from B's rows

    # Wrap in our _Mat type
    obj = _Mat{T,Prefix}(C_mat, result_row_partition, result_col_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.mul!(C::Mat{T}, A::Mat{T}, B::Mat{T}) -> Mat{T}

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
        Base.:*(α::Number, A::Mat{T}) -> Mat{T}

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
            Cint(0),  # MAT_COPY_VALUES
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
        Base.:*(A::Mat{T}, α::Number) -> Mat{T}

    **MPI Collective**

    Matrix-scalar multiplication: A * α.

    Creates a new matrix with all entries scaled by α.
    """
    Base.:*(A::Mat{$PetscScalar,Prefix}, α::Number) where {Prefix} = α * A

    """
        Base.:+(α::Number, A::Mat{T}) -> Mat{T}

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
            Cint(0),  # MAT_COPY_VALUES
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
        Base.:+(A::Mat{T}, α::Number) -> Mat{T}

    **MPI Collective**

    Matrix-scalar addition: A + α.

    Adds scalar α to the diagonal of matrix A (equivalent to A + α*I).
    Requires A to be square.
    """
    Base.:+(A::Mat{$PetscScalar,Prefix}, α::Number) where {Prefix} = α + A
end

# PETSc matrix-matrix multiplication wrapper
PETSc.@for_libpetsc begin
    function _mat_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, Prefix::Type)
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
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
    function _mat_transpose_mat_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, Prefix::Type)
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatTransposeMatMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
        end
        return Cmat
    end

    # Matrix-transpose multiplication: C = A * B'
    function _mat_mat_transpose_mult(A::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar}, Prefix::Type)
        C = Ref{PETSc.CMat}(C_NULL)
        PETSC_DEFAULT_REAL = $PetscReal(-2.0)
        PETSc.@chk ccall((:MatMatTransposeMult, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat, Cint, $PetscReal, Ptr{PETSc.CMat}),
                         A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT_REAL, C)
        Cmat = PETSc.Mat{$PetscScalar}(C[])
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (PETSc.CMat, Cstring), Cmat, prefix_str)
        end
        return Cmat
    end
end

# Import spdiagm from SparseArrays
import SparseArrays: spdiagm

"""
    spdiagm(kv::Pair{<:Integer, <:Vec{T}}...) -> Mat{T}
    spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T}}...) -> Mat{T}

**MPI Collective**

Create a sparse diagonal matrix from distributed PETSc vectors.

Each pair `k => v` places the vector `v` on the `k`-th diagonal:
- `k = 0`: main diagonal
- `k > 0`: superdiagonal
- `k < 0`: subdiagonal

All vectors must have the same element type `T` and prefix. The matrix dimensions
are inferred from the diagonal positions and vector lengths, or can be specified explicitly.

Optional keyword arguments:
- `row_partition`: override the default equal-row partitioning (length `nranks+1`, start at 1, end at `m+1`, non-decreasing). Defaults to `default_row_partition(m, nranks)`.
- `col_partition`: override the default equal-column partitioning (length `nranks+1`, start at 1, end at `n+1`, non-decreasing). Defaults to `default_row_partition(n, nranks)`.

# Examples
```julia
# Create a tridiagonal matrix
A = spdiagm(-1 => lower, 0 => diag, 1 => upper)

# Create a 100×100 matrix with specified vectors on diagonals
B = spdiagm(100, 100, 0 => v1, 1 => v2)
```
"""
function spdiagm(kv::Pair{<:Integer, <:Vec{T,Prefix}}...; kwargs...) where {T,Prefix}
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

    return spdiagm(m, n, kv...; kwargs...)
end

function spdiagm(m::Integer, n::Integer, kv::Pair{<:Integer, <:Vec{T,Prefix}}...;
                 row_partition::Vector{Int}=default_row_partition(m, MPI.Comm_size(MPI.COMM_WORLD)),
                 col_partition::Vector{Int}=default_row_partition(n, MPI.Comm_size(MPI.COMM_WORLD))) where {T,Prefix}
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
                   Prefix=Prefix,
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
# eachrow for distributed MPIDENSE matrices
# - Assumes A is MATMPIDENSE; iterates local rows and yields SubArray views
# - Performs exactly one MatDenseGetArrayRead per iterator and restores on finish
# -----------------------------------------------------------------------------

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

"""
    Base.eachrow(A::Mat{T}) -> Iterator

**MPI Non-Collective**

Iterate over the rows of a distributed matrix.

Only iterates over rows owned by the current rank. Each row is returned as a view.
"""
function Base.eachrow(A::Mat{T,Prefix}) where {T,Prefix}
    return _eachrow_dense(A)
end

# -----------------------------------------------------------------------------
# Column Extraction
# -----------------------------------------------------------------------------

"""
    Base.getindex(A::Mat{T}, ::Colon, k::Int) -> Vec{T}

**MPI Non-Collective**

Extract column k from matrix A, returning a distributed vector.

Each rank extracts its owned rows from column k. The resulting vector has the same
row partition as matrix A. This operation is intended for dense matrices; the user
is responsible for ensuring A is dense.

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

    # Extract local portion of column k
    local_data = Vector{T}(undef, nlocal_rows)
    for i in 1:nlocal_rows
        global_row = row_lo + i - 1
        local_data[i] = _mat_getvalue(A.obj.A, global_row, k)
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
    own_row(A::Mat{T}) -> UnitRange{Int}

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
    is_dense(x::Mat{T}) -> Bool

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
    Matrix(x::Mat{T}) -> Matrix{T}

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
    SparseArrays.sparse(x::Mat{T}) -> SparseMatrixCSC{T, Int}

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
    Base.show(io::IO, x::Mat{T})

**MPI Collective**

Display a distributed PETSc Mat by converting to Julia Matrix or SparseMatrixCSC and showing it.

Dense matrices are converted to Matrix, other matrices are converted to SparseMatrixCSC.
This is a collective operation - all ranks must call it and will display the same matrix.
To print only on rank 0, use: `println(io0(), A)`
"""
Base.show(io::IO, x::Mat{T,Prefix}) where {T,Prefix} = show(io, is_dense(x) ? Matrix(x) : sparse(x))

"""
    Base.show(io::IO, mime::MIME, x::Mat{T})

**MPI Collective**

Display a distributed PETSc Mat with a specific MIME type by converting to Julia Matrix or SparseMatrixCSC.

Dense matrices are converted to Matrix, other matrices are converted to SparseMatrixCSC.
This is a collective operation - all ranks must call it and will display the same matrix.
To print only on rank 0, use: `show(io0(), mime, A)`
"""
Base.show(io::IO, mime::MIME, x::Mat{T,Prefix}) where {T,Prefix} = show(io, mime, is_dense(x) ? Matrix(x) : sparse(x))

