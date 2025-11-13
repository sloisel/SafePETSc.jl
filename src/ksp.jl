# Internal struct for KSP solver
struct _KSP{T}
    ksp::PETSc.KSP{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    prefix::String
end

"""
    Solver{T}

A PETSc KSP (Krylov Subspace) linear solver with element type `T`, managed by SafePETSc's
reference counting system.

`Solver{T}` is actually a type alias for `DRef{_KSP{T}}`, meaning solvers are automatically
tracked across MPI ranks and destroyed collectively when all ranks release their references.

Solvers can be reused for multiple linear systems with the same matrix, avoiding the cost
of repeated factorization or preconditioner setup.

# Construction

See the [`Solver`](@ref) constructor for creating solver instances.

# Usage

Solvers can be used implicitly via the backslash operator, or explicitly for reuse:

```julia
# Implicit (creates and destroys solver internally)
x = A \\ b

# Explicit (reuse solver for multiple solves)
ksp = Solver(A)
x1 = similar(b)
x2 = similar(b)
LinearAlgebra.ldiv!(ksp, x1, b1)  # First solve
LinearAlgebra.ldiv!(ksp, x2, b2)  # Second solve with same matrix
```

See also: [`Mat`](@ref), [`Vec`](@ref), the `Solver` constructor
"""
const Solver{T} = SafeMPI.DRef{_KSP{T}}

# Sizes for Solver reflect the operator dimensions recorded at construction
Base.size(r::SafeMPI.DRef{<:_KSP}) = (r.obj.row_partition[end] - 1, r.obj.col_partition[end] - 1)
Base.size(r::SafeMPI.DRef{<:_KSP}, d::Integer) = d == 1 ? (r.obj.row_partition[end] - 1) : (d == 2 ? (r.obj.col_partition[end] - 1) : 1)

"""
    Solver(A::Mat{T}; prefix::String="") -> Solver{T}

Create a PETSc KSP (Krylov Subspace) linear solver for the matrix A.

- `A::Mat{T}` is the matrix for which to create the solver.
- `prefix` is an optional string prefix for KSPSetOptionsPrefix() to set solver-specific command-line options.
- Returns a Solver that will destroy the PETSc KSP collectively when all ranks release their reference.

The Solver object can be used to solve linear systems via backslash (\\) and forward-slash (/) operators.
"""
function Solver(A::Mat{T}; prefix::String="") where T
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    # Create PETSc KSP
    petsc_ksp = _ksp_create_for_T(T, prefix)
    
    # Set the operator
    _ksp_set_operators!(petsc_ksp, A.obj.A)
    
    # Set up the KSP
    _ksp_setup!(petsc_ksp)
    
    # Wrap and DRef-manage
    obj = _KSP{T}(petsc_ksp, A.obj.row_partition, A.obj.col_partition, prefix)
    return SafeMPI.DRef(obj)
end

# Opt-in internal _KSP to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_KSP{T}}) where {T} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_KSP{T}) where {T}
    # Collective destroy of the underlying PETSc KSP on MPI.COMM_WORLD
    _destroy_petsc_ksp!(x.ksp)
    return nothing
end

# Create a distributed PETSc KSP for a given element type T
function _ksp_create_for_T(::Type{T}, prefix::String="") where {T}
    return _ksp_create_impl(T, prefix)
end

PETSc.@for_libpetsc begin
    function _ksp_create_impl(::Type{$PetscScalar}, prefix::String="")
        # Construct KSP via PETSc.jl API so internal fields are initialized correctly
        ksp = PETSc.KSP{$PetscScalar}(MPI.COMM_WORLD)
        if !isempty(prefix)
            PETSc.@chk ccall((:KSPSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CKSP, Cstring), ksp, prefix)
        end
        PETSc.@chk ccall((:KSPSetFromOptions, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP,), ksp)
        return ksp
    end
    
    function _ksp_set_operators!(ksp::PETSc.KSP{$PetscScalar}, A::PETSc.Mat{$PetscScalar})
        PETSc.@chk ccall((:KSPSetOperators, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP, CMat, CMat), ksp, A, A)
        return nothing
    end
    
    function _ksp_setup!(ksp::PETSc.KSP{$PetscScalar})
        PETSc.@chk ccall((:KSPSetUp, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP,), ksp)
        return nothing
    end
    
    function _destroy_petsc_ksp!(ksp::PETSc.AbstractKSP{$PetscScalar})
        PETSc.finalized($petsclib) || begin
            PETSc.@chk ccall((:KSPDestroy, $libpetsc), PETSc.PetscErrorCode,
                             (Ptr{CKSP},), ksp)
            ksp.ptr = C_NULL
        end
        return nothing
    end
    
    function _ksp_solve_vec!(ksp::PETSc.KSP{$PetscScalar}, x::PETSc.Vec{$PetscScalar}, b::PETSc.Vec{$PetscScalar})
        PETSc.@chk ccall((:KSPSolve, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP, CVec, CVec), ksp, b, x)
        return nothing
    end
    
    function _ksp_solve_transpose_vec!(ksp::PETSc.KSP{$PetscScalar}, x::PETSc.Vec{$PetscScalar}, b::PETSc.Vec{$PetscScalar})
        PETSc.@chk ccall((:KSPSolveTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP, CVec, CVec), ksp, b, x)
        return nothing
    end
    
    function _ksp_mat_solve!(ksp::PETSc.KSP{$PetscScalar}, X::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        PETSc.@chk ccall((:KSPMatSolve, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP, CMat, CMat), ksp, B, X)
        return nothing
    end
    
    function _ksp_mat_solve_transpose!(ksp::PETSc.KSP{$PetscScalar}, X::PETSc.Mat{$PetscScalar}, B::PETSc.Mat{$PetscScalar})
        PETSc.@chk ccall((:KSPMatSolveTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (CKSP, CMat, CMat), ksp, B, X)
        return nothing
    end
end

# Create an MPI dense PETSc matrix for a given element type T
function _mat_create_mpidense_for_T(::Type{T}, nlocal_rows::Integer, nlocal_cols::Integer,
                                   nglobal_rows::Integer, nglobal_cols::Integer,
                                   prefix::String="") where {T}
    return _mat_create_mpidense_impl(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, prefix)
end

PETSc.@for_libpetsc begin
    function _mat_create_mpidense_impl(::Type{$PetscScalar}, nlocal_rows::Integer, nlocal_cols::Integer,
                                       nglobal_rows::Integer, nglobal_cols::Integer,
                                       prefix::String="")
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
        # Set matrix type to MPI dense
        PETSc.@chk ccall((:MatSetType, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, Cstring), mat, "mpidense")
        # Allow options to apply (e.g., dense backend tuning) if a prefix was supplied
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetFromOptions, $libpetsc), PETSc.PetscErrorCode,
                             (CMat,), mat)
        end
        PETSc.@chk ccall((:MatSetUp, $libpetsc), PETSc.PetscErrorCode,
                         (CMat,), mat)
        return mat
    end
end

# Linear solve: x = A \ b (solve Ax = b)
function Base.:\(A::Mat{T}, b::Vec{T}) where {T}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && m == vec_length && A.obj.row_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition && A.obj.prefix == b.obj.prefix "Matrix must be square (A: $(m)×$(n)), matrix rows must match vector length (b: $(vec_length)), row/column partitions of A must match and equal b's row partition, and A and b must have the same prefix"

    # Create KSP solver
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, m, A.obj.prefix, A.obj.row_partition)

    # Solve
    _ksp_solve_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef
    obj = _Vec{T}(x_petsc, A.obj.row_partition, A.obj.prefix)
    return SafeMPI.DRef(obj)
end

# In-place linear solve: x = A \ b (reuses pre-allocated x)
function LinearAlgebra.ldiv!(x::Vec{T}, A::Mat{T}, b::Vec{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(A)
    b_length = size(b)[1]
    x_length = size(x)[1]
    @mpiassert (m == n && m == b_length && m == x_length &&
                A.obj.row_partition == A.obj.col_partition &&
                A.obj.row_partition == b.obj.row_partition &&
                A.obj.row_partition == x.obj.row_partition &&
                A.obj.prefix == b.obj.prefix == x.obj.prefix) "Matrix A must be square (got $(m)×$(n)), matrix rows must match vector lengths (b has length $(b_length), x has length $(x_length)), A's row and column partitions must match, A's row partition must match b's and x's row partitions, and all objects must have the same prefix"

    # Create KSP solver (will be destroyed when function exits)
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Solve into pre-allocated x
    _ksp_solve_vec!(ksp_obj.obj.ksp, x.obj.v, b.obj.v)

    PETSc.assemble(x.obj.v)

    return x
end

# In-place linear solve with pre-existing solver: x = A \ b (reuses solver and x)
function LinearAlgebra.ldiv!(ksp::Solver{T}, x::Vec{T}, b::Vec{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(ksp)
    b_length = size(b)[1]
    x_length = size(x)[1]
    @mpiassert (m == n && m == b_length && m == x_length &&
                ksp.obj.row_partition == b.obj.row_partition &&
                ksp.obj.row_partition == x.obj.row_partition &&
                ksp.obj.prefix == b.obj.prefix == x.obj.prefix) "Solver's matrix must be square (got $(m)×$(n)), solver matrix rows must match vector lengths (b has length $(b_length), x has length $(x_length)), solver's row partition must match b's and x's row partitions, and all objects must have the same prefix"

    # Solve into pre-allocated x using existing solver
    _ksp_solve_vec!(ksp.obj.ksp, x.obj.v, b.obj.v)

    PETSc.assemble(x.obj.v)

    return x
end

# Linear solve: X = A \ B (solve AX = B for multiple right-hand sides)
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE)
# as required by PETSc's KSPMatSolve. A and B may have different prefixes.
# X will inherit B's prefix.
function Base.:\(A::Mat{T}, B::Mat{T}) where {T}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && m == p && A.obj.row_partition == B.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), matrix rows must match (B: $(p)×$(q)), and row/column partitions of A must match and equal B's row partition"

    # Create KSP solver
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Create result matrix
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    col_lo = B.obj.col_partition[rank+1]
    col_hi = B.obj.col_partition[rank+2] - 1
    nlocal_cols = col_hi - col_lo + 1

    X_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, m, q, B.obj.prefix)

    # Solve with multiple RHS: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp_obj.obj.ksp, X_petsc, B.obj.A)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T}(X_petsc, A.obj.row_partition, B.obj.col_partition, B.obj.prefix)
    return SafeMPI.DRef(obj)
end

# In-place linear solve with matrix RHS: X = A \ B (reuses pre-allocated X)
# NOTE: User is responsible for ensuring B and X are dense (MATDENSE or MPIDENSE)
function LinearAlgebra.ldiv!(X::Mat{T}, A::Mat{T}, B::Mat{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(A)
    p, q = size(B)
    r, s = size(X)
    @mpiassert (m == n && m == p && m == r && q == s &&
                A.obj.row_partition == A.obj.col_partition &&
                A.obj.row_partition == B.obj.row_partition &&
                A.obj.row_partition == X.obj.row_partition &&
                B.obj.col_partition == X.obj.col_partition) "Matrix A must be square (got $(m)×$(n)), matrix A rows must match B rows (B is $(p)×$(q)), result matrix X must have dimensions $(m)×$(q) (got $(r)×$(s)), A's row and column partitions must match, A's row partition must match B's and X's row partitions, and B's column partition must match X's column partition"

    # Create KSP solver (will be destroyed when function exits)
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Solve with multiple RHS: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp_obj.obj.ksp, X.obj.A, B.obj.A)

    PETSc.assemble(X.obj.A)

    return X
end

# In-place linear solve with matrix RHS using pre-existing solver: X = A \ B (reuses solver and X)
# NOTE: User is responsible for ensuring B and X are dense (MATDENSE or MPIDENSE)
function LinearAlgebra.ldiv!(ksp::Solver{T}, X::Mat{T}, B::Mat{T}) where {T}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(ksp)
    p, q = size(B)
    r, s = size(X)
    @mpiassert (m == n && m == p && m == r && q == s &&
                ksp.obj.row_partition == B.obj.row_partition &&
                ksp.obj.row_partition == X.obj.row_partition &&
                B.obj.col_partition == X.obj.col_partition) "Solver's matrix must be square (got $(m)×$(n)), solver matrix rows must match B rows (B is $(p)×$(q)), result matrix X must have dimensions $(m)×$(q) (got $(r)×$(s)), solver's row partition must match B's and X's row partitions, and B's column partition must match X's column partition"

    # Solve with multiple RHS using existing solver: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp.obj.ksp, X.obj.A, B.obj.A)

    PETSc.assemble(X.obj.A)

    return X
end

# Transpose solve: x = A' \ b (solve A^T x = b)
function Base.:\(At::LinearAlgebra.Adjoint{T, <:Mat{T}}, b::Vec{T}) where {T}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && n == vec_length && A.obj.col_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition && A.obj.prefix == b.obj.prefix "Matrix must be square (A: $(m)×$(n)), matrix columns must match vector length (b: $(vec_length)), row/column partitions of A must match, column partition must equal b's row partition, and A and b must have the same prefix"

    # Create KSP solver
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal = col_hi - col_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, n, A.obj.prefix, A.obj.col_partition)

    # Solve transpose
    _ksp_solve_transpose_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef
    obj = _Vec{T}(x_petsc, A.obj.col_partition, A.obj.prefix)
    return SafeMPI.DRef(obj)
end

# Transpose solve: X = A' \ B (solve A^T X = B for multiple right-hand sides)
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE)
# as required by PETSc's KSPMatSolveTranspose. A and B may have different prefixes.
# X will inherit B's prefix.
function Base.:\(At::LinearAlgebra.Adjoint{T, <:Mat{T}}, B::Mat{T}) where {T}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && n == p && A.obj.col_partition == B.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), matrix columns must match (B: $(p)×$(q)), row/column partitions of A must match, and column partition must equal B's row partition"

    # Create KSP solver
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Create result matrix
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal_rows = col_hi - col_lo + 1

    col_lo_B = B.obj.col_partition[rank+1]
    col_hi_B = B.obj.col_partition[rank+2] - 1
    nlocal_cols = col_hi_B - col_lo_B + 1

    X_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, n, q, B.obj.prefix)

    # Solve A^T X = B using KSPMatSolveTranspose; B must be dense (user's responsibility)
    _ksp_mat_solve_transpose!(ksp_obj.obj.ksp, X_petsc, B.obj.A)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T}(X_petsc, A.obj.col_partition, B.obj.col_partition, B.obj.prefix)
    return SafeMPI.DRef(obj)
end

# Right division: x' = b' / A (solve x^T A = b^T, equivalent to A^T x = b)
function Base.:/(bt::LinearAlgebra.Adjoint{T, <:Vec{T}}, A::Mat{T}) where {T}
    b = parent(bt)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && m == vec_length && A.obj.row_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition && A.obj.prefix == b.obj.prefix "Matrix must be square (A: $(m)×$(n)), matrix rows must match vector length (b: $(vec_length)), row/column partitions of A must match and equal b's row partition, and A and b must have the same prefix"

    # Create KSP solver
    ksp_obj = Solver(A; prefix=A.obj.prefix)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, m, A.obj.prefix, A.obj.row_partition)

    # Solve transpose (b'/A is equivalent to A'\b)
    _ksp_solve_transpose_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef and return as adjoint
    obj = _Vec{T}(x_petsc, A.obj.row_partition, A.obj.prefix)
    x = SafeMPI.DRef(obj)
    return LinearAlgebra.Adjoint(x)
end

# Right division: X = B / A (solve XA = B)
# Rewritten as X = (A' \ B')'
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE).
# A and B may have different prefixes. X will inherit B's prefix.
function Base.:/(B::Mat{T}, A::Mat{T}) where {T}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && q == m && A.obj.row_partition == A.obj.col_partition && B.obj.col_partition == A.obj.row_partition "Matrix A must be square (A: $(m)×$(n)), B columns must match A size (B: $(p)×$(q)), row/column partitions of A must match, and B's column partition must equal A's row partition"

    # Step 1: Transpose B (B must be dense, so B^T will be dense)
    B_T = _mat_transpose(B.obj.A, B.obj.prefix)

    # Step 2: Solve A' \ B' (which is A^T Y = B^T, where Y = X^T)
    # Create KSP solver
    ksp_obj = Solver(A)

    # Create Y matrix with col partition of A for rows, row partition of B for cols
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal_rows = col_hi - col_lo + 1

    row_lo = B.obj.row_partition[rank+1]
    row_hi = B.obj.row_partition[rank+2] - 1
    nlocal_cols = row_hi - row_lo + 1

    Y_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, n, p, B.obj.prefix)

    # Solve A^T Y = B^T with dense RHS (B^T is dense since B is dense)
    _ksp_mat_solve_transpose!(ksp_obj.obj.ksp, Y_petsc, B_T)

    PETSc.assemble(Y_petsc)

    # Step 3: Transpose Y to get X = Y^T
    X_petsc = _mat_transpose(Y_petsc, B.obj.prefix)

    # Clean up intermediate matrices
    _destroy_petsc_mat!(B_T)
    _destroy_petsc_mat!(Y_petsc)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T}(X_petsc, B.obj.row_partition, A.obj.col_partition, B.obj.prefix)
    return SafeMPI.DRef(obj)
end

# Transpose right division: X = B / A' (solve XA^T = B)
# Rewritten as X = (A \ B')'
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE).
# A and B may have different prefixes. X will inherit B's prefix.
function Base.:/(B::Mat{T}, At::LinearAlgebra.Adjoint{T, <:Mat{T}}) where {T}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && q == n && A.obj.row_partition == A.obj.col_partition && B.obj.col_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), B columns must match A' size (B: $(p)×$(q)), and row/column partitions of A must match and equal B's column partition"

    # Step 1: Transpose B (B must be dense, so B^T will be dense)
    B_T = _mat_transpose(B.obj.A, B.obj.prefix)

    # Step 2: Solve A \ B' (which is AY = B^T, where Y = X^T)
    # Create KSP solver
    ksp_obj = Solver(A)

    # Create Y matrix with row partition of A for rows, row partition of B for cols
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    row_lo_B = B.obj.row_partition[rank+1]
    row_hi_B = B.obj.row_partition[rank+2] - 1
    nlocal_cols = row_hi_B - row_lo_B + 1

    Y_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, m, p, B.obj.prefix)

    # Solve AY = B^T with dense RHS (B^T is dense since B is dense)
    _ksp_mat_solve!(ksp_obj.obj.ksp, Y_petsc, B_T)

    PETSc.assemble(Y_petsc)

    # Step 3: Transpose Y to get X = Y^T
    X_petsc = _mat_transpose(Y_petsc, B.obj.prefix)

    # Clean up intermediate matrices
    _destroy_petsc_mat!(B_T)
    _destroy_petsc_mat!(Y_petsc)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T}(X_petsc, B.obj.row_partition, A.obj.row_partition, B.obj.prefix)
    return SafeMPI.DRef(obj)
end

# Matrix transpose wrapper
PETSc.@for_libpetsc begin
    # Debug helper: return PETSc MatType string
    function _mat_type_string(A::PETSc.Mat{$PetscScalar})
        type_ptr = Ref{Ptr{Cchar}}(C_NULL)
        PETSc.@chk ccall((:MatGetType, $libpetsc), PETSc.PetscErrorCode,
                         (CMat, Ptr{Ptr{Cchar}}), A, type_ptr)
        return unsafe_string(type_ptr[])
    end

    function _mat_transpose(A::PETSc.Mat{$PetscScalar}, prefix::String="",
                           row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[])
        # Create new transpose matrix using PETSc MatTranspose with MAT_INITIAL_MATRIX
        C_ptr = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall((:MatTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
                         A, MAT_INITIAL_MATRIX, C_ptr)
        C = PETSc.Mat{$PetscScalar}(C_ptr[])
        if !isempty(prefix)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CMat, Cstring), C, prefix)
        end
        return C
    end

    # In-place version using MAT_REUSE_MATRIX
    function _mat_transpose!(B::PETSc.Mat{$PetscScalar}, A::PETSc.Mat{$PetscScalar})
        # Attach precursor so PETSc accepts reuse even if B was allocated separately
        PETSc.@chk ccall((:MatTransposeSetPrecursor, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, PETSc.CMat), B, A)
        B_ref = Ref{PETSc.CMat}(B.ptr)
        PETSc.@chk ccall((:MatTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
                         A, MAT_REUSE_MATRIX, B_ref)
        return nothing
    end
end
