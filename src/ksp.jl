# Internal struct for KSP solver
struct _KSP{T,Prefix}
    ksp::PETSc.KSP{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
end

"""
    KSP{T,Prefix}

A PETSc KSP (Krylov Subspace) linear solver with element type `T` and prefix type `Prefix`,
managed by SafePETSc's reference counting system.

`KSP{T,Prefix}` is actually a type alias for `DRef{_KSP{T,Prefix}}`, meaning solvers are automatically
tracked across MPI ranks and destroyed collectively when all ranks release their references.

KSP objects can be reused for multiple linear systems with the same matrix, avoiding the cost
of repeated factorization or preconditioner setup.

# Construction

See the [`KSP`](@ref) constructor for creating solver instances.

# Usage

KSP solvers can be used implicitly via the backslash operator, or explicitly for reuse:

```julia
# Implicit (creates and destroys solver internally)
x = A \\ b

# Explicit (reuse solver for multiple solves)
ksp = KSP(A)
x1 = similar(b)
x2 = similar(b)
LinearAlgebra.ldiv!(ksp, x1, b1)  # First solve
LinearAlgebra.ldiv!(ksp, x2, b2)  # Second solve with same matrix
```

See also: [`Mat`](@ref), [`Vec`](@ref), the `KSP` constructor
"""
const KSP{T,Prefix} = SafeMPI.DRef{_KSP{T,Prefix}}

# Sizes for KSP reflect the operator dimensions recorded at construction
Base.size(r::SafeMPI.DRef{<:_KSP}) = (r.obj.row_partition[end] - 1, r.obj.col_partition[end] - 1)
Base.size(r::SafeMPI.DRef{<:_KSP}, d::Integer) = d == 1 ? (r.obj.row_partition[end] - 1) : (d == 2 ? (r.obj.col_partition[end] - 1) : 1)

"""
    KSP(A::Mat{T,Prefix}) -> KSP{T,Prefix}

**MPI Collective**

Create a PETSc KSP (Krylov Subspace) linear solver for the matrix A.

- `A::Mat{T,Prefix}` is the matrix for which to create the solver.
- Returns a KSP that will destroy the PETSc KSP collectively when all ranks release their reference.

The KSP object inherits the prefix from the matrix A and can be used to solve linear systems via backslash (\\) and forward-slash (/) operators.
"""
function KSP(A::Mat{T,Prefix}) where {T,Prefix}
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Create PETSc KSP
    petsc_ksp = _ksp_create_for_T(T, Prefix)

    # Set the operator
    _ksp_set_operators!(petsc_ksp, A.obj.A)

    # Set up the KSP
    _ksp_setup!(petsc_ksp)

    # Wrap and DRef-manage
    obj = _KSP{T,Prefix}(petsc_ksp, A.obj.row_partition, A.obj.col_partition)
    return SafeMPI.DRef(obj)
end

# Opt-in internal _KSP to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_KSP{T,Prefix}}) where {T,Prefix} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_KSP{T,Prefix}) where {T,Prefix}
    # Collective destroy of the underlying PETSc KSP on MPI.COMM_WORLD
    _destroy_petsc_ksp!(x.ksp)
    return nothing
end

# Create a distributed PETSc KSP for a given element type T
function _ksp_create_for_T(::Type{T}, Prefix::Type) where {T}
    return _ksp_create_impl(T, Prefix)
end

PETSc.@for_libpetsc begin
    function _ksp_create_impl(::Type{$PetscScalar}, Prefix::Type)
        # Construct KSP via PETSc.jl API so internal fields are initialized correctly
        ksp = PETSc.KSP{$PetscScalar}(MPI.COMM_WORLD)
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:KSPSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CKSP, Cstring), ksp, prefix_str)
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
                                   Prefix::Type) where {T}
    return _mat_create_mpidense_impl(T, nlocal_rows, nlocal_cols, nglobal_rows, nglobal_cols, Prefix)
end

PETSc.@for_libpetsc begin
    function _mat_create_mpidense_impl(::Type{$PetscScalar}, nlocal_rows::Integer, nlocal_cols::Integer,
                                       nglobal_rows::Integer, nglobal_cols::Integer,
                                       Prefix::Type)
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
end

"""
    Base.:\\(A::Mat{T,Prefix}, b::Vec{T}) -> Vec{T}

**MPI Collective**

Solve the linear system Ax = b using PETSc's KSP solver.

Creates a KSP solver internally and returns the solution vector x.
For repeated solves with the same matrix, use `KSP` explicitly for better performance.
"""
function Base.:\(A::Mat{T,Prefix}, b::Vec{T}) where {T,Prefix}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && m == vec_length && A.obj.row_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix must be square (A: $(m)×$(n)), matrix rows must match vector length (b: $(vec_length)), and row/column partitions of A must match and equal b's row partition"

    # Create KSP solver
    ksp_obj = KSP(A)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, m, A.obj.row_partition)

    # Solve
    _ksp_solve_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef
    obj = _Vec{T}(x_petsc, A.obj.row_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.ldiv!(x::Vec{T}, A::Mat{T,Prefix}, b::Vec{T}) -> Vec{T}

**MPI Collective**

In-place solve of Ax = b, storing the result in the pre-allocated vector x.

Creates a KSP solver internally. For repeated solves, use the `ldiv!(ksp, x, b)` variant with a reusable KSP object.
"""
function LinearAlgebra.ldiv!(x::Vec{T}, A::Mat{T,Prefix}, b::Vec{T}) where {T,Prefix}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(A)
    b_length = size(b)[1]
    x_length = size(x)[1]
    @mpiassert (m == n && m == b_length && m == x_length &&
                A.obj.row_partition == A.obj.col_partition &&
                A.obj.row_partition == b.obj.row_partition &&
                A.obj.row_partition == x.obj.row_partition) "Matrix A must be square (got $(m)×$(n)), matrix rows must match vector lengths (b has length $(b_length), x has length $(x_length)), A's row and column partitions must match, and A's row partition must match b's and x's row partitions"

    # Create KSP solver (will be destroyed when function exits)
    ksp_obj = KSP(A)

    # Solve into pre-allocated x
    _ksp_solve_vec!(ksp_obj.obj.ksp, x.obj.v, b.obj.v)

    PETSc.assemble(x.obj.v)

    return x
end

"""
    LinearAlgebra.ldiv!(ksp::KSP{T,Prefix}, x::Vec{T}, b::Vec{T}) -> Vec{T}

**MPI Collective**

In-place solve using a pre-existing KSP solver, storing the result in x.

Reuses the solver and the result vector for maximum efficiency in repeated solves.
"""
function LinearAlgebra.ldiv!(ksp::KSP{T,Prefix}, x::Vec{T}, b::Vec{T}) where {T,Prefix}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(ksp)
    b_length = size(b)[1]
    x_length = size(x)[1]
    @mpiassert (m == n && m == b_length && m == x_length &&
                ksp.obj.row_partition == b.obj.row_partition &&
                ksp.obj.row_partition == x.obj.row_partition) "KSP's matrix must be square (got $(m)×$(n)), KSP matrix rows must match vector lengths (b has length $(b_length), x has length $(x_length)), and KSP's row partition must match b's and x's row partitions"

    # Solve into pre-allocated x using existing solver
    _ksp_solve_vec!(ksp.obj.ksp, x.obj.v, b.obj.v)

    PETSc.assemble(x.obj.v)

    return x
end

"""
    Base.:\\(A::Mat{T,PrefixA}, B::Mat{T,PrefixB}) -> Mat{T,PrefixB} where {T,PrefixA,PrefixB}

**MPI Collective**

Solve AX = B for multiple right-hand sides using PETSc's KSPMatSolve.

The matrix B must be dense (MATDENSE or MPIDENSE). Returns the solution matrix X.
A and B may have different prefixes; X will inherit B's prefix.
"""
function Base.:\(A::Mat{T,PrefixA}, B::Mat{T,PrefixB}) where {T,PrefixA,PrefixB}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && m == p && A.obj.row_partition == B.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), matrix rows must match (B: $(p)×$(q)), and row/column partitions of A must match and equal B's row partition"

    # Create KSP solver
    ksp_obj = KSP(A)

    # Create result matrix
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    col_lo = B.obj.col_partition[rank+1]
    col_hi = B.obj.col_partition[rank+2] - 1
    nlocal_cols = col_hi - col_lo + 1

    X_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, m, q, PrefixB)

    # Solve with multiple RHS: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp_obj.obj.ksp, X_petsc, B.obj.A)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T,PrefixB}(X_petsc, A.obj.row_partition, B.obj.col_partition)
    return SafeMPI.DRef(obj)
end

"""
    LinearAlgebra.ldiv!(X::Mat{T,Prefix}, A::Mat{T,Prefix}, B::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

In-place solve of AX = B for multiple right-hand sides, storing the result in X.

Both B and X must be dense matrices (MATDENSE or MPIDENSE).
Creates a KSP solver internally. For repeated solves, use the `ldiv!(ksp, X, B)` variant with a reusable KSP object.
"""
function LinearAlgebra.ldiv!(X::Mat{T,PrefixX}, A::Mat{T,PrefixA}, B::Mat{T,PrefixB}) where {T,PrefixX,PrefixA,PrefixB}
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
    ksp_obj = KSP(A)

    # Solve with multiple RHS: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp_obj.obj.ksp, X.obj.A, B.obj.A)

    PETSc.assemble(X.obj.A)

    return X
end

"""
    LinearAlgebra.ldiv!(ksp::KSP{T,Prefix}, X::Mat{T,Prefix}, B::Mat{T,Prefix}) -> Mat{T,Prefix}

**MPI Collective**

In-place solve using a pre-existing KSP solver for multiple right-hand sides.

Reuses the solver and result matrix for maximum efficiency. B and X must be dense matrices.
"""
function LinearAlgebra.ldiv!(ksp::KSP{T,PrefixKSP}, X::Mat{T,PrefixX}, B::Mat{T,PrefixB}) where {T,PrefixKSP,PrefixX,PrefixB}
    # Validate dimensions and partitioning - single @mpiassert for efficiency
    m, n = size(ksp)
    p, q = size(B)
    r, s = size(X)
    @mpiassert (m == n && m == p && m == r && q == s &&
                ksp.obj.row_partition == B.obj.row_partition &&
                ksp.obj.row_partition == X.obj.row_partition &&
                B.obj.col_partition == X.obj.col_partition) "KSP's matrix must be square (got $(m)×$(n)), KSP matrix rows must match B rows (B is $(p)×$(q)), result matrix X must have dimensions $(m)×$(q) (got $(r)×$(s)), KSP's row partition must match B's and X's row partitions, and B's column partition must match X's column partition"

    # Solve with multiple RHS using existing solver: B must be dense (user's responsibility)
    _ksp_mat_solve!(ksp.obj.ksp, X.obj.A, B.obj.A)

    PETSc.assemble(X.obj.A)

    return X
end

"""
    Base.:\\(At::LinearAlgebra.Adjoint{T, <:Mat{T,Prefix}}, b::Vec{T}) -> Vec{T}

**MPI Collective**

Solve the transposed system A'x = b using PETSc's KSPSolveTranspose.

Returns the solution vector x. Creates a KSP solver internally.
"""
function Base.:\(At::LinearAlgebra.Adjoint{T, <:Mat{T,Prefix}}, b::Vec{T}) where {T,Prefix}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && n == vec_length && A.obj.col_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix must be square (A: $(m)×$(n)), matrix columns must match vector length (b: $(vec_length)), row/column partitions of A must match, and column partition must equal b's row partition"

    # Create KSP solver
    ksp_obj = KSP(A)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal = col_hi - col_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, n, A.obj.col_partition)

    # Solve transpose
    _ksp_solve_transpose_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef
    obj = _Vec{T}(x_petsc, A.obj.col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:\\(At::LinearAlgebra.Adjoint{T, <:Mat{T,PrefixA}}, B::Mat{T,PrefixB}) -> Mat{T,PrefixB} where {T,PrefixA,PrefixB}

**MPI Collective**

Solve A'X = B for multiple right-hand sides using PETSc's KSPMatSolveTranspose.

The matrix B must be dense (MATDENSE or MPIDENSE). Returns the solution matrix X.
A and B may have different prefixes; X will inherit B's prefix.
"""
function Base.:\(At::LinearAlgebra.Adjoint{T, <:Mat{T,PrefixA}}, B::Mat{T,PrefixB}) where {T,PrefixA,PrefixB}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && n == p && A.obj.col_partition == B.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), matrix columns must match (B: $(p)×$(q)), row/column partitions of A must match, and column partition must equal B's row partition"

    # Create KSP solver
    ksp_obj = KSP(A)

    # Create result matrix
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal_rows = col_hi - col_lo + 1

    col_lo_B = B.obj.col_partition[rank+1]
    col_hi_B = B.obj.col_partition[rank+2] - 1
    nlocal_cols = col_hi_B - col_lo_B + 1

    X_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, n, q, PrefixB)

    # Solve A^T X = B using KSPMatSolveTranspose; B must be dense (user's responsibility)
    _ksp_mat_solve_transpose!(ksp_obj.obj.ksp, X_petsc, B.obj.A)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T,PrefixB}(X_petsc, A.obj.col_partition, B.obj.col_partition)
    return SafeMPI.DRef(obj)
end

"""
    Base.:/(bt::LinearAlgebra.Adjoint{T, <:Vec{T}}, A::Mat{T,Prefix}) -> Adjoint{T, Vec{T}}

**MPI Collective**

Right division b'/A, which solves x^T A = b^T (equivalent to A^T x = b).

Returns the solution as an adjoint vector x'.
"""
function Base.:/(bt::LinearAlgebra.Adjoint{T, <:Vec{T}}, A::Mat{T,Prefix}) where {T,Prefix}
    b = parent(bt)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    vec_length = size(b)[1]
    @mpiassert m == n && m == vec_length && A.obj.row_partition == b.obj.row_partition && A.obj.row_partition == A.obj.col_partition "Matrix must be square (A: $(m)×$(n)), matrix rows must match vector length (b: $(vec_length)), and row/column partitions of A must match and equal b's row partition"

    # Create KSP solver
    ksp_obj = KSP(A)

    # Create result vector
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal = row_hi - row_lo + 1

    x_petsc = _vec_create_mpi_for_T(T, nlocal, m, A.obj.row_partition)

    # Solve transpose (b'/A is equivalent to A'\b)
    _ksp_solve_transpose_vec!(ksp_obj.obj.ksp, x_petsc, b.obj.v)

    PETSc.assemble(x_petsc)

    # Wrap in DRef and return as adjoint
    obj = _Vec{T}(x_petsc, A.obj.row_partition)
    x = SafeMPI.DRef(obj)
    return LinearAlgebra.Adjoint(x)
end

# Right division: X = B / A (solve XA = B)
# Rewritten as X = (A' \ B')'
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE).
# A and B may have different prefixes. X will inherit B's prefix.
function Base.:/(B::Mat{T,PrefixB}, A::Mat{T,PrefixA}) where {T,PrefixB,PrefixA}
    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && q == m && A.obj.row_partition == A.obj.col_partition && B.obj.col_partition == A.obj.row_partition "Matrix A must be square (A: $(m)×$(n)), B columns must match A size (B: $(p)×$(q)), row/column partitions of A must match, and B's column partition must equal A's row partition"

    # Step 1: Transpose B (B must be dense, so B^T will be dense)
    B_T = _mat_transpose(B.obj.A, PrefixB)

    # Step 2: Solve A' \ B' (which is A^T Y = B^T, where Y = X^T)
    # Create KSP solver
    ksp_obj = KSP(A)

    # Create Y matrix with col partition of A for rows, row partition of B for cols
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    col_lo = A.obj.col_partition[rank+1]
    col_hi = A.obj.col_partition[rank+2] - 1
    nlocal_rows = col_hi - col_lo + 1

    row_lo = B.obj.row_partition[rank+1]
    row_hi = B.obj.row_partition[rank+2] - 1
    nlocal_cols = row_hi - row_lo + 1

    Y_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, n, p, PrefixB)

    # Solve A^T Y = B^T with dense RHS (B^T is dense since B is dense)
    _ksp_mat_solve_transpose!(ksp_obj.obj.ksp, Y_petsc, B_T)

    PETSc.assemble(Y_petsc)

    # Step 3: Transpose Y to get X = Y^T
    X_petsc = _mat_transpose(Y_petsc, PrefixB)

    # Clean up intermediate matrices
    _destroy_petsc_mat!(B_T)
    _destroy_petsc_mat!(Y_petsc)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T,PrefixB}(X_petsc, B.obj.row_partition, A.obj.col_partition)
    return SafeMPI.DRef(obj)
end

# Transpose right division: X = B / A' (solve XA^T = B)
# Rewritten as X = (A \ B')'
# NOTE: User is responsible for ensuring B is dense (MATDENSE or MPIDENSE).
# A and B may have different prefixes. X will inherit B's prefix.
function Base.:/(B::Mat{T,PrefixB}, At::LinearAlgebra.Adjoint{T, <:Mat{T,PrefixA}}) where {T,PrefixB,PrefixA}
    A = parent(At)

    # Check dimensions and partitioning - coalesced into single MPI synchronization
    m, n = size(A)
    p, q = size(B)
    @mpiassert m == n && q == n && A.obj.row_partition == A.obj.col_partition && B.obj.col_partition == A.obj.col_partition "Matrix A must be square (A: $(m)×$(n)), B columns must match A' size (B: $(p)×$(q)), and row/column partitions of A must match and equal B's column partition"

    # Step 1: Transpose B (B must be dense, so B^T will be dense)
    B_T = _mat_transpose(B.obj.A, PrefixB)

    # Step 2: Solve A \ B' (which is AY = B^T, where Y = X^T)
    # Create KSP solver
    ksp_obj = KSP(A)

    # Create Y matrix with row partition of A for rows, row partition of B for cols
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    row_lo_B = B.obj.row_partition[rank+1]
    row_hi_B = B.obj.row_partition[rank+2] - 1
    nlocal_cols = row_hi_B - row_lo_B + 1

    Y_petsc = _mat_create_mpidense_for_T(T, nlocal_rows, nlocal_cols, m, p, PrefixB)

    # Solve AY = B^T with dense RHS (B^T is dense since B is dense)
    _ksp_mat_solve!(ksp_obj.obj.ksp, Y_petsc, B_T)

    PETSc.assemble(Y_petsc)

    # Step 3: Transpose Y to get X = Y^T
    X_petsc = _mat_transpose(Y_petsc, PrefixB)

    # Clean up intermediate matrices
    _destroy_petsc_mat!(B_T)
    _destroy_petsc_mat!(Y_petsc)

    PETSc.assemble(X_petsc)

    # Wrap in DRef with B's prefix
    obj = _Mat{T,PrefixB}(X_petsc, B.obj.row_partition, A.obj.row_partition)
    return SafeMPI.DRef(obj)
end

# Matrix transpose wrapper
PETSc.@for_libpetsc begin
    function _mat_transpose(A::PETSc.Mat{$PetscScalar}, Prefix::Type,
                           row_partition::Vector{Int}=Int[], col_partition::Vector{Int}=Int[])
        # Create new transpose matrix using PETSc MatTranspose with MAT_INITIAL_MATRIX
        C_ptr = Ref{PETSc.CMat}(C_NULL)
        PETSc.@chk ccall((:MatTranspose, $libpetsc), PETSc.PetscErrorCode,
                         (PETSc.CMat, Cint, Ptr{PETSc.CMat}),
                         A, MAT_INITIAL_MATRIX, C_ptr)
        C = PETSc.Mat{$PetscScalar}(C_ptr[])
        prefix_str = SafePETSc.prefix(Prefix)
        if !isempty(prefix_str)
            PETSc.@chk ccall((:MatSetOptionsPrefix, $libpetsc), PETSc.PetscErrorCode,
                             (CMat, Cstring), C, prefix_str)
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
