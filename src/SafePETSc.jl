module SafePETSc

include("SafeMPI.jl")
using .SafeMPI
using PETSc
using MPI
using SparseArrays
using LinearAlgebra
using SHA

# PETSc C pointer types
const CVec = Ptr{Cvoid}
const CMat = Ptr{Cvoid}
const CKSP = Ptr{Cvoid}

# Re-export default_check from SafeMPI for convenient access
using .SafeMPI: default_check
export default_check


# PETSc constants for matrix operations
# These match PETSc's MatReuse enum:
#   MAT_INITIAL_MATRIX = 0, MAT_REUSE_MATRIX = 1, MAT_INPLACE_MATRIX = 2
# Used in MatConvert, MatTranspose, MatMatMult, etc.
const MAT_INITIAL_MATRIX = Cint(0)
const MAT_REUSE_MATRIX   = Cint(1)
const MAT_INPLACE_MATRIX = Cint(2)

# PETSc boolean constants (PetscBool type)
const PETSC_FALSE = PETSc.PetscBool(0)
const PETSC_TRUE  = PETSc.PetscBool(1)

# PETSc MatStructure enum values (used by MatAXPY and related ops)
const SAME_NONZERO_PATTERN      = Cint(0)
const DIFFERENT_NONZERO_PATTERN = Cint(1)
const SUBSET_NONZERO_PATTERN    = Cint(2)

# PETSc MatProductType enum values
# Used to track what type of matrix product operation created a matrix
const MATPRODUCT_UNSPECIFIED = Cint(0)  # Not a product matrix
const MATPRODUCT_AB  = Cint(1)          # C = A * B
const MATPRODUCT_AtB = Cint(2)          # C = A' * B
const MATPRODUCT_ABt = Cint(3)          # C = A * B'

"""
    petsc_options_insert_string(options_string::String)

Insert command-line style options into PETSc's global options database.

Example: `petsc_options_insert_string("-dense_mat_type mpidense")`

This sets options that will be used by PETSc objects created with the corresponding prefix.
PETSc must already be initialized by the caller.
"""
function petsc_options_insert_string(options_string::String)
    # Default to the first configured PETSc library; dispatches to the
    # library-specific implementation defined below.
    return petsc_options_insert_string(PETSc.petsclibs[1], options_string)
end

# Library-specific implementation without using Libdl; binds the correct
# `petsc_library` at compile-time via PETSc.@for_petsc.
PETSc.@for_petsc function petsc_options_insert_string(
    ::$UnionPetscLib,
    options_string::String,
)
    PETSc.@chk ccall(
        (:PetscOptionsInsertString, $petsc_library),
        PETSc.PetscErrorCode,
        (Ptr{Cvoid}, Cstring),
        C_NULL,
        options_string,
    )
    return nothing
end

struct _Vec{T}
    v::PETSc.Vec{T}
    row_partition::Vector{Int}
    prefix::String
end

"""
    Vec{T}

A distributed PETSc vector with element type `T`, managed by SafePETSc's reference counting system.

`Vec{T}` is a type alias for `DRef{_Vec{T}}` and is released collectively when all ranks release their references. By default, released PETSc vectors are returned to an internal pool for reuse rather than destroyed immediately. To force destruction instead of pooling, set `ENABLE_VEC_POOL[] = false`, or call `clear_vec_pool!()` to free pooled vectors.

# Construction

Use [`Vec_uniform`](@ref) or [`Vec_sum`](@ref) to create distributed vectors:

```julia
# Create from uniform data (same on all ranks)
v = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# Create from sparse contributions (summed across ranks)
using SparseArrays
v = Vec_sum(sparsevec([1, 3], [1.0, 3.0], 4))
```

# Operations

Vectors support standard arithmetic operations via broadcasting:
```julia
y = x .+ 1.0        # Element-wise addition
y .= 2.0 .* x       # In-place scaling
z = x .+ y          # Vector addition
```

Matrix-vector multiplication:
```julia
y = A * x           # Matrix-vector product
LinearAlgebra.mul!(y, A, x)  # In-place version
```

See also: [`Vec_uniform`](@ref), [`Vec_sum`](@ref), [`Mat`](@ref), [`zeros_like`](@ref), [`ENABLE_VEC_POOL`](@ref), [`clear_vec_pool!`](@ref)
"""
const Vec{T} = SafeMPI.DRef{_Vec{T}}

struct _Mat{T}
    A::PETSc.Mat{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    prefix::String
    fingerprint::Vector{UInt8}
    product_type::Cint                      # MATPRODUCT_* enum value
    product_args::Vector{Vector{UInt8}}     # Fingerprints of source matrices (A, B, etc.)
end

"""
    _Mat{T}(A::PETSc.Mat{T}, row_partition::Vector{Int}, col_partition::Vector{Int}, prefix::String; product_type=MATPRODUCT_UNSPECIFIED, product_args=Vector{UInt8}[])

Outer constructor for _Mat that automatically computes and stores the structural fingerprint.

The matrix A must be fully assembled before calling this constructor. The fingerprint
is computed from the CSR structure and is a 20-byte SHA-1 hash that uniquely identifies
the matrix structure (but not its values).

# Keyword Arguments
- `product_type::Cint`: MatProductType enum indicating if this matrix is a product (default: MATPRODUCT_UNSPECIFIED)
- `product_args::Vector{Vector{UInt8}}`: Fingerprints of source matrices if this is a product (default: empty)
"""
function _Mat{T}(A::PETSc.Mat{T}, row_partition::Vector{Int},
                 col_partition::Vector{Int}, prefix::String;
                 product_type::Cint=MATPRODUCT_UNSPECIFIED,
                 product_args::Vector{Vector{UInt8}}=Vector{UInt8}[]) where T
    # Verify that row_partition and col_partition match the actual PETSc matrix's dimensions
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    expected_local_rows = row_partition[rank+2] - row_partition[rank+1]
    expected_local_cols = col_partition[rank+2] - col_partition[rank+1]
    expected_global_rows = row_partition[end] - 1
    expected_global_cols = col_partition[end] - 1

    # Query actual dimensions from PETSc matrix
    # Note: PETSc uses Int64 for indices (configured with --with-64-bit-indices=1)
    actual_local_rows_ref = Ref{Int64}(0)
    actual_local_cols_ref = Ref{Int64}(0)
    actual_global_rows_ref = Ref{Int64}(0)
    actual_global_cols_ref = Ref{Int64}(0)

    PETSc.@chk ccall((:MatGetLocalSize, PETSc.libpetsc), PETSc.PetscErrorCode,
                     (PETSc.CMat, Ptr{Int64}, Ptr{Int64}),
                     A, actual_local_rows_ref, actual_local_cols_ref)
    PETSc.@chk ccall((:MatGetSize, PETSc.libpetsc), PETSc.PetscErrorCode,
                     (PETSc.CMat, Ptr{Int64}, Ptr{Int64}),
                     A, actual_global_rows_ref, actual_global_cols_ref)

    actual_local_rows = Int(actual_local_rows_ref[])
    actual_local_cols = Int(actual_local_cols_ref[])
    actual_global_rows = Int(actual_global_rows_ref[])
    actual_global_cols = Int(actual_global_cols_ref[])

    if actual_local_rows != expected_local_rows || actual_local_cols != expected_local_cols
        error("[Rank $rank] _Mat constructor: PETSc matrix local dimensions $(actual_local_rows)×$(actual_local_cols) " *
              "do not match row_partition/col_partition which specify $(expected_local_rows)×$(expected_local_cols). " *
              "row_partition=$row_partition, col_partition=$col_partition")
    end

    if actual_global_rows != expected_global_rows || actual_global_cols != expected_global_cols
        error("[Rank $rank] _Mat constructor: PETSc matrix global dimensions $(actual_global_rows)×$(actual_global_cols) " *
              "do not match row_partition/col_partition which specify $(expected_global_rows)×$(expected_global_cols). " *
              "row_partition=$row_partition, col_partition=$col_partition")
    end

    # Only compute fingerprints for product matrices (used for pooling)
    # Non-product matrices (from Mat_sum/Mat_uniform) don't use pooling, so skip the expensive SHA-1 computation
    fingerprint = (ENABLE_MAT_POOL[] && product_type != MATPRODUCT_UNSPECIFIED) ?
                  _matrix_fingerprint(A, row_partition, col_partition, prefix) : UInt8[]
    return _Mat{T}(A, row_partition, col_partition, prefix, fingerprint,
                   product_type, product_args)
end

"""
    Mat{T}

A distributed PETSc matrix with element type `T`, managed by SafePETSc's reference counting system.

`Mat{T}` is actually a type alias for `DRef{_Mat{T}}`, meaning matrices are automatically
tracked across MPI ranks and destroyed collectively when all ranks release their references.

# Construction

Use [`Mat_uniform`](@ref) or [`Mat_sum`](@ref) to create distributed matrices:

```julia
# Create from uniform data (same on all ranks)
A = Mat_uniform([1.0 2.0; 3.0 4.0])

# Create from sparse contributions (summed across ranks)
using SparseArrays
A = Mat_sum(sparse([1, 2], [1, 2], [1.0, 4.0], 2, 2))
```

# Operations

Matrices support standard linear algebra operations:
```julia
# Matrix-vector multiplication
y = A * x

# Matrix-matrix multiplication
C = A * B

# Matrix transpose
B = A'
B = Mat(A')  # Materialize transpose

# Linear solve
x = A \\ b

# Concatenation
C = vcat(A, B)  # or cat(A, B; dims=1)
D = hcat(A, B)  # or cat(A, B; dims=2)
E = blockdiag(A, B)

# Diagonal matrix from vectors
using SparseArrays
A = spdiagm(0 => diag_vec, 1 => upper_diag)
```

See also: [`Mat_uniform`](@ref), [`Mat_sum`](@ref), [`Vec`](@ref), [`Solver`](@ref)
"""
const Mat{T} = SafeMPI.DRef{_Mat{T}}

export Vec, Vec_uniform, Vec_sum, default_row_partition
export zeros_like, ones_like, fill_like
export Mat, Mat_uniform, Mat_sum
export Solver
export petsc_options_insert_string
export Init, Initialized
export ENABLE_VEC_POOL, clear_vec_pool!, get_vec_pool_stats
export ENABLE_MAT_POOL, clear_mat_pool!, get_mat_pool_stats

include("vec.jl")
include("mat.jl")
include("ksp.jl")

# Opt-in internal _Vec to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Vec{T}}) where {T} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Vec{T}) where {T}
    # Try to return to pool, otherwise destroy
    if ENABLE_VEC_POOL[]
        _return_vec_to_pool!(x.v, x.row_partition, x.prefix)
    else
        # Collective destroy of the underlying PETSc Vec on MPI.COMM_WORLD
        _destroy_petsc_vec!(x.v)
    end
    return nothing
end

# Opt-in internal _Mat to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Mat{T}}) where {T} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Mat{T}) where {T}
    # Try to return to pool, otherwise destroy
    if ENABLE_MAT_POOL[]
        _return_mat_to_pool!(x.A, x.row_partition, x.col_partition, x.prefix, x.product_type, x.product_args)
    else
        # Collective destroy of the underlying PETSc Mat on MPI.COMM_WORLD
        _destroy_petsc_mat!(x.A)
    end
    return nothing
end

# Note: _KSP destruction is defined in ksp.jl

# Initialize MPI then PETSc if not already initialized (idempotent)
"""
    Init() -> nothing

Ensure MPI and PETSc are initialized in the recommended order (MPI first, then PETSc).
Safe to call multiple times. Does not register custom finalizers; rely on library
defaults for shutdown (MPI.jl finalizes at exit; PETSc may remain initialized).
"""
function Init()
    if !MPI.Initialized()
        MPI.Init()
    end
    if !PETSc.initialized(PETSc.petsclibs[1])
        PETSc.initialize()
    end
    return nothing
end

"""
    Initialized() -> Bool

Return true if both MPI and PETSc are initialized. This is a simple conjunction
of `MPI.Initialized()` and `PETSc.initialized` for the active PETSc library.
"""
function Initialized()
    return MPI.Initialized() && PETSc.initialized(PETSc.petsclibs[1])
end

end
