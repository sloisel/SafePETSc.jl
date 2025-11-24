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

# -----------------------------------------------------------------------------
# Prefix Type System
# -----------------------------------------------------------------------------

"""
    MPIDENSE

Prefix type for dense matrix operations and associated vectors.

# String Prefix
The string prefix is `"MPIDENSE_"`, which is prepended to PETSc option names.

# Default PETSc Types
- Matrices: `mpidense` (MPI dense matrix, row-major storage)
- Vectors: `mpi` (standard MPI vector)

# Usage
Use `MPIDENSE` for:
- Dense matrices where all elements are stored
- Operations requiring dense storage (e.g., `eachrow`)
- Direct solvers and dense linear algebra

# Example
```julia
# Create dense matrix
A = Mat_uniform([1.0 2.0; 3.0 4.0]; Prefix=MPIDENSE)

# Configure GPU acceleration for dense matrices
petsc_options_insert_string("-MPIDENSE_mat_type mpidense")
```

See also: [`MPIAIJ`](@ref), [`Mat`](@ref), [`Vec`](@ref)
"""
struct MPIDENSE end

"""
    MPIAIJ

Prefix type for sparse matrices and general vectors (default).

# String Prefix
The string prefix is `"MPIAIJ_"`, which is prepended to PETSc option names.

# Default PETSc Types
- Matrices: `mpiaij` (MPI sparse matrix, compressed row storage)
- Vectors: `mpi` (standard MPI vector)

# Usage
Use `MPIAIJ` for:
- Sparse matrices with few nonzeros per row
- Memory-efficient storage of large sparse systems
- Iterative solvers and sparse linear algebra
- General-purpose vector operations

This is the default prefix type when not specified.

# Example
```julia
# Create sparse matrix (MPIAIJ is the default)
A = Mat_uniform(sparse([1.0 0.0; 0.0 2.0]))

# Explicitly specify MPIAIJ prefix
B = Mat_uniform(data; Prefix=MPIAIJ)

# Configure iterative solver for sparse matrices
petsc_options_insert_string("-MPIAIJ_ksp_type gmres")
```

See also: [`MPIDENSE`](@ref), [`Mat`](@ref), [`Vec`](@ref)
"""
struct MPIAIJ end

"""
    prefix(::Type{<:Prefix}) -> String

Return the string prefix for a given prefix type.

The string prefix is prepended to PETSc option names. For example, with prefix type
`MPIDENSE`, the option `-mat_type mpidense` becomes `-MPIDENSE_mat_type mpidense`.

# Examples
```julia
prefix(MPIDENSE)  # Returns "MPIDENSE_"
prefix(MPIAIJ)    # Returns "MPIAIJ_"
```
"""
prefix(::Type{MPIDENSE}) = "MPIDENSE_"
prefix(::Type{MPIAIJ}) = "MPIAIJ_"

# Re-export default_check, finalize, and is_finalized from SafeMPI for convenient access
using .SafeMPI: default_check, finalize, is_finalized
export default_check, finalize, is_finalized


# PETSc constants for matrix operations
# These match PETSc's MatReuse enum:
#   MAT_INITIAL_MATRIX = 0, MAT_REUSE_MATRIX = 1
# Used in MatConvert, MatTranspose, MatMatMult, etc.
const MAT_INITIAL_MATRIX = Cint(0)
const MAT_REUSE_MATRIX   = Cint(1)

# PETSc MatStructure enum values (used by MatAXPY and related ops)
const SAME_NONZERO_PATTERN      = Cint(0)
const DIFFERENT_NONZERO_PATTERN = Cint(1)

"""
    petsc_options_insert_string(options_string::String)

**MPI Collective**

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

struct _Vec{T,Prefix}
    v::PETSc.Vec{T}
    row_partition::Vector{Int}
end

"""
    Vec{T,Prefix}

A distributed PETSc vector with element type `T` and prefix type `Prefix`, managed by SafePETSc's reference counting system.

`Vec{T,Prefix}` is a type alias for `DRef{_Vec{T,Prefix}}` and is released collectively when all ranks release their references. By default, released PETSc vectors are returned to an internal pool for reuse rather than destroyed immediately. To force destruction instead of pooling, set `ENABLE_VEC_POOL[] = false`, or call `clear_vec_pool!()` to free pooled vectors.

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
const Vec{T,Prefix} = SafeMPI.DRef{_Vec{T,Prefix}}

struct _Mat{T,Prefix}
    A::PETSc.Mat{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
end

"""
    Mat{T,Prefix}

A distributed PETSc matrix with element type `T` and prefix type `Prefix`, managed by SafePETSc's reference counting system.

`Mat{T,Prefix}` is actually a type alias for `DRef{_Mat{T,Prefix}}`, meaning matrices are automatically
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

See also: [`Mat_uniform`](@ref), [`Mat_sum`](@ref), [`Vec`](@ref), [`KSP`](@ref)
"""
const Mat{T,Prefix} = SafeMPI.DRef{_Mat{T,Prefix}}

export Vec, Vec_uniform, Vec_sum, default_row_partition
export zeros_like, ones_like, fill_like
export Mat, Mat_uniform, Mat_sum
export J
export KSP
export MPIDENSE, MPIAIJ, prefix
export petsc_options_insert_string
export Init, Initialized
export ENABLE_VEC_POOL, clear_vec_pool!, get_vec_pool_stats
export BlockProduct, calculate!
export io0
export map_rows
export own_row

"""
    io0(io=stdout; r::Set{Int}=Set{Int}([0]), dn=devnull)

**MPI Non-Collective**

Return `io` if the current rank is in `r`, otherwise return `dn`.

This is useful for printing output only on specific ranks to avoid duplicate output.

# Parameters
- `io`: The IO stream to use (default: `stdout`)
- `r`: Set of ranks that should produce output (default: `Set{Int}([0])`)
- `dn`: The IO stream to return for non-selected ranks (default: `devnull`)

# Examples
```julia
# Print only on rank 0 (default)
println(io0(), "This prints only on rank 0")

# Print only on rank 2
println(io0(r=Set([2])), "This prints only on rank 2")

# Print on ranks 0 and 3
println(io0(r=Set([0, 3])), "This prints on ranks 0 and 3")

# Write to file only on rank 1
open("output.txt", "w") do f
    println(io0(f; r=Set([1])), "This writes only on rank 1")
end
```
"""
function io0(io=stdout; r::Set{Int}=Set{Int}([0]), dn=devnull)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    return rank ∈ r ? io : dn
end

# -----------------------------------------------------------------------------
# Type Conversion Utilities (must be defined before includes)
# -----------------------------------------------------------------------------

"""
    J(x)

**MPI Collective** (when applied to PETSc types)

Universal conversion function that converts PETSc types to their native Julia equivalents.
For non-PETSc types, returns the input unchanged.

This function provides a uniform interface for converting any SafePETSc object to standard
Julia types, automatically choosing the appropriate conversion based on the input type.

# Behavior by Type
- `Vec{T}` → `Vector{T}` (via `Vector(v)`)
- `Mat{T}` (dense) → `Matrix{T}` (via `Matrix(A)`)
- `Mat{T}` (sparse) → `SparseMatrixCSC{T,Int}` (via `sparse(A)`)
- `Adjoint{T, Vec}` → `Adjoint{T, Vector{T}}`
- `Adjoint{T, Mat}` → `Adjoint{T, Matrix{T}}` or `Adjoint{T, SparseMatrixCSC}`
- `Pair` → converts the value while preserving the key
- Other types → returned unchanged

# Warning
When applied to PETSc types (`Vec`, `Mat`), this is a **collective operation** - all MPI
ranks must call it. The result is gathered to all ranks.

# Examples
```julia
# Convert Vec to Vector
v = Vec_uniform([1.0, 2.0, 3.0])
v_julia = J(v)  # Vector{Float64}

# Convert dense Mat to Matrix
A = Mat_uniform([1.0 2.0; 3.0 4.0])
A_julia = J(A)  # Matrix{Float64}

# Convert sparse Mat to SparseMatrixCSC
using SparseArrays
B = Mat_uniform(sparse([1.0 0.0; 0.0 2.0]))
B_julia = J(B)  # SparseMatrixCSC{Float64, Int}

# Scalars pass through unchanged
x = J(3.14)  # 3.14

# Useful in generic code
function compare_to_julia(petsc_result, julia_func, args...)
    expected = julia_func(J.(args)...)  # Convert all args
    actual = J(petsc_result)
    return norm(actual - expected)
end
```

See also: [`Vector`](@ref), [`Matrix`](@ref), [`sparse`](@ref)
"""
J(x) = x

# Special handling for Pair to convert the value
J(p::Pair) = p.first => J(p.second)

# Special handling for Adjoint to convert the parent
J(At::LinearAlgebra.Adjoint) = J(parent(At))'

include("vec.jl")
include("mat.jl")
include("ksp.jl")
include("blockproduct.jl")
include("map_rows.jl")

# Opt-in internal _Vec to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Vec{T,Prefix}}) where {T,Prefix} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Vec{T,Prefix}) where {T,Prefix}
    # Try to return to pool, otherwise destroy
    if ENABLE_VEC_POOL[]
        _return_vec_to_pool!(x.v, x.row_partition, Prefix)
    else
        # Collective destroy of the underlying PETSc Vec on MPI.COMM_WORLD
        _destroy_petsc_vec!(x.v)
    end
    return nothing
end

# Opt-in internal _Mat to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Mat{T,Prefix}}) where {T,Prefix} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Mat{T,Prefix}) where {T,Prefix}
    # Collective destroy of the underlying PETSc Mat on MPI.COMM_WORLD
    _destroy_petsc_mat!(x.A)
    return nothing
end

# Note: _KSP destruction is defined in ksp.jl

# Initialize MPI then PETSc if not already initialized (idempotent)
"""
    Init() -> nothing

**MPI Collective**

Ensure MPI and PETSc are initialized in the recommended order (MPI first, then PETSc).
Safe to call multiple times. Does not register custom finalizers; rely on library
defaults for shutdown (MPI.jl finalizes at exit; PETSc may remain initialized).

Sets up PETSc options for prefix types (MPIDENSE and MPIAIJ).
"""
function Init()
    if !MPI.Initialized()
        MPI.Init()
    end
    if !PETSc.initialized(PETSc.petsclibs[1])
        PETSc.initialize()
    end

    # Setup options for prefix types
    petsc_options_insert_string("-MPIDENSE_mat_type mpidense -MPIDENSE_vec_type mpi")
    petsc_options_insert_string("-MPIAIJ_mat_type mpiaij -MPIAIJ_vec_type mpi")

    # Exercise the options by creating dummy objects to prevent PETSc from
    # complaining about unused options. These go out of scope immediately.
    let
        _ = Vec_uniform([1.0], Prefix=MPIDENSE)
        _ = Vec_uniform([1.0], Prefix=MPIAIJ)
        _ = Mat_uniform([1.0;;], Prefix=MPIDENSE)
        _ = Mat_uniform([1.0;;], Prefix=MPIAIJ)
    end

    return nothing
end

"""
    Initialized() -> Bool

**MPI Non-Collective**

Return true if both MPI and PETSc are initialized. This is a simple conjunction
of `MPI.Initialized()` and `PETSc.initialized` for the active PETSc library.
"""
function Initialized()
    return MPI.Initialized() && PETSc.initialized(PETSc.petsclibs[1])
end

end
