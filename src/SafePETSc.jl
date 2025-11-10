module SafePETSc

include("SafeMPI.jl")
using .SafeMPI
using PETSc
using MPI
using SparseArrays
using LinearAlgebra

# PETSc C pointer types
const CVec = Ptr{Cvoid}
const CMat = Ptr{Cvoid}
const CKSP = Ptr{Cvoid}
const default_check = Ref{Int}(10)


# PETSc constants for matrix operations
# These match PETSc's MatReuse enum:
#   MAT_INITIAL_MATRIX = 0, MAT_REUSE_MATRIX = 1, MAT_INPLACE_MATRIX = 2
# Used in MatConvert, MatTranspose, MatMatMult, etc.
const MAT_INITIAL_MATRIX = Cint(0)
const MAT_REUSE_MATRIX   = Cint(1)
const MAT_INPLACE_MATRIX = Cint(2)

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

`Vec{T}` is actually a type alias for `DRef{_Vec{T}}`, meaning vectors are automatically
tracked across MPI ranks and destroyed collectively when all ranks release their references.

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

See also: [`Vec_uniform`](@ref), [`Vec_sum`](@ref), [`Mat`](@ref), [`zeros_like`](@ref)
"""
const Vec{T} = SafeMPI.DRef{_Vec{T}}

struct _Mat{T}
    A::PETSc.Mat{T}
    row_partition::Vector{Int}
    col_partition::Vector{Int}
    prefix::String
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

include("vec.jl")
include("mat.jl")
include("ksp.jl")

# Opt-in internal _Vec to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Vec{T}}) where {T} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Vec{T}) where {T}
    # Collective destroy of the underlying PETSc Vec on MPI.COMM_WORLD
    _destroy_petsc_vec!(x.v)
    return nothing
end

# Opt-in internal _Mat to DRef-managed destruction
SafeMPI.destroy_trait(::Type{_Mat{T}}) where {T} = SafeMPI.CanDestroy()

function SafeMPI.destroy_obj!(x::_Mat{T}) where {T}
    # Collective destroy of the underlying PETSc Mat on MPI.COMM_WORLD
    _destroy_petsc_mat!(x.A)
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
