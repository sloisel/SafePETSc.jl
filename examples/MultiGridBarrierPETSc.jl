"""
    MultiGridBarrierPETSc

A module that provides a convenient interface for using MultiGridBarrier with PETSc
distributed types through SafePETSc.

# Exports
- `fem2d_petsc`: Creates a PETSc-based Geometry from fem2d parameters
- `fem2d_solve`: Solves a fem2d problem using amgb with PETSc types

# Usage
```julia
using SafePETSc
SafePETSc.Init()
using .MultiGridBarrierPETSc

# Create PETSc geometry
g = fem2d_petsc(Float64; maxh=0.1)

# Solve the problem
sol = fem2d_solve(Float64; maxh=0.1, p=2.0, verbose=true)
```
"""
module MultiGridBarrierPETSc

using MPI
using SafePETSc
using SafePETSc: MPIDENSE, MPIAIJ
using LinearAlgebra
using SparseArrays

# Install MultiGridBarrier if needed
using Pkg
if !haskey(Pkg.project().dependencies, "MultiGridBarrier")
    println(SafePETSc.io0(), "Installing MultiGridBarrier.jl v0.11.25...")
    Pkg.add(name="MultiGridBarrier", version="0.11.25")
end
using MultiGridBarrier

# ============================================================================
# MultiGridBarrier API Implementation for SafePETSc Types
# ============================================================================

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_hcat, amgb_diag, amgb_blockdiag, map_rows

# amgb_zeros: Create zero matrices with appropriate type
MultiGridBarrier.amgb_zeros(::Mat{T, MPIAIJ}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIAIJ}}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)
MultiGridBarrier.amgb_zeros(::Mat{T, MPIDENSE}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIDENSE}}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)

# amgb_all_isfinite: Check if all elements are finite
MultiGridBarrier.amgb_all_isfinite(z::Vec{T}) where {T} = all(isfinite.(Vector(z)))

# amgb_hcat: Horizontal concatenation
# Just use the built-in hcat - it handles partitions correctly
MultiGridBarrier.amgb_hcat(A::Mat...) = hcat(A...)

# amgb_diag: Create diagonal matrix from vector
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIAIJ)
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIAIJ)
MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIDENSE)
MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIDENSE)

# amgb_blockdiag: Block diagonal concatenation
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIAIJ}...) where {T} = blockdiag(args...)
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIDENSE}...) where {T} = blockdiag(args...)

# map_rows: Apply function to each row
function MultiGridBarrier.map_rows(f, A::Union{Vec{T}, Mat{T}, LinearAlgebra.Adjoint{T, <:Mat{T}}}...) where {T}
    # Materialize any adjoint matrices using Mat(A') constructor
    materialized = [a isa LinearAlgebra.Adjoint ? Mat(a) : a for a in A]
    return SafePETSc.map_rows(f, materialized...)
end

# Additional base functions needed for MultiGridBarrier
Base.minimum(v::Vec{T}) where {T} = minimum(Vector(v))
Base.maximum(v::Vec{T}) where {T} = maximum(Vector(v))

# ============================================================================
# Geometry Conversion
# ============================================================================

"""
    geometry_native_to_petsc(g_native::Geometry)

Convert a native Geometry object (with Julia arrays) to use PETSc distributed types.

This is a collective operation. Each rank calls fem2d() to get the same native
geometry, then this function converts:
- x::Matrix{Float64} -> x::Mat{Float64, MPIDENSE}
- w::Vector{Float64} -> w::Vec{Float64, MPIDENSE}
- operators[key]::SparseMatrixCSC -> operators[key]::Mat{Float64, MPIAIJ}
- subspaces[key][i]::SparseMatrixCSC -> subspaces[key][i]::Mat{Float64, MPIAIJ}

The MPIDENSE prefix indicates dense storage (for geometry data and weights),
while MPIAIJ indicates sparse storage (for operators and subspace matrices).
"""
function geometry_native_to_petsc(g_native)
    # Convert x (geometry coordinates) to MPIDENSE Mat
    x_petsc = Mat_uniform(g_native.x; Prefix=MPIDENSE)

    # Convert w (weights) to MPIDENSE Vec (weights are uniform/dense data)
    w_petsc = Vec_uniform(g_native.w; Prefix=MPIDENSE)

    # Convert all operators to MPIAIJ Mat
    # Mat_uniform distributes the uniform matrix across ranks as MPIAIJ (sparse, partitioned)
    # Sort keys to ensure deterministic order across all ranks
    operators_petsc = Dict{Symbol, Any}()
    for key in sort(collect(keys(g_native.operators)))
        op = g_native.operators[key]
        operators_petsc[key] = Mat_uniform(op; Prefix=MPIAIJ)
    end

    # Convert all subspace matrices to MPIAIJ Mat
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_petsc = Dict{Symbol, Vector{Any}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspace_vec = g_native.subspaces[key]
        petsc_vec = Vector{Any}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            petsc_vec[i] = Mat_uniform(subspace_vec[i]; Prefix=MPIAIJ)
        end
        subspaces_petsc[key] = petsc_vec
    end

    # Convert refine and coarsen vectors to MPIAIJ Mat
    refine_petsc = Vector{Any}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        refine_petsc[i] = Mat_uniform(g_native.refine[i]; Prefix=MPIAIJ)
    end

    coarsen_petsc = Vector{Any}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        coarsen_petsc[i] = Mat_uniform(g_native.coarsen[i]; Prefix=MPIAIJ)
    end

    # Determine PETSc types for Geometry type parameters
    XType = typeof(x_petsc)
    WType = typeof(w_petsc)
    MType = typeof(operators_petsc[:id])  # Use id operator as representative
    DType = typeof(g_native.discretization)

    # Create typed dicts and vectors for Geometry constructor
    operators_typed = Dict{Symbol, MType}()
    for key in keys(operators_petsc)
        operators_typed[key] = operators_petsc[key]
    end

    subspaces_typed = Dict{Symbol, Vector{MType}}()
    for key in keys(subspaces_petsc)
        subspaces_typed[key] = convert(Vector{MType}, subspaces_petsc[key])
    end

    refine_typed = convert(Vector{MType}, refine_petsc)
    coarsen_typed = convert(Vector{MType}, coarsen_petsc)

    # Create new Geometry with PETSc types using explicit type parameters
    return Geometry{Float64, XType, WType, MType, DType}(
        g_native.discretization,
        x_petsc,
        w_petsc,
        subspaces_typed,
        operators_typed,
        refine_typed,
        coarsen_typed
    )
end

# ============================================================================
# Public API
# ============================================================================

"""
    fem2d_petsc(::Type{T}=Float64; kwargs...) where {T}

Create a PETSc-based Geometry from fem2d parameters.

This function calls `fem2d(kwargs...)` to create a native geometry, then converts
it to use PETSc distributed types (Mat and Vec) for distributed computing.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem2d()`

# Returns
A Geometry object with PETSc distributed types.

# Example
```julia
g = fem2d_petsc(Float64; maxh=0.1)
```
"""
function fem2d_petsc(::Type{T}=Float64; kwargs...) where {T}
    # Create native geometry
    g_native = fem2d(; kwargs...)

    # Convert to PETSc types
    return geometry_native_to_petsc(g_native)
end

"""
    fem2d_solve(::Type{T}=Float64; kwargs...) where {T}

Solve a fem2d problem using amgb with PETSc distributed types.

This is a convenience function that combines `fem2d_petsc` and `amgb` into a
single call. It creates a PETSc-based geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem2d_petsc` and `amgb`
  - `maxh`: Maximum mesh size (passed to fem2d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem2d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem2d_solve(Float64; maxh=0.1, p=2.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem2d_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create PETSc geometry
    g = fem2d_petsc(T; kwargs...)

    # Solve using amgb
    return amgb(g; kwargs...)
end

# Export the public API
export fem2d_petsc, fem2d_solve

end # module MultiGridBarrierPETSc
