#!/usr/bin/env julia

# Example integrating SafePETSc.jl with MultiGridBarrier.jl
# This demonstrates using PETSc distributed arrays with MultiGridBarrier geometry
# Run with: mpiexec -n 4 julia --project=. examples/mgb.jl

using MPI
using SafePETSc
using SafePETSc: MPIDENSE, MPIAIJ
SafePETSc.Init()
using LinearAlgebra
using SparseArrays

# Install MultiGridBarrier if needed
using Pkg
if !haskey(Pkg.project().dependencies, "MultiGridBarrier")
    println(io0(), "Installing MultiGridBarrier.jl v0.11.25...")
    Pkg.add(name="MultiGridBarrier", version="0.11.25")
end
using MultiGridBarrier

println(io0(), "\n" * "="^70)
println(io0(), "MultiGridBarrier + SafePETSc Integration Example")
println(io0(), "="^70)

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

    # Convert w (weights) to MPIDENSE Vec
    w_petsc = Vec_uniform(g_native.w)

    # Convert all operators to MPIAIJ Mat
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
            println(io0(), "About to call Mat_uniform for subspace[:$key][$i], size=$(size(subspace_vec[i]))")
            petsc_vec[i] = Mat_uniform(subspace_vec[i]; Prefix=MPIAIJ)
            println(io0(), "Successfully converted subspace[:$key][$i]")
        end
        subspaces_petsc[key] = petsc_vec
    end

    # Convert refine and coarsen vectors to MPIAIJ Mat
    refine_petsc = Vector{Any}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        println(io0(), "Converting refine[$i], size=$(size(g_native.refine[i]))")
        refine_petsc[i] = Mat_uniform(g_native.refine[i]; Prefix=MPIAIJ)
    end

    coarsen_petsc = Vector{Any}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        println(io0(), "Converting coarsen[$i], size=$(size(g_native.coarsen[i]))")
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

println(io0(), "\n" * "="^70)
println(io0(), "Phase 1: Geometry Conversion")
println(io0(), "="^70)

# Create native geometry using fem2d()
println(io0(), "\nCreating native geometry with fem2d()...")
g_native = fem2d()  # Use default parameters

println(io0(), "Native geometry properties:")
println(io0(), "  x type: ", typeof(g_native.x))
println(io0(), "  x size: ", size(g_native.x))
println(io0(), "  w type: ", typeof(g_native.w))
println(io0(), "  w length: ", length(g_native.w))
println(io0(), "  operators keys: ", collect(keys(g_native.operators)))
println(io0(), "  subspaces keys: ", collect(keys(g_native.subspaces)))
println(io0(), "  discretization: ", typeof(g_native.discretization))

# Debug: Check if operators and subspaces are mpi_uniform
println(io0(), "\nDebug: Checking if operators are mpi_uniform...")
using SafePETSc.SafeMPI: mpi_uniform
for key in sort(collect(keys(g_native.operators)))
    is_uniform = mpi_uniform(g_native.operators[key])
    println(io0(), "  operators[:$key] mpi_uniform: ", is_uniform)
end

println(io0(), "\nDebug: Writing sparse matrix internals and dense versions to files...")
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# Write the Dirichlet subspace matrices to files for inspection
for key in [:dirichlet]
    subspace_vec = g_native.subspaces[key]
    for i in 1:length(subspace_vec)
        A = subspace_vec[i]
        open("/tmp/sparse_$(key)_$(i)_rank$(rank).txt", "w") do f
            println(f, "Rank: $rank")
            println(f, "Matrix: subspaces[:$key][$i]")
            println(f, "Size: ", size(A))
            println(f, "nnz: ", nnz(A))
            println(f, "\ncolptr (length=$(length(A.colptr))):")
            println(f, A.colptr)
            println(f, "\nrowval (length=$(length(A.rowval))):")
            println(f, A.rowval)
            println(f, "\nnzval (length=$(length(A.nzval))):")
            println(f, A.nzval)
            println(f, "\n\nDENSE MATRIX:")
            println(f, "="^70)
            show(f, MIME("text/plain"), Matrix(A))
        end
    end
end

println(io0(), "Debug: Files written to /tmp/sparse_dirichlet_*_rank*.txt")

# Convert to PETSc types
println(io0(), "\nConverting to PETSc types...")
g_petsc = geometry_native_to_petsc(g_native)

println(io0(), "PETSc geometry properties:")
println(io0(), "  x type: ", typeof(g_petsc.x))
println(io0(), "  x size: ", size(g_petsc.x))
println(io0(), "  w type: ", typeof(g_petsc.w))
println(io0(), "  w length: ", length(g_petsc.w))
println(io0(), "  operators keys: ", collect(keys(g_petsc.operators)))
println(io0(), "  subspaces keys: ", collect(keys(g_petsc.subspaces)))
println(io0(), "  discretization: ", typeof(g_petsc.discretization))

# Verify data matches
println(io0(), "\nVerifying data integrity...")
x_diff = norm(Matrix(g_petsc.x) - g_native.x)
w_diff = norm(Vector(g_petsc.w) - g_native.w)

println(io0(), "  ||x_petsc - x_native|| = ", x_diff)
println(io0(), "  ||w_petsc - w_native|| = ", w_diff)

# Check all operators
max_op_diff = 0.0
for key in keys(g_native.operators)
    diff = norm(Matrix(g_petsc.operators[key]) - Matrix(g_native.operators[key]))
    println(io0(), "  ||operators[:$key]_petsc - operators[:$key]_native|| = ", diff)
    global max_op_diff = max(max_op_diff, diff)
end

# Check all subspaces
max_sub_diff = 0.0
for key in keys(g_native.subspaces)
    for i in 1:length(g_native.subspaces[key])
        diff = norm(Matrix(g_petsc.subspaces[key][i]) - Matrix(g_native.subspaces[key][i]))
        println(io0(), "  ||subspaces[:$key][$i]_petsc - subspaces[:$key][$i]_native|| = ", diff)
        global max_sub_diff = max(max_sub_diff, diff)
    end
end

if x_diff < 1e-14 && w_diff < 1e-14 && max_op_diff < 1e-14 && max_sub_diff < 1e-14
    println(io0(), "  ✓ All conversions exact to machine precision")
else
    println(io0(), "  ✗ WARNING: Conversion errors detected!")
end

println(io0(), "\n" * "="^70)
println(io0(), "Phase 1 completed successfully!")
println(io0(), "="^70)
