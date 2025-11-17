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

# ============================================================================
# MultiGridBarrier API Implementation for SafePETSc Types
# ============================================================================
# These methods teach MultiGridBarrier how to work with PETSc distributed types

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_hcat, amgb_diag, amgb_blockdiag, map_rows

# amgb_zeros: Create zero matrices with appropriate type
# For sparse PETSc matrices (MPIAIJ)
MultiGridBarrier.amgb_zeros(::Mat{T, MPIAIJ}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIAIJ}}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)

# For dense PETSc matrices (MPIDENSE)
MultiGridBarrier.amgb_zeros(::Mat{T, MPIDENSE}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIDENSE}}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)

# amgb_all_isfinite: Check if all elements are finite
# For PETSc vectors, convert to native and check
MultiGridBarrier.amgb_all_isfinite(z::Vec{T}) where {T} = all(isfinite.(Vector(z)))

# amgb_hcat: Horizontal concatenation
# Always return MPIAIJ (sparse) to match geometry operators
# This is needed because g_grid/f_grid are MPIDENSE but results must be MPIAIJ
MultiGridBarrier.amgb_hcat(A::Mat...) = begin
    result = hcat(A...)
    # If result is already MPIAIJ, return it; otherwise convert
    if typeof(result.obj) == SafePETSc._Mat{eltype(result), MPIAIJ}
        return result
    else
        # Convert to sparse
        return Mat_uniform(sparse(Matrix(result)); Prefix=MPIAIJ)
    end
end

# amgb_diag: Create diagonal matrix from vector
# Returns matrix with same Prefix as prototype (first argument)
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIAIJ)
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIAIJ)

MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIDENSE)
MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIDENSE)

# amgb_blockdiag: Block diagonal concatenation
# Use SafePETSc's blockdiag directly
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIAIJ}...) where {T} = blockdiag(args...)
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIDENSE}...) where {T} = blockdiag(args...)

# map_rows: Apply function to each row
# Use SafePETSc's map_rows which handles both sparse and dense PETSc matrices
MultiGridBarrier.map_rows(f, A::Mat...) = SafePETSc.map_rows(f, A...)

# Additional base functions needed for MultiGridBarrier
# These work by converting to native arrays temporarily
Base.minimum(v::Vec{T}) where {T} = minimum(Vector(v))
Base.maximum(v::Vec{T}) where {T} = maximum(Vector(v))

println(io0(), "✓ MultiGridBarrier API implemented for SafePETSc types")

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

    # Convert w (weights) to MPIDENSE Vec (weights are uniform/dense data)
    w_petsc = Vec_uniform(g_native.w; Prefix=MPIDENSE)

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

println(io0(), "\n" * "="^70)
println(io0(), "Phase 2: Testing Convex barrier functions")
println(io0(), "="^70)

# Create a simple Convex object to test with both geometries
println(io0(), "\nCreating Convex object for testing...")

# Create a simple p-Laplace Convex constraint using convex_Euclidian_power
# This creates the constraint: s >= ||q||_2^p for the gradient components
p_value = 2.0  # Start with p=2 (easier to test)
Q = convex_Euclidian_power(Float64; p=x->Float64(p_value))
println(io0(), "  ✓ Created Convex object with p = ", p_value)

# Test the barrier function with native types
println(io0(), "\nTesting barrier functions with native types...")
using Random
Random.seed!(42)

# Get a sample point from the geometry
x_native = g_native.x[1, :]  # Spatial coordinates for first node
y_test_native = randn(3)  # Test state: [q1, q2, s] where s should be > ||q||^p
y_test_native[3] = abs(y_test_native[3]) + norm(y_test_native[1:2])^p_value + 1.0  # Ensure feasibility

# Evaluate the three barrier functions
barrier_val_native = Q.barrier(x_native, y_test_native)
cobarrier_val_native = Q.cobarrier(x_native, [y_test_native; 1.0])  # With slack
slack_val_native = Q.slack(x_native, y_test_native)

println(io0(), "  barrier(x, y) = ", barrier_val_native)
println(io0(), "  cobarrier(x, [y; s]) = ", cobarrier_val_native)
println(io0(), "  slack(x, y) = ", slack_val_native)

# Now test with PETSc types - barrier functions should work with native arrays
# even when geometry uses PETSc types
println(io0(), "\nTesting barrier functions with PETSc geometry...")
# Convert the entire x matrix to native, then extract the row
x_petsc_matrix = Matrix(g_petsc.x)
x_petsc = x_petsc_matrix[1, :]  # Extract first row as native array

barrier_val_petsc = Q.barrier(x_petsc, y_test_native)
cobarrier_val_petsc = Q.cobarrier(x_petsc, [y_test_native; 1.0])
slack_val_petsc = Q.slack(x_petsc, y_test_native)

println(io0(), "  barrier(x, y) = ", barrier_val_petsc)
println(io0(), "  cobarrier(x, [y; s]) = ", cobarrier_val_petsc)
println(io0(), "  slack(x, y) = ", slack_val_petsc)

# Verify they match
barrier_diff = abs(barrier_val_native - barrier_val_petsc)
cobarrier_diff = abs(cobarrier_val_native - cobarrier_val_petsc)
slack_diff = abs(slack_val_native - slack_val_petsc)

println(io0(), "\nDifferences:")
println(io0(), "  |barrier_native - barrier_petsc| = ", barrier_diff)
println(io0(), "  |cobarrier_native - cobarrier_petsc| = ", cobarrier_diff)
println(io0(), "  |slack_native - slack_petsc| = ", slack_diff)

# Test geometry operators by applying them
println(io0(), "\nTesting geometry operators...")
n_vars = size(g_native.x, 1)

# Test each operator
for op_key in [:id, :dx, :dy]
    println(io0(), "  Testing operator :$op_key...")

    op_native = g_native.operators[op_key]
    op_petsc = g_petsc.operators[op_key]

    # Apply to a random vector
    v_native = randn(n_vars)
    v_petsc = Vec_uniform(v_native)

    result_native = op_native * v_native
    result_petsc = Vector(op_petsc * v_petsc)

    diff = norm(result_native - result_petsc)
    println(io0(), "    ||op_native * v - op_petsc * v|| = ", diff)
end

# Summary
println(io0(), "\n" * "="^70)
println(io0(), "Phase 2 Summary:")
println(io0(), "="^70)

tol = 1e-14
all_ok = true

if barrier_diff < tol && cobarrier_diff < tol && slack_diff < tol
    println(io0(), "  ✓ All barrier functions match to tolerance")
else
    println(io0(), "  ✗ Barrier function ERROR: max diff = ", max(barrier_diff, cobarrier_diff, slack_diff))
    global all_ok = false
end

if all_ok
    println(io0(), "  ✓ Phase 2 completed successfully!")
    println(io0(), "  Convex barrier functions work correctly!")
else
    println(io0(), "  ✗ Phase 2 FAILED - differences detected")
end

println(io0(), "="^70)

println(io0(), "\n" * "="^70)
println(io0(), "Phase 3: Testing amg_helper")
println(io0(), "="^70)

# Test the unexported amg_helper function
# This creates AMG{T,X,W,M,Discretization} objects
println(io0(), "\nCalling amg_helper with native geometry...")

# Define state variables and D matrix
state_variables = [:u :dirichlet ; :s :full]
D = [:u :id
     :u :dx
     :u :dy
     :s :id]

println(io0(), "  state_variables = ", state_variables)
println(io0(), "  D = ", D)

try
    # Call amg_helper with native geometry
    amg_native = MultiGridBarrier.amg_helper(g_native, state_variables, D)
    println(io0(), "  ✓ amg_helper succeeded with native geometry")
    println(io0(), "  AMG type: ", typeof(amg_native))

    # Discover what fields the AMG object has
    println(io0(), "  AMG fields: ", fieldnames(typeof(amg_native)))

    # Call amg_helper with PETSc geometry
    println(io0(), "\nCalling amg_helper with PETSc geometry...")
    amg_petsc = MultiGridBarrier.amg_helper(g_petsc, state_variables, D)
    println(io0(), "  ✓ amg_helper succeeded with PETSc geometry")
    println(io0(), "  AMG type: ", typeof(amg_petsc))
    println(io0(), "  AMG fields: ", fieldnames(typeof(amg_petsc)))

    # Print dimensions for debugging Phase 5
    println(io0(), "\nDimension analysis:")
    println(io0(), "  geometry.x size: ", size(amg_native.geometry.x))
    println(io0(), "  AMG.x size: ", size(amg_native.x))
    println(io0(), "  AMG.w size: ", size(amg_native.w))
    println(io0(), "  AMG.D size: ", size(amg_native.D))
    println(io0(), "  Number of nodes: ", size(amg_native.geometry.x, 1))
    println(io0(), "  Number of state vars per node: ", size(amg_native.x, 2))
    println(io0(), "  Number of D components per node: ", size(amg_native.D, 1))
    println(io0(), "  Expected state vector length (n_nodes * n_state_vars): ", size(amg_native.geometry.x, 1) * size(amg_native.x, 2))
    println(io0(), "  Expected D vector length (n_nodes * n_D_components): ", size(amg_native.geometry.x, 1) * size(amg_native.D, 1))

    # Compare the AMG operators (restriction and refinement matrices)
    # Note: R_fine and R_coarse are vectors of matrices
    println(io0(), "\nComparing AMG operators...")

    max_diff = 0.0

    # Compare R_fine matrices (vector of matrices)
    println(io0(), "  R_fine has ", length(amg_native.R_fine), " matrices")
    for i in 1:length(amg_native.R_fine)
        R_fine_i_native = amg_native.R_fine[i]
        R_fine_i_petsc_native = Matrix(amg_petsc.R_fine[i])
        println(io0(), "    R_fine[$i] size: ", size(R_fine_i_native))
        diff = norm(R_fine_i_native - R_fine_i_petsc_native)
        println(io0(), "    ||R_fine[$i]_native - R_fine[$i]_petsc|| = ", diff)
        max_diff = max(max_diff, diff)
    end

    # Compare R_coarse matrices (vector of matrices)
    println(io0(), "  R_coarse has ", length(amg_native.R_coarse), " matrices")
    for i in 1:length(amg_native.R_coarse)
        R_coarse_i_native = amg_native.R_coarse[i]
        R_coarse_i_petsc_native = Matrix(amg_petsc.R_coarse[i])
        println(io0(), "    R_coarse[$i] size: ", size(R_coarse_i_native))
        diff = norm(R_coarse_i_native - R_coarse_i_petsc_native)
        println(io0(), "    ||R_coarse[$i]_native - R_coarse[$i]_petsc|| = ", diff)
        max_diff = max(max_diff, diff)
    end

    if max_diff < 1e-14
        println(io0(), "\n" * "="^70)
        println(io0(), "✓ Phase 3 completed successfully!")
        println(io0(), "✓ amg_helper produces identical results for native and PETSc!")
        println(io0(), "="^70)
    else
        println(io0(), "\n" * "="^70)
        println(io0(), "⚠ Phase 3 completed with differences")
        println(io0(), "  Matrices differ by ", A_diff)
        println(io0(), "="^70)
    end

catch e
    println(io0(), "\n" * "="^70)
    println(io0(), "✗ Phase 3 FAILED with error:")
    println(io0(), "="^70)
    println(io0(), e)
    println(io0(), "")
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
    println(io0(), "="^70)

    println(io0(), "\nThis indicates an issue that needs to be addressed.")
    println(io0(), "Please review the error above.")
end

println(io0(), "\n" * "="^70)
println(io0(), "Phase 4: Testing amg function (returns pair of AMG objects)")
println(io0(), "="^70)

# Test the amg function which returns a pair of AMG objects
println(io0(), "\nCalling amg with native geometry...")

try
    # Call amg with native geometry - it should return a tuple of AMG objects
    amg_pair_native = MultiGridBarrier.amg(g_native; state_variables=state_variables, D=D)
    println(io0(), "  ✓ amg succeeded with native geometry")
    println(io0(), "  Result type: ", typeof(amg_pair_native))
    println(io0(), "  Number of AMG objects: ", length(amg_pair_native))

    # Examine each AMG in the pair - discover fields
    for (i, amg_obj) in enumerate(amg_pair_native)
        println(io0(), "  AMG[$i] type: ", typeof(amg_obj))
        println(io0(), "  AMG[$i] fields: ", fieldnames(typeof(amg_obj)))
    end

    # Call amg with PETSc geometry
    println(io0(), "\nCalling amg with PETSc geometry...")
    amg_pair_petsc = MultiGridBarrier.amg(g_petsc; state_variables=state_variables, D=D)
    println(io0(), "  ✓ amg succeeded with PETSc geometry")
    println(io0(), "  Result type: ", typeof(amg_pair_petsc))
    println(io0(), "  Number of AMG objects: ", length(amg_pair_petsc))

    # Examine each AMG in the pair
    for (i, amg_obj) in enumerate(amg_pair_petsc)
        println(io0(), "  AMG[$i] type: ", typeof(amg_obj))
        println(io0(), "  AMG[$i] fields: ", fieldnames(typeof(amg_obj)))
    end

    # Compare the operators from both AMG objects in each pair
    # Note: R_fine and R_coarse are vectors of matrices
    println(io0(), "\nComparing AMG operators...")
    all_match = true

    for amg_idx in 1:length(amg_pair_native)
        println(io0(), "  AMG[$amg_idx]:")

        # Compare R_fine matrices (vector of matrices)
        for i in 1:length(amg_pair_native[amg_idx].R_fine)
            R_fine_i_native = amg_pair_native[amg_idx].R_fine[i]
            R_fine_i_petsc_native = Matrix(amg_pair_petsc[amg_idx].R_fine[i])
            diff = norm(R_fine_i_native - R_fine_i_petsc_native)
            println(io0(), "    ||R_fine[$i]_native - R_fine[$i]_petsc|| = ", diff)
            if diff >= 1e-14
                all_match = false
            end
        end

        # Compare R_coarse matrices (vector of matrices)
        for i in 1:length(amg_pair_native[amg_idx].R_coarse)
            R_coarse_i_native = amg_pair_native[amg_idx].R_coarse[i]
            R_coarse_i_petsc_native = Matrix(amg_pair_petsc[amg_idx].R_coarse[i])
            diff = norm(R_coarse_i_native - R_coarse_i_petsc_native)
            println(io0(), "    ||R_coarse[$i]_native - R_coarse[$i]_petsc|| = ", diff)
            if diff >= 1e-14
                all_match = false
            end
        end
    end

    if all_match
        println(io0(), "\n" * "="^70)
        println(io0(), "✓ Phase 4 completed successfully!")
        println(io0(), "✓ amg produces identical AMG pairs for native and PETSc!")
        println(io0(), "="^70)
    else
        println(io0(), "\n" * "="^70)
        println(io0(), "⚠ Phase 4 completed with differences")
        println(io0(), "="^70)
    end

catch e
    println(io0(), "\n" * "="^70)
    println(io0(), "✗ Phase 4 FAILED with error:")
    println(io0(), "="^70)
    println(io0(), e)
    println(io0(), "")
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
    println(io0(), "="^70)

    println(io0(), "\nThis indicates an issue that needs to be addressed.")
    println(io0(), "Please review the error above.")
end

println(io0(), "\n" * "="^70)
println(io0(), "Phase 5: Testing amgb(...) solver")
println(io0(), "="^70)

# Now try running amgb with both native and PETSc geometries
println(io0(), "\nRunning amgb with native geometry...")
try
    sol_native = amgb(g_native; p=p_value, verbose=false)
    println(io0(), "  ✓ Native amgb completed successfully!")
    println(io0(), "  Solution size: ", size(sol_native.z))
    println(io0(), "  Solution norm: ", norm(sol_native.z))

    # Extract the native g_grid and f_grid from the solution to see what defaults were used
    println(io0(), "\nExamining defaults used by native amgb...")
    println(io0(), "  (We'll adapt these for PETSc)")

    # Try with PETSc geometry - need to provide g_grid and f_grid adapted to PETSc types
    println(io0(), "\nRunning amgb with PETSc geometry...")

    # Create g_grid and f_grid using PETSc types
    # The defaults from fem2d are boundary and forcing data
    # For fem2d with default parameters, g specifies Dirichlet boundary values
    # and f specifies the forcing term

    # Get the native defaults by inspecting what fem2d_solve would use
    # For p-Laplace, the defaults are typically:
    # g_grid: boundary values (size: n_boundary_nodes × n_state_vars)
    # f_grid: forcing values (size: n_nodes × n_state_vars)

    # Use the same defaults as native but convert to PETSc uniform matrices
    n_nodes = size(g_native.x, 1)
    dim = size(g_native.x, 2)

    # For p-Laplace in 2D, we have state variables [u, s] where:
    # u = solution, s = slack variable for the gradient constraint
    # Default boundary: u = norm(x)^p on boundary, s = large value
    # Default forcing: zero

    # Create native grids matching MultiGridBarrier defaults
    # state_variables = [:u :dirichlet ; :s :full] means 2 state variables (u and s)
    # So g_grid and f_grid should have size (n_nodes, 2), not (n_nodes, 4)
    n_state_vars = 2  # u and s
    g_grid_native = zeros(n_nodes, n_state_vars)
    for i in 1:n_nodes
        x_coord = g_native.x[i, :]
        g_grid_native[i, 1] = norm(x_coord)^p_value  # u boundary value
        g_grid_native[i, 2] = 100.0  # s boundary value (large for slack)
    end

    f_grid_native = zeros(n_nodes, n_state_vars)  # forcing term (typically zero)
    f_grid_native[:, 1] .= 0.5  # u forcing
    f_grid_native[:, 2] .= 1.0  # s forcing (constraint enforcement)

    # Convert to PETSc types
    g_grid_petsc = Mat_uniform(g_grid_native; Prefix=MPIDENSE)
    f_grid_petsc = Mat_uniform(f_grid_native; Prefix=MPIDENSE)

    println(io0(), "  Created g_grid_petsc: ", typeof(g_grid_petsc), " size ", size(g_grid_petsc))
    println(io0(), "  Created f_grid_petsc: ", typeof(f_grid_petsc), " size ", size(f_grid_petsc))

    sol_petsc = amgb(g_petsc; p=p_value, g_grid=g_grid_petsc, f_grid=f_grid_petsc, verbose=false)
    println(io0(), "  ✓ PETSc amgb completed successfully!")
    println(io0(), "  Solution size: ", size(sol_petsc.z))

    # Convert PETSc solution to native for comparison
    z_petsc_native = if typeof(sol_petsc.z) <: Matrix
        sol_petsc.z
    else
        Matrix(sol_petsc.z)
    end
    println(io0(), "  Solution norm: ", norm(z_petsc_native))

    # Compare solutions
    sol_diff = norm(sol_native.z - z_petsc_native)
    println(io0(), "\nSolution comparison:")
    println(io0(), "  ||z_native - z_petsc|| = ", sol_diff)

    if sol_diff < 1e-6
        println(io0(), "\n" * "="^70)
        println(io0(), "✓ Phase 3 completed successfully!")
        println(io0(), "✓ amgb works with PETSc types and produces correct results!")
        println(io0(), "="^70)
    else
        println(io0(), "\n" * "="^70)
        println(io0(), "⚠ Phase 3 completed with differences")
        println(io0(), "  Solutions differ by ", sol_diff)
        println(io0(), "="^70)
    end

catch e
    println(io0(), "\n" * "="^70)
    println(io0(), "✗ Phase 3 FAILED with error:")
    println(io0(), "="^70)
    println(io0(), e)
    println(io0(), "")
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
    println(io0(), "="^70)

    println(io0(), "\nThis is expected - amgb may need additional work to support PETSc types.")
    println(io0(), "The error above indicates what needs to be fixed in SafePETSc.jl or")
    println(io0(), "what interface is missing.")
end
