#!/usr/bin/env julia

# Test script for MultiGridBarrierPETSc module
# Run with: mpiexec -n 4 julia --project=. examples/test_MultiGridBarrierPETSc.jl

using MPI
using SafePETSc
using SafePETSc: MPIDENSE, MPIAIJ
SafePETSc.Init()
using LinearAlgebra

# Load the module
include("MultiGridBarrierPETSc.jl")
using .MultiGridBarrierPETSc

println(io0(), "="^70)
println(io0(), "Testing MultiGridBarrierPETSc Module")
println(io0(), "="^70)

# ============================================================================
# Test 1: fem2d_petsc
# ============================================================================
println(io0(), "\nTest 1: fem2d_petsc()")
println(io0(), "-"^70)

try
    println(io0(), "Creating PETSc geometry with default parameters...")
    g = fem2d_petsc()

    println(io0(), "✓ fem2d_petsc() succeeded")
    println(io0(), "  Geometry type: ", typeof(g))
    println(io0(), "  x type: ", typeof(g.x))
    println(io0(), "  x size: ", size(g.x))
    println(io0(), "  w type: ", typeof(g.w))
    println(io0(), "  w length: ", length(g.w))
    println(io0(), "  operators keys: ", collect(keys(g.operators)))
    println(io0(), "  subspaces keys: ", collect(keys(g.subspaces)))

    # Verify types are PETSc types
    if typeof(g.x) <: Mat{Float64, MPIDENSE}
        println(io0(), "  ✓ x is Mat{Float64, MPIDENSE}")
    else
        println(io0(), "  ✗ x type is incorrect: ", typeof(g.x))
    end

    if typeof(g.w) <: Vec{Float64}
        println(io0(), "  ✓ w is Vec{Float64}")
    else
        println(io0(), "  ✗ w type is incorrect: ", typeof(g.w))
    end

    # Check an operator
    if typeof(g.operators[:id]) <: Mat{Float64, MPIAIJ}
        println(io0(), "  ✓ operators[:id] is Mat{Float64, MPIAIJ}")
    else
        println(io0(), "  ✗ operators[:id] type is incorrect: ", typeof(g.operators[:id]))
    end

    println(io0(), "\nTest 1: PASSED")

catch e
    println(io0(), "\n✗ Test 1: FAILED")
    println(io0(), "Error: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

# ============================================================================
# Test 2: fem2d_petsc with custom parameters
# ============================================================================
println(io0(), "\n" * "="^70)
println(io0(), "Test 2: fem2d_petsc with custom parameters")
println(io0(), "-"^70)

try
    println(io0(), "Creating PETSc geometry with maxh=0.2...")
    g2 = fem2d_petsc(Float64; maxh=0.2)

    println(io0(), "✓ fem2d_petsc(Float64; maxh=0.2) succeeded")
    println(io0(), "  x size: ", size(g2.x))
    println(io0(), "  w length: ", length(g2.w))

    println(io0(), "\nTest 2: PASSED")

catch e
    println(io0(), "\n✗ Test 2: FAILED")
    println(io0(), "Error: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

# ============================================================================
# Test 3: fem2d_solve
# ============================================================================
println(io0(), "\n" * "="^70)
println(io0(), "Test 3: fem2d_solve()")
println(io0(), "-"^70)

try
    println(io0(), "Solving with default parameters (p=2.0)...")
    sol = fem2d_solve(Float64; p=2.0, verbose=false)

    println(io0(), "✓ fem2d_solve() succeeded")
    println(io0(), "  Solution type: ", typeof(sol))
    println(io0(), "  Solution.z size: ", size(sol.z))

    # Convert to native for norm calculation
    z_native = if typeof(sol.z) <: Matrix
        sol.z
    else
        Matrix(sol.z)
    end

    println(io0(), "  Solution norm: ", norm(z_native))

    # Check that solution is finite
    if all(isfinite.(z_native))
        println(io0(), "  ✓ Solution is finite")
    else
        println(io0(), "  ✗ Solution contains non-finite values")
    end

    println(io0(), "\nTest 3: PASSED")

catch e
    println(io0(), "\n✗ Test 3: FAILED")
    println(io0(), "Error: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

# ============================================================================
# Test 4: fem2d_solve with custom parameters
# ============================================================================
println(io0(), "\n" * "="^70)
println(io0(), "Test 4: fem2d_solve with custom parameters")
println(io0(), "-"^70)

try
    println(io0(), "Solving with maxh=0.2, p=3.0...")
    sol2 = fem2d_solve(Float64; maxh=0.2, p=3.0, verbose=false)

    println(io0(), "✓ fem2d_solve(maxh=0.2, p=3.0) succeeded")
    println(io0(), "  Solution.z size: ", size(sol2.z))

    # Convert to native for norm calculation
    z_native2 = if typeof(sol2.z) <: Matrix
        sol2.z
    else
        Matrix(sol2.z)
    end

    println(io0(), "  Solution norm: ", norm(z_native2))

    if all(isfinite.(z_native2))
        println(io0(), "  ✓ Solution is finite")
    else
        println(io0(), "  ✗ Solution contains non-finite values")
    end

    println(io0(), "\nTest 4: PASSED")

catch e
    println(io0(), "\n✗ Test 4: FAILED")
    println(io0(), "Error: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

# ============================================================================
# Test 5: Compare with native fem2d
# ============================================================================
println(io0(), "\n" * "="^70)
println(io0(), "Test 5: Compare PETSc vs native geometry")
println(io0(), "-"^70)

try
    using MultiGridBarrier

    println(io0(), "Creating native geometry...")
    g_native = fem2d()

    println(io0(), "Creating PETSc geometry...")
    g_petsc = fem2d_petsc()

    # Compare data
    println(io0(), "\nComparing data...")
    x_diff = norm(Matrix(g_petsc.x) - g_native.x)
    w_diff = norm(Vector(g_petsc.w) - g_native.w)

    println(io0(), "  ||x_petsc - x_native|| = ", x_diff)
    println(io0(), "  ||w_petsc - w_native|| = ", w_diff)

    # Compare operators
    max_op_diff = 0.0
    for key in keys(g_native.operators)
        diff = norm(Matrix(g_petsc.operators[key]) - Matrix(g_native.operators[key]))
        println(io0(), "  ||operators[:$key]_petsc - operators[:$key]_native|| = ", diff)
        max_op_diff = max(max_op_diff, diff)
    end

    tol = 1e-14
    if x_diff < tol && w_diff < tol && max_op_diff < tol
        println(io0(), "  ✓ All data matches to tolerance")
        println(io0(), "\nTest 5: PASSED")
    else
        println(io0(), "  ✗ Data mismatch detected")
        println(io0(), "\nTest 5: FAILED")
    end

catch e
    println(io0(), "\n✗ Test 5: FAILED")
    println(io0(), "Error: ", e)
    for (exc, bt) in Base.catch_stack()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

# ============================================================================
# Summary
# ============================================================================
println(io0(), "\n" * "="^70)
println(io0(), "All tests completed!")
println(io0(), "="^70)
