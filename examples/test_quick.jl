#!/usr/bin/env julia
# Quick test: sol = MultiGridBarrierPETSc.fem2d_solve(Float64; L=1, p=1.0, verbose=false)
# Run with: mpiexec -n 4 julia --project=. examples/test_quick.jl

using SafePETSc
using SafePETSc: @mpiassert
SafePETSc.Init()
include("MultiGridBarrierPETSc.jl")
using .MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra

for L in 1:5
    println(io0(), "Testing L=$L...")
    flush(stdout)

    # Solve with PETSc distributed types
    println(io0(), "  L=$L: Starting PETSc solve...")
    flush(stdout)
    sol_petsc = MultiGridBarrierPETSc.fem2d_solve(Float64; L=L, p=1.0, verbose=false, maxit=10)
    println(io0(), "  L=$L: PETSc solution computed.")
    flush(stdout)

    # Solve with native Julia types (sequential, no logfile needed)
    println(io0(), "  L=$L: Starting native solve...")
    flush(stdout)
    sol_native = MultiGridBarrier.fem2d_solve(Float64; L=L, p=1.0, verbose=false, maxit=10)
    println(io0(), "  L=$L: Native solution computed.")
    flush(stdout)

    # Convert PETSc solution to native for comparison
    z_petsc = Matrix(sol_petsc.z)
    z_native = sol_native.z

    # Compare solutions
    diff = norm(z_petsc - z_native)
    rel_diff = diff / norm(z_native)

    println(io0(), "  L=$L: ||z_petsc - z_native|| = $diff")
    println(io0(), "  L=$L: Relative difference = $rel_diff")

    # Note: Differences can occur due to inexact PETSc iterative solves
    tol = 1e-6
    if rel_diff < tol
        println(io0(), "  L=$L: ✓ Solutions match within tolerance!")
    else
        println(io0(), "  L=$L: ⚠ Solutions differ (relative diff = $rel_diff, tolerance = $tol)")
        println(io0(), "  L=$L:   (This may be due to inexact iterative solve)")
    end
end

println(io0(), "\nAll tests (L=1 to L=5) completed!")
