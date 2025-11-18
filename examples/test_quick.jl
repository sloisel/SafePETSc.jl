#!/usr/bin/env julia
# Quick test: sol = MultiGridBarrierPETSc.fem2d_solve(Float64; L=1, p=1.0, verbose=false)
# Run with: mpiexec -n 4 julia --project=. examples/test_quick.jl

using SafePETSc
SafePETSc.Init()
include("MultiGridBarrierPETSc.jl")
using .MultiGridBarrierPETSc

for L in 1:5
    println(io0(), "Testing L=$L...")
    sol = fem2d_solve(Float64; L=L, p=1.0, verbose=false)
    println(io0(), "  L=$L: Success! Solution computed.")
end

println(io0(), "\nAll tests (L=1 to L=5) completed successfully!")
