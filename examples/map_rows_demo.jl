#!/usr/bin/env julia

# Demo of map_rows functionality
# Run with: mpiexec -n 4 julia --project=. examples/map_rows_demo.jl

using MPI
using SafePETSc
using SafePETSc: MPIDENSE
SafePETSc.Init()
using LinearAlgebra

println(io0(), "\n" * "="^70)
println(io0(), "map_rows() Demonstration")
println(io0(), "="^70)

# Example data (same as in the user's examples)
B_data = [-0.143343  -0.706601   1.07202;
          -1.2582     1.74033   -0.421202;
          -1.14758    0.280491   1.17004;
          -0.319982  -1.13914   -0.523086;
          -1.58944    1.09067   -0.329869]

C_data = [0.5718337774182461,
          -0.20748982677011224,
          1.18861871474166,
          -1.6872856064399722,
          -0.016672510170726528]

# Create distributed PETSc objects
# map_rows always returns MPIDENSE, so we use MPIDENSE for inputs too
B = SafePETSc.Mat_uniform(B_data; Prefix=MPIDENSE)
C = SafePETSc.Vec_uniform(C_data)

println(io0(), "\n1. Scalar output - compute sum of each row:")
println(io0(), "   map_rows(sum, B)")
result1 = map_rows(sum, B)
println(io0(), "   Result: ", Vector(result1))

println(io0(), "\n2. Vector output - compute [sum, product] for each row:")
println(io0(), "   map_rows(x -> [sum(x), prod(x)], B)")
result2 = map_rows(x -> [sum(x), prod(x)], B)
println(io0(), "   Result (10 elements, 5 rows × 2 values): ", Vector(result2))

println(io0(), "\n3. Matrix output - compute [sum, product] as matrix rows:")
println(io0(), "   map_rows(x -> [sum(x), prod(x)]', B)")
result3 = map_rows(x -> [sum(x), prod(x)]', B)
println(io0(), "   Result (5×2 matrix):")
show(io0(), MIME("text/plain"), Matrix(result3))
println(io0(), "\n")

println(io0(), "\n4. Multiple inputs - combine matrix and vector:")
println(io0(), "   map_rows((x, y) -> [sum(x), prod(x), y[1]]', B, C)")
result4 = map_rows((x, y) -> [sum(x), prod(x), y[1]]', B, C)
println(io0(), "   Result (5×3 matrix):")
show(io0(), MIME("text/plain"), Matrix(result4))
println(io0(), "\n")

println(io0(), "\n5. Vec to Mat transformation:")
println(io0(), "   map_rows(x -> [x[1], x[1]^2, x[1]^3]', C)")
result5 = map_rows(x -> [x[1], x[1]^2, x[1]^3]', C)
println(io0(), "   Result (5×3 matrix, [value, value², value³] per row):")
show(io0(), MIME("text/plain"), Matrix(result5))
println(io0(), "\n")

println(io0(), "\n" * "="^70)
println(io0(), "All examples completed successfully!")
println(io0(), "="^70)
