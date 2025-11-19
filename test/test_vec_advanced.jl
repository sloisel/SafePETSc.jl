using Test
using MPI
using SafePETSc
SafePETSc.Init()
using LinearAlgebra
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

ts = @testset MPITestHarness.QuietTestSet "Advanced vector operations" begin
    println(io0(), "[DEBUG] Advanced vector operations test starting")

    # Test 1: Inner product v' * w (dot product)
    println(io0(), "[DEBUG] Test 1: Inner product v' * w")
    a = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    b = Vec_uniform([5.0, 6.0, 7.0, 8.0])
    dot_product = a' * b
    @test dot_product ≈ 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0
    @test dot_product ≈ 70.0
    println(io0(), "[DEBUG] Test 1 passed")

    # Test 2: Row vector times transposed matrix v' * A'
    println(io0(), "[DEBUG] Test 2: Row vector times transposed matrix v' * A'")
    M = Mat_uniform([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])
    c = Vec_uniform([1.0, 2.0, 3.0])
    result = c' * M'
    # c' * M' = (M * c)'
    expected_vec = M * c
    @test Vector(parent(result)) ≈ Vector(expected_vec)
    println(io0(), "[DEBUG] Test 2 passed")

    # Test 3: Size and shape queries for adjoint vectors
    println(io0(), "[DEBUG] Test 3: Size and shape queries for adjoint vectors")
    p = Vec_uniform([1.0, 2.0, 3.0, 4.0, 5.0])
    pt = p'
    @test size(pt) == (1, 5)
    @test size(pt, 1) == 1
    @test size(pt, 2) == 5
    @test size(pt, 3) == 1  # Higher dimensions default to 1
    @test length(pt) == 5
    @test axes(pt) == (Base.OneTo(1), Base.OneTo(5))
    println(io0(), "[DEBUG] Test 3 passed")

    # Test 4: Size queries for DRef-wrapped Vec
    println(io0(), "[DEBUG] Test 4: Size queries for DRef-wrapped Vec")
    q = Vec_uniform([1.0, 2.0, 3.0])
    @test size(q.obj, 1) == 3
    @test size(q.obj, 2) == 1  # Vectors are 1D, second dimension is 1
    @test axes(q.obj) == (Base.OneTo(3),)
    @test length(q.obj) == 3
    @test eltype(q.obj) == Float64
    println(io0(), "[DEBUG] Test 4 passed")

    # Test 5: LinearAlgebra.dot function
    println(io0(), "[DEBUG] Test 5: LinearAlgebra.dot function")
    x = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    y = Vec_uniform([5.0, 6.0, 7.0, 8.0])
    dot_result = LinearAlgebra.dot(x, y)
    @test dot_result ≈ 70.0
    @test dot_result ≈ x' * y  # Should match inner product
    println(io0(), "[DEBUG] Test 5 passed")

    # Test 6: eachrow iterator for Vec
    println(io0(), "[DEBUG] Test 6: eachrow iterator for Vec")
    v = Vec_uniform([10.0, 20.0, 30.0, 40.0])
    iter = eachrow(v)
    # length(iter) returns the LOCAL number of rows
    local_len = length(iter)
    @test local_len >= 0  # Each rank may have 0 or more elements
    # Test manual iteration - count should match length
    count = 0
    for row in iter
        count += 1
        @test length(row) == 1  # Each row is 1-element view
    end
    @test count == local_len  # Iterations should match local length
    println(io0(), "[DEBUG] Test 6 passed")

    # Test 7: Base.sum function
    println(io0(), "[DEBUG] Test 7: Base.sum function")
    s = Vec_uniform([1.0, 2.0, 3.0, 4.0, 5.0])
    sum_result = sum(s)
    @test sum_result ≈ 15.0  # 1+2+3+4+5 = 15
    # Test with negative values
    t = Vec_uniform([-1.0, 2.0, -3.0, 4.0])
    @test sum(t) ≈ 2.0  # -1+2-3+4 = 2
    # Test with zeros
    u = Vec_uniform([0.0, 0.0, 0.0])
    @test sum(u) ≈ 0.0
    # Test with single element
    w = Vec_uniform([42.0])
    @test sum(w) ≈ 42.0
    println(io0(), "[DEBUG] Test 7 passed")

    # Test 8: Scalar multiplication with vectors α * v
    println(io0(), "[DEBUG] Test 8: Scalar multiplication α * v and v * α")
    v1 = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    v2 = 3.0 * v1
    @test Vector(v2) ≈ [3.0, 6.0, 9.0, 12.0]
    v3 = v1 * 3.0
    @test Vector(v3) ≈ [3.0, 6.0, 9.0, 12.0]
    # Test with negative scalar
    v4 = -2.0 * v1
    @test Vector(v4) ≈ [-2.0, -4.0, -6.0, -8.0]
    # Test with zero scalar
    v5 = 0.0 * v1
    @test Vector(v5) ≈ [0.0, 0.0, 0.0, 0.0]
    println(io0(), "[DEBUG] Test 8 passed")

    # Test 9: Scalar multiplication with adjoint vectors α * vt and vt * α
    println(io0(), "[DEBUG] Test 9: Scalar multiplication α * vt and vt * α")
    v6 = Vec_uniform([1.0, 2.0, 3.0])
    vt1 = v6'
    vt2 = 2.0 * vt1
    @test Vector(parent(vt2)) ≈ [2.0, 4.0, 6.0]
    vt3 = vt1 * 2.0
    @test Vector(parent(vt3)) ≈ [2.0, 4.0, 6.0]
    # Test with fractional scalar
    vt4 = 0.5 * vt1
    @test Vector(parent(vt4)) ≈ [0.5, 1.0, 1.5]
    println(io0(), "[DEBUG] Test 9 passed")

    # Test 10: Addition of adjoint vectors (row vectors)
    println(io0(), "[DEBUG] Test 10: Addition of adjoint vectors vt1 + vt2")
    a1 = Vec_uniform([1.0, 2.0, 3.0])
    a2 = Vec_uniform([4.0, 5.0, 6.0])
    sum_adjoint = a1' + a2'
    @test Vector(parent(sum_adjoint)) ≈ [5.0, 7.0, 9.0]
    # Test with subtraction-like addition (negative values)
    a3 = Vec_uniform([-1.0, -2.0, -3.0])
    result_adj = a1' + a3'
    @test Vector(parent(result_adj)) ≈ [0.0, 0.0, 0.0]
    println(io0(), "[DEBUG] Test 10 passed")

    # Test 11: Outer product v * w' (returns Mat)
    println(io0(), "[DEBUG] Test 11: Outer product v * w'")
    # Note: Both vectors must have same length for compatible row partitions in MPI
    v_outer = Vec_uniform([1.0, 2.0, 3.0])
    w_outer = Vec_uniform([4.0, 5.0, 6.0])
    outer_mat = v_outer * w_outer'
    # Outer product: [1,2,3]' * [4,5,6] = [[4,5,6],[8,10,12],[12,15,18]]
    outer_mat_dense = Matrix(outer_mat)
    @test size(outer_mat_dense) == (3, 3)
    @test outer_mat_dense ≈ [4.0 5.0 6.0; 8.0 10.0 12.0; 12.0 15.0 18.0]
    # Test outer product with 4-element vectors
    v_outer2 = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    w_outer2 = Vec_uniform([10.0, 20.0, 30.0, 40.0])
    outer_mat2 = v_outer2 * w_outer2'
    @test size(Matrix(outer_mat2)) == (4, 4)
    @test Matrix(outer_mat2)[1, :] ≈ [10.0, 20.0, 30.0, 40.0]
    @test Matrix(outer_mat2)[2, :] ≈ [20.0, 40.0, 60.0, 80.0]
    println(io0(), "[DEBUG] Test 11 passed")

    println(io0(), "[DEBUG] All advanced vector operations tests completed")
end

# Aggregate per-rank counts and print a single summary on root
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: Advanced vector operations (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
