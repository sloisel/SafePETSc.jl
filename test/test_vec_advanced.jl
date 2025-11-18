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
