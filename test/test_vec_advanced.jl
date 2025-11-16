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
