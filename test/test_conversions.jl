using MPI
using SafePETSc
using PETSc
using SparseArrays
using LinearAlgebra

MPI.Init()
PETSc.initialize()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

try
    println(io0(), "=" ^ 60)
    println(io0(), "Testing show() and conversions for Vec and Mat")
    println(io0(), "=" ^ 60)

    # Test 1: Vec show (collective, displays on all ranks)
    println(io0(), "\n[Test 1] Vec show (collective, displays data)")
    v = Vec_uniform(collect(1.0:8.0))
    println(io0(), "Vec (showing via io0()): ", v)

    # Test 2: Vector conversion
    println(io0(), "\n[Test 2] Vector(v) explicit conversion")
    v_julia = Vector(v)
    if rank == 0
        @assert v_julia == collect(1.0:8.0) "Vector conversion failed"
        println("✓ Vector conversion passed")
    end

    # Test 3: Dense Mat show (collective, displays on all ranks)
    println(io0(), "\n[Test 3] Dense Mat show (collective, displays data)")
    A_data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    A = Mat_uniform(A_data)
    println(io0(), "Dense Mat (showing via io0()):")
    println(io0(), A)

    # Test 4: Matrix conversion
    println(io0(), "\n[Test 4] Matrix(A) explicit conversion")
    A_julia = Matrix(A)
    if rank == 0
        @assert A_julia == A_data "Matrix conversion failed"
        println("✓ Matrix conversion passed")
    end

    # Test 5: Sparse Mat show
    println(io0(), "\n[Test 5] Sparse Mat show (collective, displays data)")
    B_data = Matrix([2.0 1.0 0.0 0.0;
                     1.0 2.0 1.0 0.0;
                     0.0 1.0 2.0 1.0;
                     0.0 0.0 1.0 2.0])
    B_sparse = sparse(B_data)
    B = Mat_uniform(B_data)
    println(io0(), "Sparse Mat (showing via io0()):")
    println(io0(), B)

    # Test 6: sparse conversion
    println(io0(), "\n[Test 6] sparse(B) explicit conversion")
    B_julia = sparse(B)
    if rank == 0
        @assert B_julia == B_sparse "Sparse conversion failed"
        println("✓ sparse conversion passed")
    end

    # Test 7: Demonstrate io0() usage
    println(io0(), "\n[Test 7] io0() helper function")
    println(io0(), "This line prints only on rank 0 (via io0())")
    println(io0(r=Set([1])), "This line prints only on rank 1 (via io0(r=Set([1])))")
    println(io0(), "✓ io0() helper works correctly")

    # Test 8: Vector adjoint conversion
    println(io0(), "\n[Test 8] Vector(v') adjoint conversion")
    v_adj = v'
    v_adj_julia = Vector(v_adj)
    if rank == 0
        @assert v_adj_julia == collect(1.0:8.0)' "Vector adjoint conversion failed"
        println("✓ Vector adjoint conversion passed")
    end

    # Test 9: Matrix adjoint conversion
    println(io0(), "\n[Test 9] Matrix(A') adjoint conversion")
    A_adj = A'
    A_adj_julia = Matrix(A_adj)
    if rank == 0
        @assert A_adj_julia == A_data' "Matrix adjoint conversion failed"
        println("✓ Matrix adjoint conversion passed")
    end

    # Test 10: sparse adjoint conversion
    println(io0(), "\n[Test 10] sparse(B') adjoint conversion")
    B_adj = B'
    B_adj_julia = sparse(B_adj)
    if rank == 0
        @assert B_adj_julia == B_sparse' "sparse adjoint conversion failed"
        println("✓ sparse adjoint conversion passed")
    end

    # Clean up - objects will be finalized automatically when they go out of scope
    # User can call GC.gc() if they want to accelerate resource recovery
    # Then check_and_destroy!() performs the actual collective cleanup
    SafePETSc.SafeMPI.check_and_destroy!()

    println(io0(), "\n" * "=" ^ 60)
    println(io0(), "All tests passed!")
    println(io0(), "=" ^ 60)

catch e
    if rank == 0
        println("Error during testing: ", e)
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    rethrow(e)
finally
    PETSc.finalize()
    MPI.Finalize()
end
