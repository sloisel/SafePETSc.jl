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
    println(io0(), "Testing adjoint conversions: sparse(Mat') = sparse(Mat)'")
    println(io0(), "=" ^ 60)

    # Test 1: Vec' to Vector
    println(io0(), "\n[Test 1] Vector(Vec') = Vector(Vec)'")
    v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    v_adj_converted = Vector(v')
    v_expected = Vector(v)'
    if rank == 0
        @assert v_adj_converted == v_expected "Vector(Vec') should equal Vector(Vec)'"
        println("✓ Vector(Vec') = Vector(Vec)' passed")
        println("   Result type: ", typeof(v_adj_converted))
        println("   Result value: ", v_adj_converted)
    end

    # Test 2: Mat' to Matrix
    println(io0(), "\n[Test 2] Matrix(Mat') = Matrix(Mat)'")
    A_data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
    A = Mat_uniform(A_data)
    A_adj_converted = Matrix(A')
    A_expected = Matrix(A)'
    if rank == 0
        @assert A_adj_converted == A_expected "Matrix(Mat') should equal Matrix(Mat)'"
        println("✓ Matrix(Mat') = Matrix(Mat)' passed")
        println("   Result type: ", typeof(A_adj_converted))
        println("   Result size: ", size(A_adj_converted))
    end

    # Test 3: Mat' to sparse
    println(io0(), "\n[Test 3] sparse(Mat') = sparse(Mat)'")
    B_data = [2.0 1.0 0.0 0.0;
              1.0 2.0 1.0 0.0;
              0.0 1.0 2.0 1.0;
              0.0 0.0 1.0 2.0]
    B = Mat_uniform(B_data)
    B_adj_converted = sparse(B')
    B_expected = sparse(B)'
    if rank == 0
        @assert B_adj_converted == B_expected "sparse(Mat') should equal sparse(Mat)'"
        println("✓ sparse(Mat') = sparse(Mat)' passed")
        println("   Result type: ", typeof(B_adj_converted))
        println("   Result size: ", size(B_adj_converted))
        println("   Result nnz: ", nnz(B_adj_converted))
    end

    # Test 4: Verify actual transpose of sparse matrix
    println(io0(), "\n[Test 4] Verify sparse transpose correctness")
    C_data = [1.0 2.0 3.0;
              4.0 5.0 6.0]
    C = Mat_uniform(C_data)
    C_adj = sparse(C')
    if rank == 0
        expected_transpose = sparse(C_data')
        @assert C_adj == expected_transpose "Transpose should match Julia's expected result"
        println("✓ Sparse transpose correctness verified")
        println("   Original size: ", size(sparse(C)))
        println("   Transpose size: ", size(C_adj))
        @assert size(C_adj) == (3, 2) "Transpose should have swapped dimensions"
    end

    # Clean up
    SafePETSc.SafeMPI.check_and_destroy!()

    println(io0(), "\n" * "=" ^ 60)
    println(io0(), "All adjoint conversion tests passed!")
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

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
