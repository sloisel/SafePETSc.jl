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
    println(io0(), "Testing J() conversion function")
    println(io0(), "=" ^ 60)

    # Test 1: J(Vec) -> Vector
    println(io0(), "\n[Test 1] J(Vec) -> Vector")
    v = Vec_uniform([1.0, 2.0, 3.0, 4.0])
    v_julia = J(v)
    if rank == 0
        @assert v_julia == [1.0, 2.0, 3.0, 4.0] "J(Vec) failed"
        @assert v_julia isa Vector{Float64} "J(Vec) should return Vector"
        println("✓ J(Vec) = Vector(Vec) passed")
        println("   Type: ", typeof(v_julia))
    end

    # Test 2: J(Vec') -> adjoint Vector
    println(io0(), "\n[Test 2] J(Vec') -> Adjoint Vector")
    v_adj = v'
    v_adj_julia = J(v_adj)
    if rank == 0
        @assert v_adj_julia == [1.0, 2.0, 3.0, 4.0]' "J(Vec') failed"
        @assert v_adj_julia isa Adjoint{Float64, Vector{Float64}} "J(Vec') should return Adjoint Vector"
        println("✓ J(Vec') = Vector(Vec)' passed")
        println("   Type: ", typeof(v_adj_julia))
    end

    # Test 3: J(Mat) -> appropriate type based on is_dense
    println(io0(), "\n[Test 3] J(Mat) returns correct type")
    A_data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    A = Mat_uniform(A_data)
    A_julia = J(A)
    if rank == 0
        # J should return Matrix if dense, sparse otherwise
        expected_type = SafePETSc.is_dense(A) ? Matrix(A) : sparse(A)
        @assert A_julia == expected_type "J(Mat) failed"
        println("✓ J(Mat) returns correct type based on is_dense")
        println("   is_dense: ", SafePETSc.is_dense(A))
        println("   Type: ", typeof(A_julia))
    end

    # Test 4: J(Mat') -> adjoint of appropriate type
    println(io0(), "\n[Test 4] J(Mat') returns correct adjoint type")
    A_adj = A'
    A_adj_julia = J(A_adj)
    if rank == 0
        expected_adj = SafePETSc.is_dense(parent(A_adj)) ? Matrix(A_adj) : sparse(A_adj)
        @assert A_adj_julia == expected_adj "J(Mat') failed"
        println("✓ J(Mat') returns correct adjoint type")
        println("   Type: ", typeof(A_adj_julia))
    end

    # Test 5: J(Mat) sparse -> SparseMatrixCSC
    println(io0(), "\n[Test 5] J(Mat) sparse -> SparseMatrixCSC")
    B_data = [2.0 1.0 0.0 0.0;
              1.0 2.0 1.0 0.0;
              0.0 1.0 2.0 1.0;
              0.0 0.0 1.0 2.0]
    B = Mat_uniform(B_data)
    B_julia = J(B)
    if rank == 0
        @assert B_julia == sparse(B_data) "J(Mat sparse) failed"
        @assert B_julia isa SparseMatrixCSC{Float64, Int} "J(Mat sparse) should return SparseMatrixCSC"
        println("✓ J(Mat sparse) = sparse(Mat) passed")
        println("   Type: ", typeof(B_julia))
        println("   nnz: ", nnz(B_julia))
    end

    # Test 6: J(Mat') sparse -> adjoint SparseMatrixCSC
    println(io0(), "\n[Test 6] J(Mat') sparse -> Adjoint SparseMatrixCSC")
    B_adj = B'
    B_adj_julia = J(B_adj)
    if rank == 0
        @assert B_adj_julia == sparse(B_data)' "J(Mat' sparse) failed"
        @assert B_adj_julia isa Adjoint{Float64, SparseMatrixCSC{Float64, Int}} "J(Mat' sparse) should return Adjoint SparseMatrixCSC"
        println("✓ J(Mat' sparse) = sparse(Mat)' passed")
        println("   Type: ", typeof(B_adj_julia))
        println("   nnz: ", nnz(parent(B_adj_julia)))
    end

    # Test 7: Verify J preserves values correctly with non-square matrix
    println(io0(), "\n[Test 7] J with non-square matrix")
    C_data = [1.0 2.0 3.0; 4.0 5.0 6.0]
    C = Mat_uniform(C_data)
    C_julia = J(C)
    C_adj_julia = J(C')
    if rank == 0
        @assert C_julia == C_data "J(C) failed"
        @assert C_adj_julia == C_data' "J(C') failed"
        @assert size(C_julia) == (2, 3) "J(C) has wrong size"
        @assert size(C_adj_julia) == (3, 2) "J(C') has wrong size"
        println("✓ J preserves values and dimensions correctly")
    end

    # Clean up
    SafePETSc.SafeMPI.check_and_destroy!()

    println(io0(), "\n" * "=" ^ 60)
    println(io0(), "All J() function tests passed!")
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
end
