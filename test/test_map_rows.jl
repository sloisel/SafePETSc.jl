using Test
using MPI
using SafePETSc
using SafePETSc: DENSE, MPIAIJ
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Ensure matrices with prefix "dense_" are mpidense
SafePETSc.petsc_options_insert_string("-dense_mat_type mpidense")

# Reference implementation for native Julia types
map_rows_native(f, A...) = vcat((f.((eachrow.(A))...))...)

ts = @testset MPITestHarness.QuietTestSet "map_rows" begin
    # Test data from user's examples
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

    # Test 1: scalar output (sum of row elements)
    begin
        # Test with matrix returning scalar per row
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)

        # map_rows with scalar output
        result = map_rows(sum, B)

        # Compare with native implementation
        expected = map_rows_native(sum, B_data)

        # Collect result on all ranks
        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 2: vector output → Vec (sum and product per row)
    begin
        # Test from user's first example: map_rows((x)->[sum(x),prod(x)],B)
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)

        result = map_rows(x -> [sum(x), prod(x)], B)
        expected = map_rows_native(x -> [sum(x), prod(x)], B_data)

        # Should return a Vec with 10 elements (5 rows * 2 outputs)
        @test size(result)[1] == 10

        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 3: adjoint vector output → Mat (sum and product per row as matrix)
    begin
        # Test from user's second example: map_rows((x)->[sum(x),prod(x)]',B)
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)

        result = map_rows(x -> [sum(x), prod(x)]', B; Prefix=DENSE)
        expected = map_rows_native(x -> [sum(x), prod(x)]', B_data)

        # Should return a Mat with 5 rows, 2 cols
        @test size(result) == (5, 2)

        # Convert to Julia matrix for comparison
        result_mat = Matrix(result)
        @test result_mat ≈ expected rtol=1e-10
    end

    # Test 4: multiple inputs (matrix + vector)
    begin
        # Test from user's third example: map_rows((x,y)->[sum(x),prod(x),y[1]]',B,C)
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)
        C = SafePETSc.Vec_uniform(C_data)

        result = map_rows((x, y) -> [sum(x), prod(x), y[1]]', B, C; Prefix=DENSE)

        # For the native version, we need to reshape C_data for eachrow
        C_reshaped = reshape(C_data, :, 1)
        expected = map_rows_native((x, y) -> [sum(x), prod(x), y[1]]', B_data, C_reshaped)

        # Should return a Mat with 5 rows, 3 cols
        @test size(result) == (5, 3)

        result_mat = Matrix(result)
        @test result_mat ≈ expected rtol=1e-10
    end

    # Test 5: simple element-wise operation on Vec
    begin
        # Test map_rows on a vector (each row is a 1-element vector)
        C = SafePETSc.Vec_uniform(C_data)

        # Square each element (x is a 1-element vector, so use x[1])
        result = map_rows(x -> x[1]^2, C)
        expected = C_data .^ 2

        @test size(result)[1] == 5

        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 6: Vec to Vec with expansion
    begin
        # Apply function that returns 2-element vector from each 1-element input
        C = SafePETSc.Vec_uniform(C_data)

        result = map_rows(x -> [x[1], -x[1]], C)

        # Should expand from 5 to 10 elements
        @test size(result)[1] == 10

        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        expected = vcat([[x, -x] for x in C_data]...)

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 7: Vec to Mat
    begin
        # Convert vector to matrix by outputting row vectors
        C = SafePETSc.Vec_uniform(C_data)

        result = map_rows(x -> [x[1], x[1]^2, x[1]^3]', C; Prefix=DENSE)

        @test size(result) == (5, 3)

        result_mat = Matrix(result)
        expected = hcat(C_data, C_data.^2, C_data.^3)

        @test result_mat ≈ expected rtol=1e-10
    end

    # Test 8: multiple Vecs
    begin
        # Combine two vectors
        C = SafePETSc.Vec_uniform(C_data)
        D_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Use deterministic data
        D = SafePETSc.Vec_uniform(D_data)

        result = map_rows((x, y) -> x[1] + y[1], C, D)
        expected = C_data .+ D_data

        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 9: multiple Mats
    begin
        # Element-wise operation on two matrices
        A_data = reshape(Float64.(1:15), 5, 3)  # Use deterministic data
        A = SafePETSc.Mat_uniform(A_data; Prefix=DENSE)
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)

        # Compute dot product of corresponding rows
        result = map_rows((x, y) -> sum(x .* y), A, B)

        expected = [sum(A_data[i,:] .* B_data[i,:]) for i in 1:5]

        result_local = PETSc.unsafe_localarray(result.obj.v; read=true)
        row_lo = result.obj.row_partition[rank+1]
        row_hi = result.obj.row_partition[rank+2] - 1

        try
            @test result_local ≈ expected[row_lo:row_hi] rtol=1e-10
        finally
            Base.finalize(result_local)
        end
    end

    # Test 10: prefix propagation
    begin
        # Test that prefix is properly set
        B = SafePETSc.Mat_uniform(B_data; Prefix=DENSE)
        result = map_rows(sum, B)

        # Result should be a valid Vec
        @test result isa SafePETSc.Vec{Float64,DENSE}
    end
end

MPI.Barrier(comm)

# Aggregate counts and print summary on root
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
    println("Test Summary: map_rows (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end
