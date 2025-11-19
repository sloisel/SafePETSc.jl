using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

const _VERBOSE = get(ENV, "VERBOSE_MAT_INDEXING", "0") == "1"

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Mat indexing test starting")
    flush(stdout)
end

function _mat_indexing_tests_body()
    if rank == 0
        println("[DEBUG] Mat indexing Test 1: Single element indexing")
        flush(stdout)
    end

    # Test 1: Single element indexing on dense matrix
    A_data = Float64[1 2 3; 4 5 6; 7 8 9; 10 11 12]  # 4×3 matrix
    A = SafePETSc.Mat_uniform(A_data)

    # Get local row range
    row_lo = A.obj.row_partition[rank+1]
    row_hi = A.obj.row_partition[rank+2] - 1

    # Test accessing elements in our local row range
    for i in row_lo:row_hi
        for j in 1:3
            val = A[i, j]
            @test val ≈ A_data[i, j]
        end
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Mat indexing Test 2: Row range, single column")
        flush(stdout)
    end

    # Test 2: Extract rows from a single column
    B_data = Float64[10 20 30; 40 50 60; 70 80 90; 100 110 120]  # 4×3 matrix
    B = SafePETSc.Mat_uniform(B_data)

    row_lo = B.obj.row_partition[rank+1]
    row_hi = B.obj.row_partition[rank+2] - 1

    # Test full local row range
    vals = B[row_lo:row_hi, 2]
    @test vals isa Vector{Float64}
    @test length(vals) == row_hi - row_lo + 1
    @test all(vals .≈ B_data[row_lo:row_hi, 2])

    # Test partial row range (if we have more than one row)
    if row_hi > row_lo
        vals_partial = B[row_lo:row_lo+1, 1]
        @test length(vals_partial) == 2
        @test all(vals_partial .≈ B_data[row_lo:row_lo+1, 1])
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Mat indexing Test 3: Single row, column range")
        flush(stdout)
    end

    # Test 3: Extract columns from a single row
    # Using 4×4 matrix to ensure all ranks have rows (avoid empty-owned-rows edge case)
    C_data = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]  # 4×4 matrix
    C = SafePETSc.Mat_uniform(C_data)

    row_lo = C.obj.row_partition[rank+1]
    row_hi = C.obj.row_partition[rank+2] - 1

    # Test column range for each local row (rank 3 will have empty range, which is fine)
    for i in row_lo:row_hi
        vals = C[i, 2:4]
        @test vals isa Vector{Float64}
        @test length(vals) == 3
        @test all(vals .≈ C_data[i, 2:4])
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Mat indexing Test 4: Row range, column range (dense)")
        flush(stdout)
    end

    # Test 4: Extract submatrix from matrix (sparse by default)
    D_data = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]  # 4×4 matrix
    D = SafePETSc.Mat_uniform(D_data)

    row_lo = D.obj.row_partition[rank+1]
    row_hi = D.obj.row_partition[rank+2] - 1

    # Test full local row range with column range
    # Mat_uniform creates sparse (AIJ) matrices by default, so expect sparse result
    submat = D[row_lo:row_hi, 2:3]
    @test submat isa SparseMatrixCSC{Float64}
    @test size(submat) == (row_hi - row_lo + 1, 2)
    @test all(Matrix(submat) .≈ D_data[row_lo:row_hi, 2:3])

    # Test partial row range with column range (if we have more than one row)
    if row_hi > row_lo
        submat_partial = D[row_lo:row_lo+1, 1:2]
        @test submat_partial isa SparseMatrixCSC{Float64}
        @test size(submat_partial) == (2, 2)
        @test all(Matrix(submat_partial) .≈ D_data[row_lo:row_lo+1, 1:2])
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Mat indexing Test 5: Row range, column range (sparse)")
        flush(stdout)
    end

    # Test 5: Extract submatrix from diagonal sparse matrix
    E_sparse = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)
    E = SafePETSc.Mat_uniform(Matrix(E_sparse))

    row_lo = E.obj.row_partition[rank+1]
    row_hi = E.obj.row_partition[rank+2] - 1

    # Test that sparse matrix returns sparse submatrix (only for non-empty ranges)
    if row_hi >= row_lo
        submat_sparse = E[row_lo:row_hi, 1:4]
        @test submat_sparse isa SparseMatrixCSC{Float64}
        @test size(submat_sparse) == (row_hi - row_lo + 1, 4)
        # Verify values - the diagonal matrix only has values on the diagonal
        E_full = Matrix(E_sparse)
        @test all(Matrix(submat_sparse) .≈ E_full[row_lo:row_hi, 1:4])
    end

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Mat indexing tests completed")
        flush(stdout)
    end
end

# Keep output tidy and aggregate at the end
if _VERBOSE
    @testset "Mat indexing tests" begin
        _mat_indexing_tests_body()
    end
else
    ts = @testset MPITestHarness.QuietTestSet "Mat indexing tests" begin
        _mat_indexing_tests_body()
    end
end

if !_VERBOSE
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
    println("Test Summary: Mat indexing tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Mat indexing test file completed successfully")
    flush(stdout)
end
end # !_VERBOSE

# Finalize SafeMPI to prevent shutdown race conditions
SafePETSc.finalize()
