using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

const _VERBOSE = get(ENV, "VERBOSE_MAT_UNIFORM", "0") == "1"

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Mat_uniform test starting")
    flush(stdout)
end

# Test body extracted for reuse
function _mat_uniform_tests_body()
    if rank == 0
        println("[DEBUG] Mat_uniform Test 1 starting")
        flush(stdout)
    end

    # Test 1: Create a uniform matrix (non-square to expose row/col bugs)
    A = ones(6, 10)  # Non-square: 6 rows, 10 columns
    dr = SafePETSc.Mat_uniform(A)

    @test dr isa SafeMPI.DRef
    obj = dr.obj
    @test obj.A isa PETSc.Mat
    @test length(obj.row_partition) == nranks + 1
    @test length(obj.col_partition) == nranks + 1

    # Objects are garbage collected automatically via finalizers
    # Manually trigger check_and_destroy to ensure collective cleanup
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 2: Check default row and column partitions (non-square)
    row_partition = SafePETSc.default_row_partition(6, nranks)
    col_partition = SafePETSc.default_row_partition(10, nranks)

    @test length(row_partition) == nranks + 1
    @test row_partition[1] == 1
    @test row_partition[end] == 7  # 6 rows + 1

    @test length(col_partition) == nranks + 1
    @test col_partition[1] == 1
    @test col_partition[end] == 11  # 10 cols + 1

    # Check coverage: all rows should be covered exactly once
    all_rows = Set()
    for i in 0:(nranks-1)
        start = row_partition[i+1]
        stop = row_partition[i+2] - 1
        @test start <= stop
        for row in start:stop
            push!(all_rows, row)
        end
    end
    @test all_rows == Set(1:6)

    # Test 3: Custom row and column partitions (non-square)
    A = ones(8, 12)  # Non-square: 8 rows, 12 columns
    custom_row_partition = [1, 3, 5, 7, 9]
    custom_col_partition = [1, 4, 7, 10, 13]  # Different for columns

    if nranks == 4
        dr = SafePETSc.Mat_uniform(A; row_partition=custom_row_partition, col_partition=custom_col_partition)
        @test dr.obj.row_partition == custom_row_partition
        @test dr.obj.col_partition == custom_col_partition
        SafeMPI.check_and_destroy!()
        MPI.Barrier(comm)
    end

    # Test 4: Verify mpi_uniform assertion works (non-square)
    A_uniform = ones(5, 7)  # Non-square: 5 rows, 7 columns
    # This should not error
    dr = SafePETSc.Mat_uniform(A_uniform)
    @test dr.obj.A isa PETSc.Mat
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 5: Non-square matrix
    A_rect = ones(12, 8)
    row_partition_rect = SafePETSc.default_row_partition(12, nranks)
    col_partition_rect = SafePETSc.default_row_partition(8, nranks)

    dr_rect = SafePETSc.Mat_uniform(A_rect; row_partition=row_partition_rect, col_partition=col_partition_rect)
    @test size(dr_rect.obj) == (12, 8)
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 6: Identity-like matrix (non-square with eye-like pattern)
    # Create a 5×8 matrix with 1s on the "diagonal"
    A_eye = zeros(5, 8)
    for i in 1:min(5, 8)
        A_eye[i, i] = 1.0
    end
    dr_eye = SafePETSc.Mat_uniform(A_eye)
    @test dr_eye.obj.A isa PETSc.Mat
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 7: Matrix with different values (non-square)
    A_vals = reshape(Float64.(1:54), 6, 9)  # Non-square: 6×9
    dr_vals = SafePETSc.Mat_uniform(A_vals)
    @test dr_vals.obj.A isa PETSc.Mat
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Starting sparse matrix tests")
        flush(stdout)
    end

    # Test 8: Sparse identity matrix
    A_sparse_id = sparse(1.0I, 8, 8)
    dr_sparse_id = SafePETSc.Mat_uniform(A_sparse_id; Prefix=SafePETSc.MPIAIJ)
    @test dr_sparse_id isa SafeMPI.DRef
    @test dr_sparse_id.obj.A isa PETSc.Mat
    # Verify values are preserved
    A_reconstructed = Matrix(dr_sparse_id)
    @test norm(A_reconstructed - Matrix(A_sparse_id)) < 1e-14
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 9: Sparse diagonal matrix
    A_sparse_diag = spdiagm(0 => [1.0, 2.0, 3.0, 4.0, 5.0])
    dr_sparse_diag = SafePETSc.Mat_uniform(A_sparse_diag; Prefix=SafePETSc.MPIAIJ)
    @test dr_sparse_diag isa SafeMPI.DRef
    A_diag_reconstructed = Matrix(dr_sparse_diag)
    @test norm(A_diag_reconstructed - Matrix(A_sparse_diag)) < 1e-14
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 10: Sparse tridiagonal matrix
    n = 10
    A_sparse_tri = spdiagm(
        -1 => fill(-1.0, n-1),
        0 => fill(2.0, n),
        1 => fill(-1.0, n-1)
    )
    dr_sparse_tri = SafePETSc.Mat_uniform(A_sparse_tri; Prefix=SafePETSc.MPIAIJ)
    @test dr_sparse_tri isa SafeMPI.DRef
    A_tri_reconstructed = Matrix(dr_sparse_tri)
    @test norm(A_tri_reconstructed - Matrix(A_sparse_tri)) < 1e-14
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 11: Non-square sparse matrix
    A_sparse_rect = sparse([1, 2, 3, 4], [1, 3, 2, 4], [1.5, 2.5, 3.5, 4.5], 6, 8)
    dr_sparse_rect = SafePETSc.Mat_uniform(A_sparse_rect; Prefix=SafePETSc.MPIAIJ)
    @test dr_sparse_rect isa SafeMPI.DRef
    @test size(dr_sparse_rect) == (6, 8)
    A_rect_reconstructed = Matrix(dr_sparse_rect)
    @test norm(A_rect_reconstructed - Matrix(A_sparse_rect)) < 1e-14
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 12: Very sparse matrix (few non-zeros)
    A_very_sparse = spzeros(20, 20)
    A_very_sparse[5, 5] = 1.0
    A_very_sparse[10, 15] = 2.0
    A_very_sparse[15, 3] = 3.0
    dr_very_sparse = SafePETSc.Mat_uniform(A_very_sparse; Prefix=SafePETSc.MPIAIJ)
    @test dr_very_sparse isa SafeMPI.DRef
    A_very_sparse_reconstructed = Matrix(dr_very_sparse)
    @test norm(A_very_sparse_reconstructed - Matrix(A_very_sparse)) < 1e-14
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Test 13: Custom partitions with sparse matrix
    if nranks == 4
        A_sparse_custom = spdiagm(0 => collect(1.0:12.0))
        custom_row_part = [1, 4, 7, 10, 13]
        custom_col_part = [1, 4, 7, 10, 13]
        dr_sparse_custom = SafePETSc.Mat_uniform(
            A_sparse_custom;
            row_partition=custom_row_part,
            col_partition=custom_col_part,
            Prefix=SafePETSc.MPIAIJ
        )
        @test dr_sparse_custom.obj.row_partition == custom_row_part
        @test dr_sparse_custom.obj.col_partition == custom_col_part
        A_custom_reconstructed = Matrix(dr_sparse_custom)
        @test norm(A_custom_reconstructed - Matrix(A_sparse_custom)) < 1e-14
        SafeMPI.check_and_destroy!()
        MPI.Barrier(comm)
    end

    if rank == 0
        println("[DEBUG] Sparse matrix tests completed")
        flush(stdout)
    end
end

# Keep output tidy and aggregate at the end
if _VERBOSE
    @testset "Mat_uniform tests" begin
        _mat_uniform_tests_body()
    end
else
    ts = @testset MPITestHarness.QuietTestSet "Mat_uniform tests" begin
        _mat_uniform_tests_body()
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
    println("Test Summary: Mat_uniform tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Mat_uniform test file completed successfully")
    flush(stdout)
end
end # !_VERBOSE



# Note: We don't call MPI.Finalize() here because Julia's MPI.jl
# automatically finalizes MPI at exit via atexit hook
