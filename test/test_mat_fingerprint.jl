using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

const _VERBOSE = get(ENV, "VERBOSE_MAT_FINGERPRINT", "0") == "1"

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Matrix fingerprint test starting")
    flush(stdout)
end

# Test body extracted for reuse
function _mat_fingerprint_tests_body()
    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 1 starting: identical structures")
        flush(stdout)
    end

    # Test 1: Identical sparse structures should produce same fingerprint
    A1 = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)
    A2 = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)

    mat1 = SafePETSc.Mat_sum(A1)
    mat2 = SafePETSc.Mat_sum(A2)

    # Fingerprints should be stored in the _Mat objects
    fp1 = mat1.obj.fingerprint
    fp2 = mat2.obj.fingerprint

    @test fp1 == fp2
    @test length(fp1) == 20  # SHA-1 produces 20 bytes

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 2 starting: same structure, different values")
        flush(stdout)
    end

    # Test 2: Same structure with different values should produce same fingerprint
    A3 = sparse([1, 2, 3, 4], [1, 2, 3, 4], [10.0, 20.0, 30.0, 40.0], 4, 4)
    mat3 = SafePETSc.Mat_sum(A3)

    fp3 = mat3.obj.fingerprint

    @test fp1 == fp3  # Should match fp1 since structure is identical

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 3 starting: different sparsity pattern")
        flush(stdout)
    end

    # Test 3: Different sparsity patterns should produce different fingerprints
    A4 = sparse([1, 1, 2, 3], [1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0], 4, 4)  # Different structure
    mat4 = SafePETSc.Mat_sum(A4)

    fp4 = mat4.obj.fingerprint

    @test fp1 != fp4  # Different structure should give different fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 4 starting: different dimensions")
        flush(stdout)
    end

    # Test 4: Different dimensions should produce different fingerprints
    A5 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)  # 3x3 instead of 4x4
    mat5 = SafePETSc.Mat_sum(A5)

    fp5 = mat5.obj.fingerprint

    @test fp1 != fp5  # Different size should give different fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 5 starting: different prefix")
        flush(stdout)
    end

    # Test 5: Same structure but different prefix should produce different fingerprints
    A6 = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)
    mat6 = SafePETSc.Mat_sum(A6; prefix="test_")

    fp6 = mat6.obj.fingerprint

    @test fp1 != fp6  # Different prefix should give different fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 6 starting: tridiagonal matrix")
        flush(stdout)
    end

    # Test 6: More complex structure - tridiagonal matrix
    n = 8
    A7 = spdiagm(-1 => ones(n-1), 0 => 2*ones(n), 1 => ones(n-1))
    A8 = spdiagm(-1 => 5*ones(n-1), 0 => 10*ones(n), 1 => 5*ones(n-1))  # Different values

    mat7 = SafePETSc.Mat_sum(A7)
    mat8 = SafePETSc.Mat_sum(A8)

    fp7 = mat7.obj.fingerprint
    fp8 = mat8.obj.fingerprint

    @test fp7 == fp8  # Same tridiagonal structure, different values

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 7 starting: dense vs sparse same structure")
        flush(stdout)
    end

    # Test 7: Dense matrix representation (all nonzeros explicit)
    I9 = vec([i for i in 1:4, j in 1:4])
    J9 = vec([j for i in 1:4, j in 1:4])
    V9 = ones(16)
    A9 = sparse(I9, J9, V9, 4, 4)  # Fully dense as sparse
    mat9 = SafePETSc.Mat_sum(A9)

    fp9 = mat9.obj.fingerprint

    @test fp9 != fp1  # Dense structure is different from diagonal
    @test length(fp9) == 20

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 8 starting: larger sparse matrix")
        flush(stdout)
    end

    # Test 8: Larger sparse matrix to test scalability
    n = 100
    I = vcat(1:n, 1:n-1)  # Diagonal and superdiagonal
    J = vcat(1:n, 2:n)
    V = ones(2*n - 1)
    A10 = sparse(I, J, V, n, n)
    A11 = sparse(I, J, 2*V, n, n)  # Different values

    mat10 = SafePETSc.Mat_sum(A10)
    mat11 = SafePETSc.Mat_sum(A11)

    fp10 = mat10.obj.fingerprint
    fp11 = mat11.obj.fingerprint

    @test fp10 == fp11  # Same structure, different values

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 9 starting: rectangular matrix")
        flush(stdout)
    end

    # Test 9: Rectangular matrices
    A12 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 6)  # 4x6 matrix
    A13 = sparse([1, 2, 3], [1, 2, 3], [5.0, 6.0, 7.0], 4, 6)  # Same structure

    mat12 = SafePETSc.Mat_sum(A12)
    mat13 = SafePETSc.Mat_sum(A13)

    fp12 = mat12.obj.fingerprint
    fp13 = mat13.obj.fingerprint

    @test fp12 == fp13  # Same structure, different values

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix fingerprint Test 10 starting: empty sparse matrix")
        flush(stdout)
    end

    # Test 10: Empty sparse matrix
    A14 = spzeros(5, 5)
    A15 = spzeros(5, 5)

    mat14 = SafePETSc.Mat_sum(A14)
    mat15 = SafePETSc.Mat_sum(A15)

    fp14 = mat14.obj.fingerprint
    fp15 = mat15.obj.fingerprint

    @test fp14 == fp15  # Empty matrices should have same fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

# Keep output tidy and aggregate at the end
if _VERBOSE
    @testset "Matrix fingerprint tests" begin
        _mat_fingerprint_tests_body()
    end
else
    ts = @testset MPITestHarness.QuietTestSet "Matrix fingerprint tests" begin
        _mat_fingerprint_tests_body()
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
MPI.Reduce!(local_counts, global_counts, +, 0, comm)

if rank == 0
    println("Test Summary: Matrix fingerprint tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix fingerprint test file completed successfully")
    flush(stdout)
end
end # !_VERBOSE

# Note: We don't call MPI.Finalize() here because Julia's MPI.jl
# automatically finalizes MPI at exit via atexit hook
