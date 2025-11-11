using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

const _VERBOSE = get(ENV, "VERBOSE_MAT_PRODUCT", "0") == "1"

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Matrix product tracking test starting")
    flush(stdout)
end

# Test body extracted for reuse
function _mat_product_tests_body()
    if rank == 0
        println("[DEBUG] Matrix product Test 1: Non-product matrices have UNSPECIFIED type")
        flush(stdout)
    end

    # Test 1: Non-product matrices should have UNSPECIFIED type and empty args
    A = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], 4, 4)
    mat = SafePETSc.Mat_sum(A)

    @test mat.obj.product_type == SafePETSc.MATPRODUCT_UNSPECIFIED
    @test isempty(mat.obj.product_args)

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 2: A * B tracks MATPRODUCT_AB")
        flush(stdout)
    end

    # Test 2: A * B should track product type and fingerprints
    A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
    B1 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 4, 4)

    matA = SafePETSc.Mat_sum(A1)
    matB = SafePETSc.Mat_sum(B1)
    matC = matA * matB

    @test matC.obj.product_type == SafePETSc.MATPRODUCT_AB
    @test length(matC.obj.product_args) == 2
    @test matC.obj.product_args[1] == matA.obj.fingerprint
    @test matC.obj.product_args[2] == matB.obj.fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 3: A' * B tracks MATPRODUCT_AtB")
        flush(stdout)
    end

    # Test 3: A' * B should track transpose-matrix product
    A2 = sparse([1, 2, 3], [1, 1, 1], [1.0, 2.0, 3.0], 4, 3)  # 4x3 matrix
    B2 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 4, 4)  # 4x4 matrix

    matA2 = SafePETSc.Mat_sum(A2)
    matB2 = SafePETSc.Mat_sum(B2)
    matC2 = matA2' * matB2  # (4x3)' * (4x4) = 3x4

    @test matC2.obj.product_type == SafePETSc.MATPRODUCT_AtB
    @test length(matC2.obj.product_args) == 2
    @test matC2.obj.product_args[1] == matA2.obj.fingerprint
    @test matC2.obj.product_args[2] == matB2.obj.fingerprint
    @test size(matC2) == (3, 4)

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 4: A * B' tracks MATPRODUCT_ABt")
        flush(stdout)
    end

    # Test 4: A * B' should track matrix-transpose product
    A3 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)  # 4x4 matrix
    B3 = sparse([1, 2, 3], [1, 2, 3], [4.0, 5.0, 6.0], 3, 4)  # 3x4 matrix

    matA3 = SafePETSc.Mat_sum(A3)
    matB3 = SafePETSc.Mat_sum(B3)
    matC3 = matA3 * matB3'  # (4x4) * (3x4)' = 4x3

    @test matC3.obj.product_type == SafePETSc.MATPRODUCT_ABt
    @test length(matC3.obj.product_args) == 2
    @test matC3.obj.product_args[1] == matA3.obj.fingerprint
    @test matC3.obj.product_args[2] == matB3.obj.fingerprint
    @test size(matC3) == (4, 3)

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 5: A' * B' tracks product args")
        flush(stdout)
    end

    # Test 5: A' * B' should track both fingerprints (even though UNSPECIFIED)
    A4 = sparse([1, 2], [1, 1], [1.0, 2.0], 3, 2)  # 3x2 matrix
    B4 = sparse([1, 2], [1, 2], [4.0, 5.0], 3, 3)  # 3x3 matrix

    matA4 = SafePETSc.Mat_sum(A4)
    matB4 = SafePETSc.Mat_sum(B4)
    matC4 = matA4' * matB4'  # (3x2)' * (3x3)' = 2x3

    @test matC4.obj.product_type == SafePETSc.MATPRODUCT_UNSPECIFIED  # No direct PETSc support
    @test length(matC4.obj.product_args) == 2  # But still tracks fingerprints
    @test matC4.obj.product_args[1] == matA4.obj.fingerprint
    @test matC4.obj.product_args[2] == matB4.obj.fingerprint
    @test size(matC4) == (2, 3)

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 6: Fingerprints distinguish different structures")
        flush(stdout)
    end

    # Test 6: Verify fingerprints are different for different matrices
    A5 = sparse([1, 2], [1, 2], [1.0, 2.0], 4, 4)
    A6 = sparse([1, 3], [1, 3], [1.0, 2.0], 4, 4)  # Different structure

    matA5 = SafePETSc.Mat_sum(A5)
    matA6 = SafePETSc.Mat_sum(A6)

    @test matA5.obj.fingerprint != matA6.obj.fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    if rank == 0
        println("[DEBUG] Matrix product Test 7: Product of products tracks correctly")
        flush(stdout)
    end

    # Test 7: Nested products - (A*B)*C
    A7 = sparse([1, 2], [1, 2], [1.0, 2.0], 3, 3)
    B7 = sparse([1, 2], [1, 2], [3.0, 4.0], 3, 3)
    C7 = sparse([1, 2], [1, 2], [5.0, 6.0], 3, 3)

    matA7 = SafePETSc.Mat_sum(A7)
    matB7 = SafePETSc.Mat_sum(B7)
    matC7 = SafePETSc.Mat_sum(C7)

    matAB = matA7 * matB7
    matABC = matAB * matC7

    # matAB should track A and B
    @test matAB.obj.product_type == SafePETSc.MATPRODUCT_AB
    @test matAB.obj.product_args[1] == matA7.obj.fingerprint
    @test matAB.obj.product_args[2] == matB7.obj.fingerprint

    # matABC should track AB and C
    @test matABC.obj.product_type == SafePETSc.MATPRODUCT_AB
    @test matABC.obj.product_args[1] == matAB.obj.fingerprint  # Fingerprint of AB
    @test matABC.obj.product_args[2] == matC7.obj.fingerprint

    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

# Keep output tidy and aggregate at the end
if _VERBOSE
    @testset "Matrix product tracking tests" begin
        _mat_product_tests_body()
    end
else
    ts = @testset MPITestHarness.QuietTestSet "Matrix product tracking tests" begin
        _mat_product_tests_body()
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
    println("Test Summary: Matrix product tracking tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Matrix product tracking test file completed successfully")
    flush(stdout)
end
end # !_VERBOSE

# Note: We don't call MPI.Finalize() here because Julia's MPI.jl
# automatically finalizes MPI at exit via atexit hook
