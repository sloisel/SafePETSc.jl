using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Starting minimal test")
    flush(stdout)
end

ts = @testset MPITestHarness.QuietTestSet "Minimal addition test" begin

clear_mat_pool!()

A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
B1 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)

matA1 = Mat_sum(A1)
matB1 = Mat_sum(B1)

matC1 = matA1 + matB1
@test matC1 isa SafeMPI.DRef
@test size(matC1) == (4, 4)

C1_local = SafePETSc._mat_to_local_sparse(matC1)
C1_sum = zeros(4, 4)
MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
if rank == 0
    expected = Matrix(A1 + B1)
    @test all(isapprox.(C1_sum, expected, atol=1e-10))
end

matC1 = nothing
GC.gc()
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

stats = get_mat_pool_stats()
if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct)
    @test nonproduct_count == 1
end

end  # testset

# Aggregate results
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
]

global_counts = similar(local_counts)
MPI.Reduce!(local_counts, global_counts, +, 0, comm)

if rank == 0
    println("Test Summary: Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])")
end

MPI.Barrier(comm)

if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)
end
