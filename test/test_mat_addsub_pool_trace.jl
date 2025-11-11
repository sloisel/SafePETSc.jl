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

println("[rank $rank] Test starting")
flush(stdout)

clear_mat_pool!()
println("[rank $rank] Cleared pool")
flush(stdout)

ts = @testset MPITestHarness.QuietTestSet "Trace test" begin

println("[rank $rank] Inside testset")
flush(stdout)

A1 = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 4, 4)
println("[rank $rank] Created A1")
flush(stdout)

B1 = sparse([1, 2, 4], [1, 2, 4], [4.0, 5.0, 6.0], 4, 4)
println("[rank $rank] Created B1")
flush(stdout)

matA1 = Mat_sum(A1)
println("[rank $rank] Created matA1")
flush(stdout)

matB1 = Mat_sum(B1)
println("[rank $rank] Created matB1")
flush(stdout)

matC1 = matA1 + matB1
println("[rank $rank] Added matrices")
flush(stdout)

@test matC1 isa SafeMPI.DRef
println("[rank $rank] Test 1")
flush(stdout)

@test size(matC1) == (4, 4)
println("[rank $rank] Test 2")
flush(stdout)

C1_local = SafePETSc._mat_to_local_sparse(matC1)
println("[rank $rank] Got local sparse")
flush(stdout)

C1_sum = zeros(4, 4)
println("[rank $rank] Created zeros")
flush(stdout)

MPI.Reduce!(Matrix(C1_local), C1_sum, +, 0, comm)
println("[rank $rank] Reduced")
flush(stdout)

if rank == 0
    expected = Matrix(A1 + B1)
    @test all(isapprox.(C1_sum, expected, atol=1e-10))
    println("[rank $rank] Verified")
    flush(stdout)
end

matC1 = nothing
println("[rank $rank] Set matC1 to nothing")
flush(stdout)

GC.gc()
println("[rank $rank] GC done")
flush(stdout)

SafeMPI.check_and_destroy!()
println("[rank $rank] check_and_destroy done")
flush(stdout)

MPI.Barrier(comm)
println("[rank $rank] Barrier 1 done")
flush(stdout)

stats = get_mat_pool_stats()
println("[rank $rank] Got stats")
flush(stdout)

if rank == 0
    nonproduct_count = sum(v for (k, v) in stats if length(k) >= 5 && k[end] == :nonproduct)
    @test nonproduct_count == 1
    println("[rank $rank] Verified count")
    flush(stdout)
end

matC2 = matA1 + matB1
println("[rank $rank] Second addition done")
flush(stdout)

@test matC2 isa SafeMPI.DRef
println("[rank $rank] Test 3")
flush(stdout)

end  # testset

println("[rank $rank] Testset ended")
flush(stdout)

# Aggregate results
local_counts = [get(ts.counts, :pass, 0), get(ts.counts, :fail, 0)]
global_counts = similar(local_counts)
MPI.Reduce!(local_counts, global_counts, +, 0, comm)

if rank == 0
    println("Pass: $(global_counts[1])  Fail: $(global_counts[2])")
end
