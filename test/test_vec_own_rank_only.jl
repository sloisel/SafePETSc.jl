using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

@testset MPITestHarness.QuietTestSet "Vec own_rank_only correctness and no-deadlock" begin
    N = 16
    rowp = SafePETSc.default_row_partition(N, nranks)
    lo = rowp[rank+1]
    hi = rowp[rank+2] - 1

    # Case A: all ranks set one local entry within their own partition
    vA = sparsevec(Int[], Float64[], N)
    if lo <= hi
        vA[lo] = Float64(rank + 1)
    end
    drA = SafePETSc.Vec_sum(vA; row_partition=rowp, own_rank_only=true)
    @test drA isa SafeMPI.DRef
    viewA = PETSc.unsafe_localarray(drA.obj.v; read=true, write=false)
    try
        if lo <= hi
            local_len = length(viewA)
            @test viewA[1] ≈ Float64(rank + 1)
            if local_len >= 2
                @test all(viewA[2:local_len] .== 0.0)
            end
        end
    finally
        Base.finalize(viewA)
    end
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Case B: only one rank contributes (others have nnz=0)
    owner = 1 % nranks  # pick rank 1 if present; else 0
    vB = sparsevec(Int[], Float64[], N)
    if rank == owner && lo <= hi
        vB[lo] = 42.0
    end
    drB = SafePETSc.Vec_sum(vB; row_partition=rowp, own_rank_only=true)
    @test drB isa SafeMPI.DRef
    viewB = PETSc.unsafe_localarray(drB.obj.v; read=true, write=false)
    try
        local_len = length(viewB)
        if rank == owner && lo <= hi
            @test viewB[1] ≈ 42.0
            if local_len >= 2
                @test all(viewB[2:local_len] .== 0.0)
            end
        else
            if lo <= hi
                @test all(viewB[1:local_len] .== 0.0)
            end
        end
    finally
        Base.finalize(viewB)
    end
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)

    # Case C: multiple quick iterations cycling owner to catch any latent mismatched collectives
    for t in 1:20
        owner_t = (t - 1) % nranks
        vC = sparsevec(Int[], Float64[], N)
        if rank == owner_t && lo <= hi
            vC[lo] = Float64(t)
        end
        drC = SafePETSc.Vec_sum(vC; row_partition=rowp, own_rank_only=true)
        @test drC isa SafeMPI.DRef
        SafeMPI.check_and_destroy!()
    end
    MPI.Barrier(comm)
end
