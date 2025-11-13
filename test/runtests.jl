using Test
using MPI

# Precompile all dependencies once before any mpiexec to avoid concurrent compilation
# This significantly reduces hangs due to pidfile contention and MPI initialization conflicts
try
    # Force precompilation of all test dependencies
    @eval using SafePETSc
    @eval using PETSc
    @eval using LinearAlgebra
    @eval using SparseArrays
    println("Precompilation complete for test environment")
    flush(stdout)
catch err
    @warn "Precompile step hit an error; tests may still proceed" err
end

# Helper to run a test file under mpiexec with a fixed project and check exit status
function run_mpi_test(test_file::AbstractString; nprocs::Integer=4, expect_success::Bool=true)
    mpiexec_cmd = MPI.mpiexec()
    # Use the active test environment project (which has already been precompiled above)
    # This ensures LocalPreferences.toml is honored and avoids recompilation under mpiexec
    test_proj = Base.active_project()
    cmd = `$mpiexec_cmd -n $nprocs $(Base.julia_cmd()) --project=$test_proj $test_file`
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file=test_file ok=ok expect_success=expect_success exitcode=proc.exitcode cmd=cmd active_proj=test_proj
    end
    @test ok == expect_success
end

@testset "SafePETSc Tests" begin
    @testset "MPI Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mpi.jl"); nprocs=4, expect_success=true)
    end
    
    @testset "MPI Assert Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mpiassert_ok.jl"); nprocs=4, expect_success=true)
    end

    @testset "MPI Assert Failure Behavior" begin
        run_mpi_test(joinpath(@__DIR__, "test_mpiassert_fail.jl"); nprocs=4, expect_success=false)
    end

    @testset "mpi_uniform Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mpi_uniform.jl"); nprocs=4, expect_success=true)
    end

    @testset "Vec_uniform Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_vec_uniform.jl"); nprocs=4, expect_success=true)
    end

    @testset "Vec_sum Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_vec_sum.jl"); nprocs=4, expect_success=true)
    end

    @testset "Mat_uniform Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_uniform.jl"); nprocs=4, expect_success=true)
    end

    @testset "Mat_sum Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_sum.jl"); nprocs=4, expect_success=true)
    end
    
    @testset "Mat cat Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_cat.jl"); nprocs=4, expect_success=true)
    end
    
    @testset "Mat blockdiag Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_blockdiag.jl"); nprocs=4, expect_success=true)
    end
    
    @testset "Mat spdiagm Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_spdiagm.jl"); nprocs=4, expect_success=true)
    end
    
    @testset "Matrix Operations Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_ops.jl"); nprocs=4, expect_success=true)
    end

    @testset "Mat eachrow (dense)" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_eachrow.jl"); nprocs=4, expect_success=true)
    end

    @testset "Vec Pooling Edge Cases Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_vec_pool_edge.jl"); nprocs=4, expect_success=true)
    end

    @testset "Matrix Add/Sub Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_mat_addsub_pool.jl"); nprocs=4, expect_success=true)
    end

    @testset "BlockProduct Tests" begin
        run_mpi_test(joinpath(@__DIR__, "test_blockproduct.jl"); nprocs=4, expect_success=true)
    end
end
