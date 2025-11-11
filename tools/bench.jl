#!/usr/bin/env julia
#
# To run this benchmark, do:
#   julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. tools/bench.jl`)'
using SafePETSc
using SafePETSc.SafeMPI
using MPI
using LinearAlgebra
using SparseArrays
using Random
using Printf
using Dates

# Initialize MPI + PETSc (idempotent)
SafePETSc.Init()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

const COMM = MPI.COMM_WORLD

rank() = MPI.Comm_rank(COMM)
nranks() = MPI.Comm_size(COMM)

function time_mpi(f::Function; do_barrier::Bool=true)
    do_barrier && MPI.Barrier(COMM)
    t0 = MPI.Wtime()
    f()
    do_barrier && MPI.Barrier(COMM)
    t1 = MPI.Wtime()
    dt = t1 - t0
    # Global wall time == max across all ranks
    return MPI.Allreduce(dt, MPI.MAX, COMM)
end

function write_rank0(path::AbstractString, content::AbstractString)
    if rank() == 0
        open(path, "w") do io
            write(io, content)
        end
    end
    return nothing
end

# PETSc options for type selection via object prefix
function setup_mat_options(; dense_prefix::String="D", sparse_prefix::String="S")
    # Ensure MatSetFromOptions picks the requested types
    # Note: PETSc prefixes concatenate directly (no underscore)
    # e.g., if prefix == "D", option is "-Dmat_type <type>"
    # Dense MPIDENSE for prefix dense_prefix
    SafePETSc.petsc_options_insert_string("-$(dense_prefix)mat_type mpidense")
    # Sparse MPIAIJ for prefix sparse_prefix
    SafePETSc.petsc_options_insert_string("-$(sparse_prefix)mat_type mpiaij")
    return nothing
end

# Deterministic data builders (identical on all ranks)
const N = 100

function make_rowpart(n::Int)
    return SafePETSc.default_row_partition(n, nranks())
end

function make_vec_data(n::Int)
    # A simple deterministic Vector (1.0:n)
    return collect(1.0:n)
end

function make_dense_data(n::Int)
    # Deterministic pseudo-random but identical across ranks
    rng = MersenneTwister(1234)
    return rand(rng, Float64, n, n)
end

function make_spdiag_local(n::Int, rowp::Vector{Int})
    r = rank()
    lo = rowp[r+1]
    hi = rowp[r+2] - 1
    if hi < lo
        return spzeros(Float64, n, n)
    end
    I = collect(lo:hi)
    J = I
    V = ones(Float64, length(I))
    return sparse(I, J, V, n, n)
end

# -----------------------------------------------------------------------------
# Object constructors (distributed)
# -----------------------------------------------------------------------------

function make_vec(prefix::String)
    rowp = make_rowpart(N)
    data = make_vec_data(N)
    return SafePETSc.Vec_uniform(data; row_partition=rowp, prefix=prefix)
end

function make_dense_mat(prefix::String)
    rowp = make_rowpart(N)
    colp = make_rowpart(N)
    A = make_dense_data(N)
    return SafePETSc.Mat_uniform(A; row_partition=rowp, col_partition=colp, prefix=prefix)
end

function make_sparse_diag_mat(prefix::String)
    rowp = make_rowpart(N)
    colp = make_rowpart(N)
    Alocal = make_spdiag_local(N, rowp)
    return SafePETSc.Mat_sum(Alocal; row_partition=rowp, col_partition=colp, prefix=prefix, own_rank_only=true)
end

# -----------------------------------------------------------------------------
# Benchmark wrappers
# Each benchmark performs a warmup call, then measures average of repeated runs
# -----------------------------------------------------------------------------

struct BenchResult
    op::String
    kind::String
    assert_on::Bool
    pools_on::Bool
    default_check::Int
    seconds::Float64
end

function warmup_and_avg_time(f::Function; target_seconds::Float64=0.05, max_reps::Int=1_048_576)
    # Warmup once to trigger compilation
    f()
    SafeMPI.check_and_destroy!()  # drain any pending releases
    
    nrep = 1
    dt = 0.0
    while true
        # Timed region across all ranks
        MPI.Barrier(COMM)
        t0 = MPI.Wtime()
        for i in 1:nrep
            f()
        end
        MPI.Barrier(COMM)
        t1 = MPI.Wtime()
        dt_local = t1 - t0
        dt = MPI.Allreduce(dt_local, MPI.MAX, COMM)
        # Drain after timing to keep memory stable across attempts and scenarios
        SafeMPI.check_and_destroy!()
        
        if dt >= target_seconds || nrep >= max_reps
            break
        end
        nrep *= 2
    end
    return dt / nrep
end

function bench_create_vec(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "V"
    # Warmup + adaptive measure
    dt = warmup_and_avg_time(() -> begin
        v = make_vec(prefix)
        # keep v in scope; destruction happens after barrier
        nothing
    end)
    return BenchResult("create_vec", "vec", assert_on, pools_on, dcheck, dt)
end

function bench_create_mat_dense(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    dt = warmup_and_avg_time(() -> begin
        A = make_dense_mat(prefix)
        nothing
    end)
    return BenchResult("create_mat", "dense", assert_on, pools_on, dcheck, dt)
end

function bench_create_mat_sparse(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    dt = warmup_and_avg_time(() -> begin
        A = make_sparse_diag_mat(prefix)
        nothing
    end)
    return BenchResult("create_mat", "sparse_diag", assert_on, pools_on, dcheck, dt)
end

function bench_matvec_dense(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    A = make_dense_mat(prefix)
    x = make_vec(prefix)
    dt = warmup_and_avg_time(() -> begin
        y = A * x
        nothing
    end)
    return BenchResult("matvec", "dense", assert_on, pools_on, dcheck, dt)
end

function bench_matvec_sparse(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    A = make_sparse_diag_mat(prefix)
    x = make_vec(prefix)
    dt = warmup_and_avg_time(() -> begin
        y = A * x
        nothing
    end)
    return BenchResult("matvec", "sparse_diag", assert_on, pools_on, dcheck, dt)
end

function bench_solve_dense(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    # Use SPD matrix (identity) for robust solve
    rowp = make_rowpart(N)
    colp = rowp
    Adata = Matrix{Float64}(I, N, N)
    A = SafePETSc.Mat_uniform(Adata; row_partition=rowp, col_partition=colp, prefix=prefix)
    b = make_vec(prefix)
    dt = warmup_and_avg_time(() -> begin
        x = A \ b
        nothing
    end)
    return BenchResult("solve", "dense_spd", assert_on, pools_on, dcheck, dt)
end

function bench_solve_sparse(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    # Use sparse SPD (diagonal ones)
    A = make_sparse_diag_mat(prefix)
    b = make_vec(prefix)
    dt = warmup_and_avg_time(() -> begin
        x = A \ b
        nothing
    end)
    return BenchResult("solve", "sparse_diag", assert_on, pools_on, dcheck, dt)
end

function bench_matmat_dense(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    A = make_dense_mat(prefix)
    B = make_dense_mat(prefix)
    dt = warmup_and_avg_time(() -> begin
        C = A * B
        nothing
    end)
    return BenchResult("matmat", "dense", assert_on, pools_on, dcheck, dt)
end

function bench_matmat_sparse(assert_on::Bool, pools_on::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.ENABLE_VEC_POOL[] = pools_on
    SafePETSc.ENABLE_MAT_POOL[] = pools_on
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    A = make_sparse_diag_mat(prefix)
    B = make_sparse_diag_mat(prefix)
    dt = warmup_and_avg_time(() -> begin
        C = A * B
        nothing
    end)
    return BenchResult("matmat", "sparse_diag", assert_on, pools_on, dcheck, dt)
end

# -----------------------------------------------------------------------------
# Run all combinations and collect results
# -----------------------------------------------------------------------------

function run_all_benchmarks()
    # Ensure PETSc matrix type options are set up
    setup_mat_options(; dense_prefix="D", sparse_prefix="S")

    results = BenchResult[]
    asserts = [false, true]
    pools = [false, true]
    checks = [1, 3, 10, 100]

    for a in asserts
        for p in pools
            for c in checks
                push!(results, bench_create_vec(a, p, c))
                push!(results, bench_create_mat_dense(a, p, c))
                push!(results, bench_create_mat_sparse(a, p, c))
                push!(results, bench_matvec_dense(a, p, c))
                push!(results, bench_matvec_sparse(a, p, c))
                push!(results, bench_solve_dense(a, p, c))
                push!(results, bench_solve_sparse(a, p, c))
                push!(results, bench_matmat_dense(a, p, c))
                push!(results, bench_matmat_sparse(a, p, c))
                # Drain between scenarios to keep memory stable
                SafeMPI.check_and_destroy!()
            end
        end
    end

    return results
end

function render_markdown(results::Vector{BenchResult})
    dt = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
    out = IOBuffer()
    println(out, "# SafePETSc Benchmarks")
    println(out)
    println(out, @sprintf("- Date: %s", dt))
    println(out, @sprintf("- Ranks: %d", nranks()))
    println(out, @sprintf("- Size: N = %d", N))
    println(out, "- Repetitions: adaptively chosen to reach ≥ 0.05 s total time")
    println(out)

    # Group results by (op, kind)
    bypair = Dict{Tuple{String,String}, Vector{BenchResult}}()
    for r in results
        key = (r.op, r.kind)
        push!(get!(bypair, key, BenchResult[]), r)
    end

    # Deterministic ordering of tables
    pairs = collect(keys(bypair))
    sort!(pairs)

    checks = sort(unique(r.default_check for r in results))

    for (op, kind) in pairs
        println(out, @sprintf("## %s — %s", op, kind))
        println(out)
        # Build lookup map: (default_check, assert_on, pools_on) => seconds
        times = Dict{Tuple{Int,Bool,Bool}, Float64}()

        for r in bypair[(op, kind)]
            times[(r.default_check, r.assert_on, r.pools_on)] = r.seconds
        end

        println(out, "| default_check | assert=F, pools=F | assert=F, pools=T | assert=T, pools=F | assert=T, pools=T |")
        println(out, "|---:|---:|---:|---:|---:|")
        for c in checks
            t_ff = get(times, (c, false, false), NaN)
            t_ft = get(times, (c, false, true), NaN)
            t_tf = get(times, (c, true, false), NaN)
            t_tt = get(times, (c, true, true), NaN)
            println(out, @sprintf("| %d | %.6f | %.6f | %.6f | %.6f |", c, t_ff, t_ft, t_tf, t_tt))
        end
        println(out)
    end

    return String(take!(out))
end

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

function main()
    # Silence rank chatter so output is clean
    if rank() == 0
        println("Running benchmarks on ", nranks(), " ranks…")
    end
    results = run_all_benchmarks()
    md = render_markdown(results)
    write_rank0("bench-results.md", md)
    MPI.Barrier(COMM)
    if rank() == 0
        println("Wrote bench-results.md")
    end
    return nothing
end

main()
