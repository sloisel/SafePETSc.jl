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
    destroy_early::Bool
    default_check::Int
    seconds::Float64
end

function warmup_and_avg_time(f::Function; reps::Int=10)
    # Warmup once to trigger compilation
    f()
    SafeMPI.check_and_destroy!()  # drain any pending releases
    # Single timed region, invoke f() reps times, then divide
    MPI.Barrier(COMM)
    t0 = MPI.Wtime()
    for i in 1:reps
        f()
    end
    MPI.Barrier(COMM)
    t1 = MPI.Wtime()
    dt_local = t1 - t0
    dt = MPI.Allreduce(dt_local, MPI.MAX, COMM)
    # Drain after timing to keep memory stable across scenarios
    SafeMPI.check_and_destroy!()
    return dt / reps
end

function bench_create_vec(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "V"
    reps = dcheck >= 10 ? 100 : 10
    # Warmup + measure
    dt = warmup_and_avg_time(() -> begin
        v = make_vec(prefix)
        # keep v in scope; destruction happens after barrier
        nothing
    end; reps=reps)
    return BenchResult("create_vec", "vec", assert_on, destroy_early, dcheck, dt)
end

function bench_create_mat_dense(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        A = make_dense_mat(prefix)
        nothing
    end; reps=reps)
    return BenchResult("create_mat", "dense", assert_on, destroy_early, dcheck, dt)
end

function bench_create_mat_sparse(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        A = make_sparse_diag_mat(prefix)
        nothing
    end; reps=reps)
    return BenchResult("create_mat", "sparse_diag", assert_on, destroy_early, dcheck, dt)
end

function bench_matvec_dense(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    A = make_dense_mat(prefix)
    x = make_vec(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        y = A * x
        nothing
    end; reps=reps)
    return BenchResult("matvec", "dense", assert_on, destroy_early, dcheck, dt)
end

function bench_matvec_sparse(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    A = make_sparse_diag_mat(prefix)
    x = make_vec(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        y = A * x
        nothing
    end; reps=reps)
    return BenchResult("matvec", "sparse_diag", assert_on, destroy_early, dcheck, dt)
end

function bench_solve_dense(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    # Use SPD matrix (identity) for robust solve
    rowp = make_rowpart(N)
    colp = rowp
    Adata = Matrix{Float64}(I, N, N)
    A = SafePETSc.Mat_uniform(Adata; row_partition=rowp, col_partition=colp, prefix=prefix)
    b = make_vec(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        x = A \ b
        nothing
    end; reps=reps)
    return BenchResult("solve", "dense_spd", assert_on, destroy_early, dcheck, dt)
end

function bench_solve_sparse(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    # Use sparse SPD (diagonal ones)
    A = make_sparse_diag_mat(prefix)
    b = make_vec(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        x = A \ b
        nothing
    end; reps=reps)
    return BenchResult("solve", "sparse_diag", assert_on, destroy_early, dcheck, dt)
end

function bench_matmat_dense(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "D"
    A = make_dense_mat(prefix)
    B = make_dense_mat(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        C = A * B
        nothing
    end; reps=reps)
    return BenchResult("matmat", "dense", assert_on, destroy_early, dcheck, dt)
end

function bench_matmat_sparse(assert_on::Bool, destroy_early::Bool, dcheck::Int)
    SafeMPI.set_assert(assert_on)
    SafePETSc.SafeMPI.default_manager[].destroy_early = destroy_early
    SafePETSc.default_check[] = dcheck
    prefix = "S"
    A = make_sparse_diag_mat(prefix)
    B = make_sparse_diag_mat(prefix)
    reps = dcheck >= 10 ? 100 : 10
    dt = warmup_and_avg_time(() -> begin
        C = A * B
        nothing
    end; reps=reps)
    return BenchResult("matmat", "sparse_diag", assert_on, destroy_early, dcheck, dt)
end

# -----------------------------------------------------------------------------
# Run all combinations and collect results
# -----------------------------------------------------------------------------

function run_all_benchmarks()
    # Ensure PETSc matrix type options are set up
    setup_mat_options(; dense_prefix="D", sparse_prefix="S")

    results = BenchResult[]
    checks = [1, 3, 10, 100]
    asserts = [false, true]
    destroy_earlys = [false, true]

    for a in asserts
        for d in destroy_earlys
            for c in checks
                push!(results, bench_create_vec(a, d, c))
                push!(results, bench_create_mat_dense(a, d, c))
                push!(results, bench_create_mat_sparse(a, d, c))
                push!(results, bench_matvec_dense(a, d, c))
                push!(results, bench_matvec_sparse(a, d, c))
                push!(results, bench_solve_dense(a, d, c))
                push!(results, bench_solve_sparse(a, d, c))
                push!(results, bench_matmat_dense(a, d, c))
                push!(results, bench_matmat_sparse(a, d, c))
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
    println(out, "- Repetitions: 10 when default_check < 10; 100 when default_check ≥ 10")
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
        # Build lookup maps: (assert_on, destroy_early) => (default_check => seconds)
        times = Dict{Tuple{Bool, Bool}, Dict{Int, Float64}}()
        times[(false, false)] = Dict{Int, Float64}()
        times[(false, true)]  = Dict{Int, Float64}()
        times[(true, false)]  = Dict{Int, Float64}()
        times[(true, true)]   = Dict{Int, Float64}()

        for r in bypair[(op, kind)]
            key = (r.assert_on, r.destroy_early)
            times[key][r.default_check] = r.seconds
        end

        println(out, "| default_check | assert=F, early=F | assert=F, early=T | assert=T, early=F | assert=T, early=T |")
        println(out, "|---:|---:|---:|---:|---:|")
        for c in checks
            t_ff = get(times[(false, false)], c, NaN)
            t_ft = get(times[(false, true)], c, NaN)
            t_tf = get(times[(true, false)], c, NaN)
            t_tt = get(times[(true, true)], c, NaN)
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
