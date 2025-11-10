# Repository Guidelines

## Project Structure & Module Organization
`src/` hosts the package entry point `SafePETSc.jl`, which exports the SafeMPI-managed `Vec`, `Mat`, and solver constructors while including the backing modules (`SafeMPI.jl`, `vec.jl`, `mat.jl`, `ksp.jl`). Keep PETSc/MPI-facing helpers close to their resource type to simplify destruction hooks. `test/` mirrors the module layout; `test/runtests.jl` orchestrates MPI subtests under `MPI.mpiexec()`, and standalone scripts such as `test_mat_sum.jl` let you iterate on a single feature. Scratch investigations (e.g., `dev/petsc-bug-petscjl.jl`) stay in `dev/`. `Project.toml`/`Manifest.toml` pin Julia dependencies, while `LocalPreferences.toml` captures developer-specific MPI paths—avoid editing it unless you need to point at a new system MPI.

## Build, Test, and Development Commands
- Install deps: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
- Run all tests: `julia --project=. -e 'using Pkg; Pkg.test()'` (the harness precompiles first and spawns MPI ranks internally).
- Run a single test file from the shell: `mpiexec -n 4 julia --project=. test/test_mpi.jl` (run from repo root so `LocalPreferences.toml` is honored).
- Run a single test from a REPL (portable launcher):
  `julia --project=. -e 'using MPI; run(`\$(MPI.mpiexec()) -n 4 \$(Base.julia_cmd()) --project=. test/test_mpi.jl`)'`.
- Optional: warm precompile before multi-rank runs to avoid pidfile contention: `julia --project=. -e 'using SafePETSc, PETSc'`.
- Dev REPL: `julia --project=. -i` then `using SafePETSc`.

## Coding Style & Naming Conventions
Use 4-space indentation, no tabs, and keep exports near the top of each file (`src/SafePETSc.jl`:24). Types stay in `CamelCase` (`DistributedRefManager`), while functions/macros use `snake_case` (`check_and_destroy!`, `@mpiassert`). Parameterize functions with explicit `where {T}` so the GC/finalizer logic knows the concrete payload (`src/SafeMPI.jl`:92). Prefer short docstrings for public entry points and add brief comments only when clarifying MPI ordering or GC-sensitive code. Finalizers should call into `_enqueue_release!` rather than MPI collectives to avoid crashes.

## Testing Guidelines
`Test` is the only framework; every MPI-facing test goes through `run_mpi_test` to ensure consistent process counts and error propagation. Name files `test_*.jl` so they can be picked up locally or from `Pkg.test()`. When adding a test, register it inside `test/runtests.jl` with the expected success flag—some cases intentionally assert failure (see `test_mpiassert_fail.jl`). Aim to keep the default at four ranks; if a scenario needs more, guard it with `nprocs` and document the rationale in the test file header.

## MPI Launcher Usage
- Prefer `MPI.mpiexec()` inside Julia to locate the correct launcher that matches the active MPIPreferences (e.g., MPICH_jll). Example: `run(`\$(MPI.mpiexec()) -n 4 …`)`.
- Command-line `mpiexec` also works if your PATH points to the same MPI used by MPIPreferences. Avoid mixing system/OpenMPI with MPICH_jll.
- There is no `@mpiexec` macro; use `MPI.mpiexec()` or install the wrapper via `MPI.install_mpiexecjl()` and invoke `mpiexecjl`.
- Run any Julia file on 4 ranks (portable one-liner): `julia --project=. -e 'using MPI; run(`\$(MPI.mpiexec()) -n 4 \$(Base.julia_cmd()) --project=. path/to/foo.jl`)'`. If you installed the wrapper: `mpiexecjl -n 4 julia --project=. path/to/foo.jl`.

## Commit & Pull Request Guidelines
Recent history (`git log -5`) shows short, imperative summaries such as `matrix product first version` and `Add Vec_sum for summing sparse vectors across MPI ranks`; follow that format and omit trailing periods. Each pull request should include: scope summary, links to issues or PETSc tickets, the exact test command (e.g., `Pkg.test()` plus any targeted `mpiexec` runs), and notes about MPI/MPIPreferences changes. Provide screenshots or logs only when debugging distributed failures so reviewers can see rank-tagged output.

## Security & Configuration Tips
MPI paths live in `LocalPreferences.toml` under `[MPIPreferences]`; never commit personal filesystem paths—have contributors run `using MPIPreferences; use_system_binary("OpenMPI")` instead. PETSc must match the ABI chosen there, so document any ABI or libmpi changes in your PR description. When experimenting with alternate communicators, keep secrets or cluster hostnames out of tracked files and use environment variables instead (e.g., `export JULIA_MPI_BINARY=system` before invoking the tests).

## Common PETSc mistakes
MAT_INITIAL_MATRIX = Cint(0). In the code, the incorrect value -1 can cause a segfault.
