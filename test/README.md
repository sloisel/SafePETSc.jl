# Testing SafePETSc

SafePETSc requires MPI for testing. Tests are automatically run with multiple MPI ranks.

## Running Tests

### Using Pkg.test() (recommended)

```julia
using Pkg
Pkg.test("SafePETSc")
```

Or from the command line:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This automatically spawns 4 MPI processes to run the tests.

### Running MPI tests directly

You can also run the MPI test file directly:

```bash
mpiexec -n 4 julia --project=. test/test_mpi.jl
```

## Test Coverage

The test suite covers automatic garbage collection-based destruction:

- **Single object GC destruction**: Tests that a single SafeMPI reference is automatically destroyed when it goes out of scope and is garbage collected. Finalizers automatically call `release!()` and `check_and_destroy!()` cleans up the underlying object across all ranks.

- **Multiple object GC destruction**: Tests that multiple SafeMPI references created and released simultaneously are all properly destroyed via automatic garbage collection.

## Requirements

- MPI must be installed and configured
- The package must be set up to use system MPI (see main README for configuration)
- At least 2 MPI ranks are recommended for meaningful tests (4 is used by default)

## Design Notes

- Tests use `if !MPI.Initialized() MPI.Init() end` to safely initialize MPI only when needed
- Tests do not call `MPI.Finalize()` - Julia's MPI.jl automatically handles finalization via atexit hooks
- This allows multiple test files to be added in the future and share the same MPI session
- The `release!()` function has guards to safely handle calls after MPI finalization
