# Bug Report: Test Suite Hanging Issues

## Summary

While implementing column extraction feature (`A[:,k]`), I discovered pre-existing bugs in the SafePETSc test suite that cause tests to hang after completion.

## Column Extraction Feature (COMPLETED)

The column extraction feature was successfully implemented:
- **File**: `src/mat.jl:1163-1218`
- **Function**: `Base.getindex(A::Mat{T,Prefix}, ::Colon, k::Int) -> Vec{T,Prefix}`
- **Tests**: `test/test_mat_getindex.jl` - 52 passes, 0 failures
- **Status**: Feature works correctly

## Bug #1: MPI Deadlock on Test Failures (PARTIALLY FIXED)

### Problem
When tests have errors, rank 0 calls `Base.exit(1)` while other ranks continue to `MPI.Barrier(comm)`, causing a deadlock.

### Location
Multiple test files had this pattern:
```julia
MPI.Barrier(comm)

if rank == 0 && (global_counts[2] > 0 || global_counts[3] > 0)
    Base.exit(1)  # Rank 0 exits here
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)  # Ranks 1-3 wait forever for rank 0!
```

### Fix Applied
Changed to make ALL ranks exit on errors:
```julia
if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)  # All ranks exit together
end
```

### Files Fixed (15 total)
- test_blockproduct.jl
- test_extreme_aspect_ratios.jl
- test_mat_blockdiag.jl
- test_mat_cat.jl
- test_mat_eachrow.jl
- test_mat_getindex.jl
- test_mat_ops.jl
- test_mat_spdiagm.jl
- test_mat_sum.jl
- test_mat_uniform.jl
- test_mpi_uniform.jl
- test_vec_pool_edge.jl
- test_vec_pool.jl
- test_vec_sum.jl
- test_vec_uniform.jl

### Result
BlockProduct test no longer hangs when it has errors - exits cleanly with error code 1.

## Bug #2: Tests Hang After Successful Completion (UNSOLVED)

### Problem
Tests that create PETSc objects hang **after** completing successfully, even with 0 errors.

### Evidence
```bash
Test Summary: Vec_uniform tests (aggregated across 4 ranks)
  Pass: 100  Fail: 0  Error: 0  Broken: 0  Skip: 0

# Process hangs here and gets killed by timeout
```

### Affected Tests
- test_vec_uniform.jl
- test_vec_sum.jl
- test_mat_uniform.jl
- test_mat_sum.jl
- test_mat_getindex.jl (my new test)
- test_blockproduct.jl
- Likely all tests that create PETSc Mat/Vec objects

### Not Affected
- test_mpi.jl - Completes successfully without hanging
  - This test does NOT create PETSc objects, only SafeMPI DRefs

### Observations

1. **Minimal test works**: A simple script creating a Vec and calling `check_and_destroy!()` completes without hanging

2. **Hang occurs after test completion**: All test output prints successfully, then process hangs

3. **Likely cause**: Julia's atexit hooks or finalizers trying to cleanup PETSc objects during MPI shutdown
   - Comment in tests: "Julia's MPI.jl automatically finalizes MPI at exit via atexit hook"
   - PETSc likely also has cleanup hooks
   - These may be calling MPI operations after some ranks have finalized MPI

4. **BlockProduct has 4 errors**: Test Summary shows "Error: 4" but they don't appear to cause the hang (they trigger Bug #1 instead)

### Attempted Fixes (unsuccessful)
- Added `SafeMPI.check_and_destroy!()` and `MPI.Barrier(comm)` before exit
  - Did not resolve the hang
  - Removed these changes

### Next Steps for Debugging

1. **Get stack traces from hanging processes**:
   ```bash
   # While test is hanging:
   killall -QUIT julia  # Prints stack traces to stderr
   ```

2. **Investigate atexit hooks**:
   - Check what Julia's MPI.jl atexit hook does
   - Check what PETSc.jl atexit hook does
   - See if they conflict during shutdown

3. **Test explicit MPI.Finalize()**:
   - The tests say "We don't call MPI.Finalize() here"
   - Maybe we SHOULD call it explicitly before atexit hooks run

4. **Investigate BlockProduct's 4 errors**:
   - These might be related to the hang
   - Run with verbose output to see what the errors are

5. **Compare working vs hanging tests**:
   - test_mpi.jl works (no PETSc objects)
   - test_vec_uniform.jl hangs (creates PETSc Vecs)
   - Difference: PETSc object lifecycle and finalizers

## Current State

- ✅ Column extraction feature implemented and tested
- ✅ MPI deadlock on test failures fixed
- ❌ Tests still hang after successful completion
- ❌ Full test suite cannot complete

## How to Reproduce

```bash
cd /path/to/SafePETSc.jl

# This hangs after printing "Pass: 100":
timeout 15 julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(Base.active_project()) test/test_vec_uniform.jl`)'

# This works fine:
timeout 15 julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(Base.active_project()) test/test_mpi.jl`)'
```

## Files Modified

1. `src/mat.jl` - Added `Base.getindex` for column extraction
2. `test/test_mat_getindex.jl` - New test file (created)
3. `test/runtests.jl` - Added test_mat_getindex.jl to test suite
4. 15 test files - Fixed MPI deadlock bug (*.bak files contain originals)
