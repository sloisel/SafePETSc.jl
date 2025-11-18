# MPI Programming

This chapter covers important considerations when writing MPI programs using SafePETSc, particularly regarding error handling and collective operations.

## The Challenge of Exceptions in MPI

MPI programming is fundamentally incompatible with traditional exception handling. When you write parallel code that runs across multiple MPI ranks, standard assertions or exceptions can cause serious problems:

**The Problem**: If one rank throws an exception while others continue execution, the MPI cluster becomes desynchronized and will hang. For example:

```julia
# DANGEROUS: Don't do this in MPI code!
using SafePETSc
SafePETSc.Init()

x = Vec_uniform([1.0, 2.0, NaN, 4.0])  # NaN only on some ranks

# This will hang! Some ranks will pass, others will fail
@assert all(isfinite.(Vector(x)))  # ❌ Causes hang if ranks disagree
```

In the example above, if some ranks have finite values but others have `NaN`, some ranks will assert while others continue. The MPI cluster becomes desynchronized and will deadlock.

## Safe Exception Handling

To handle errors safely in MPI programs, exceptions must be **collective operations** that either fail on all ranks simultaneously or pass on all ranks. SafePETSc provides several tools for this.

### Using `@mpiassert` for Collective Assertions

The `SafeMPI.@mpiassert` macro provides a collective assertion mechanism:

```julia
using SafePETSc
using SafePETSc.SafeMPI
SafePETSc.Init()

x = Vec_uniform([1.0, 2.0, 3.0, 4.0])

# Safe: All ranks check together
SafeMPI.@mpiassert all(isfinite.(Vector(x))) "Vector contains non-finite values"
```

How `@mpiassert` works:
1. Each rank evaluates the condition locally
2. All ranks communicate to determine if ANY rank failed
3. If any rank's condition is false, ALL ranks throw an error simultaneously
4. If all ranks' conditions are true, ALL ranks continue

**Important**: `@mpiassert` is a **collective operation** and therefore slower than regular assertions. Use it only when necessary for correctness in MPI contexts.

### Using `mpi_any` for Conditional Logic

When you need to make decisions based on conditions that might differ across ranks, use `mpi_any`:

```julia
using SafePETSc
using SafePETSc.SafeMPI
SafePETSc.Init()

# Each rank computes some local condition
local_has_error = some_local_check()

# Collective operation: true if ANY rank has an error
any_rank_has_error = mpi_any(local_has_error)

if any_rank_has_error
    # All ranks execute this branch together
    println(io0(), "Error detected on at least one rank")
    # Handle error collectively
else
    # All ranks execute this branch together
    println(io0(), "All ranks are healthy")
end
```

This ensures all ranks make the same decision, preventing desynchronization.

See the [SafeMPI API Reference](@ref SafePETSc.SafeMPI.mpi_any) for more details.

### Using `mpi_uniform` to Verify Consistency

The `mpi_uniform` function checks whether a value is identical across all ranks:

```julia
using SafePETSc
using SafePETSc.SafeMPI
SafePETSc.Init()

# Create a matrix that should be the same on all ranks
A = [1.0 2.0; 3.0 4.0]

# Verify it's actually uniform across all ranks
SafeMPI.@mpiassert mpi_uniform(A) "Matrix A is not uniform across ranks"

# Safe to use A as a uniform matrix
A_petsc = Mat_uniform(A)
```

This is particularly useful for debugging distributed algorithms where you expect certain values to be synchronized.

See the [SafeMPI API Reference](@ref SafePETSc.SafeMPI.mpi_uniform) for more details.

## Best Practices

1. **Never use standard `@assert` or `throw` in MPI code** unless you are certain all ranks will agree on the outcome

2. **Use `@mpiassert` for correctness checks** that involve distributed data:
   ```julia
   SafeMPI.@mpiassert size(A) == size(B) "Matrix dimensions must match"
   ```

3. **Use `mpi_any` for error detection** when local conditions might differ:
   ```julia
   if mpi_any(local_error_condition)
       # Handle error collectively on all ranks
   end
   ```

4. **Use `mpi_uniform` to verify assumptions** about distributed data:
   ```julia
   SafeMPI.@mpiassert mpi_uniform(config) "Configuration must be uniform"
   ```

5. **Remember that collective operations are slow** - use them judiciously. They require communication between all ranks, so they can impact performance if used excessively.

## Example: Safe Error Handling Pattern

Here's a complete example showing safe error handling in an MPI context:

```julia
using SafePETSc
using SafePETSc.SafeMPI
SafePETSc.Init()

function safe_computation(x::Vec)
    # Convert to local array for checking
    x_local = Vector(x)

    # Check for problems locally
    local_has_nan = any(isnan, x_local)
    local_has_inf = any(isinf, x_local)

    # Collective check: any rank has problems?
    if mpi_any(local_has_nan)
        error("NaN detected in vector on at least one rank")
    end

    if mpi_any(local_has_inf)
        error("Inf detected in vector on at least one rank")
    end

    # All ranks confirmed data is good, proceed with computation
    result = x .* 2.0
    return result
end

# Usage
x = Vec_uniform([1.0, 2.0, 3.0, 4.0])
y = safe_computation(x)  # Safe: all ranks execute together
```

## Common Pitfalls to Avoid

### Pitfall 1: Rank-Dependent Assertions
```julia
# ❌ WRONG: Will hang if condition differs by rank
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @assert check_something()  # Only rank 0 might assert!
end
```

```julia
# ✓ CORRECT: Use collective operations
local_check = (MPI.Comm_rank(MPI.COMM_WORLD) == 0) ? check_something() : true
SafeMPI.@mpiassert local_check "Check failed on rank 0"
```

### Pitfall 2: File I/O Errors
```julia
# ❌ WRONG: File might exist on some ranks but not others
@assert isfile("config.txt")  # Might differ by rank!
```

```julia
# ✓ CORRECT: Use collective check
SafeMPI.@mpiassert isfile("config.txt") "config.txt not found"
```

### Pitfall 3: Floating-Point Comparisons
```julia
# ❌ WRONG: Floating-point round-off might differ by rank
@assert computed_value ≈ expected_value
```

```julia
# ✓ CORRECT: Use collective assertion
SafeMPI.@mpiassert computed_value ≈ expected_value "Value mismatch detected"
```

## Summary

MPI programming requires careful handling of exceptions and error conditions:

- Use **`@mpiassert`** for collective assertions that must pass or fail on all ranks together
- Use **`mpi_any`** to make collective decisions based on local conditions
- Use **`mpi_uniform`** to verify data consistency across ranks
- **Never use standard assertions or exceptions** that might execute differently on different ranks
- Remember that collective operations have **performance costs** - use them wisely but don't hesitate to use them for correctness

By following these patterns, you can write robust MPI programs that won't hang or deadlock due to desynchronized exception handling.
