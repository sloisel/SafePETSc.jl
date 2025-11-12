using MPI
using SafePETSc
SafePETSc.Init()

# This file tests that @mpiassert properly catches dimension mismatches
# Each test is designed to fail with a dimension error
# Run individual test cases to verify error messages

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Get test case from command line argument or default to 1
test_case = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1

if test_case == 1
    # Test A*B with incompatible inner dimensions
    # Should fail: 4×3 * 5×4 (inner dimensions 3 ≠ 5)
    println("[Rank $rank] Test 1: A*B with incompatible inner dimensions (4×3 * 5×4)")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(5, 4, :row_partition)
    C = A * B  # Should fail with @mpiassert

elseif test_case == 2
    # Test A'*B with mismatched row counts
    # Should fail: (4×3)' * (5×3) means 3×4 * 5×3, inner dimensions don't match
    println("[Rank $rank] Test 2: A'*B with mismatched dimensions ((4×3)' * (5×3))")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(5, 3, :row_partition)
    C = A' * B  # Should fail with @mpiassert

elseif test_case == 3
    # Test A*B' with mismatched column counts
    # Should fail: 4×3 * (4×5)' means 4×3 * 5×4, inner dimensions don't match
    println("[Rank $rank] Test 3: A*B' with mismatched dimensions (4×3 * (4×5)')")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(4, 5, :row_partition)
    C = A * B'  # Should fail with @mpiassert

elseif test_case == 4
    # Test A+B with different sizes
    # Should fail: 4×3 + 5×3 (row dimensions don't match)
    println("[Rank $rank] Test 4: A+B with different sizes (4×3 + 5×3)")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(5, 3, :row_partition)
    C = A + B  # Should fail with @mpiassert

elseif test_case == 5
    # Test A-B with different sizes
    # Should fail: 4×3 - 4×5 (column dimensions don't match)
    println("[Rank $rank] Test 5: A-B with different sizes (4×3 - 4×5)")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(4, 5, :row_partition)
    C = A - B  # Should fail with @mpiassert

elseif test_case == 6
    # Test mul!(C, A, B) with wrong output dimensions
    # Should fail: C is 3×4 but should be 4×5 for A(4×3) * B(3×5)
    println("[Rank $rank] Test 6: mul!(C, A, B) with wrong output dimensions")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(3, 5, :row_partition)
    C = SafePETSc.create_matrix(3, 4, :row_partition)  # Wrong size!
    SafePETSc.mul!(C, A, B)  # Should fail with @mpiassert

elseif test_case == 7
    # Test transpose!(B, A) with wrong B dimensions
    # Should fail: For A(4×3), B should be 3×4, not 4×4
    println("[Rank $rank] Test 7: transpose!(B, A) with wrong B dimensions")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(4, 4, :row_partition)  # Wrong size!
    SafePETSc.transpose!(B, A)  # Should fail with @mpiassert

elseif test_case == 8
    # Test A*x with wrong vector length
    # Should fail: A is 4×3, x should be length 3, not 4
    println("[Rank $rank] Test 8: A*x with wrong vector length")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    x = SafePETSc.create_vector(4, :row_partition)  # Wrong length!
    y = A * x  # Should fail with @mpiassert

elseif test_case == 9
    # Test mul!(y, A, x) with wrong output vector length
    # Should fail: A is 4×3, x is length 3, y should be length 4 not 5
    println("[Rank $rank] Test 9: mul!(y, A, x) with wrong output vector length")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    x = SafePETSc.create_vector(3, :row_partition)
    y = SafePETSc.create_vector(5, :row_partition)  # Wrong length!
    SafePETSc.mul!(y, A, x)  # Should fail with @mpiassert

elseif test_case == 10
    # Test partition mismatch: A*B where partitions don't align
    # This is trickier - need different partition types
    println("[Rank $rank] Test 10: A*B with partition mismatch")
    A = SafePETSc.create_matrix(4, 3, :row_partition)
    B = SafePETSc.create_matrix(3, 4, :col_partition)  # Different partition scheme
    C = A * B  # Should fail with @mpiassert (partition mismatch)

else
    println("[Rank $rank] Unknown test case: $test_case")
    println("Available test cases: 1-10")
    exit(1)
end

# If we reach here, the test failed to catch the dimension error
println("[Rank $rank] ERROR: Test case $test_case did not catch dimension mismatch!")
exit(1)
