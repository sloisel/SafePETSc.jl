# -----------------------------------------------------------------------------
# map_rows - Apply function to rows of Vec/Mat
# -----------------------------------------------------------------------------

"""
    map_rows(f::Function, A::Union{Vec{T},Mat{T}}...; prefix="", col_partition=nothing) -> Union{Vec{T},Mat{T}}

**MPI Collective**

Apply a function `f` to corresponding rows across distributed PETSc vectors and matrices.

Similar to the native Julia pattern `vcat((f.((eachrow.(A))...))...)`, but works with
distributed PETSc objects. The function `f` is applied row-wise to each input, and the
results are concatenated into a new distributed vector or matrix.

# Arguments
- `f::Function`: Function to apply to each row. Should accept as many arguments as there are inputs.
- `A...::Union{Vec{T},Mat{T}}`: One or more distributed vectors or matrices. All inputs must have the same number of rows and compatible row partitions.
- `prefix::String`: PETSc options prefix for the result (default: uses prefix from first input)
- `col_partition::Union{Vector{Int},Nothing}`: Column partition for result matrix (default: use default_row_partition). Only used when `f` returns an adjoint vector (creating a matrix).

# Return value
The return type depends on what `f` returns:
- If `f` returns a scalar or Julia Vector → returns a `Vec{T}`
- If `f` returns an adjoint Julia Vector (row vector) → returns a `Mat{T}`

# Size behavior
If inputs have `m` rows and `f` returns:
- A scalar or adjoint vector → result has `m` rows
- An `n`-dimensional vector → result has `m*n` rows

# Examples
```julia
# Example 1: Sum rows of a matrix
B = Mat_uniform(randn(5, 3))
sums = map_rows(sum, B)  # Returns Vec with 5 elements (one sum per row)

# Example 2: Compute [sum, product] for each row (returns matrix)
stats = map_rows(x -> [sum(x), prod(x)]', B)  # Returns 5×2 Mat

# Example 3: Combine matrix and vector row-wise
C = Vec_uniform(randn(5))
combined = map_rows((x, y) -> [sum(x), prod(x), y[1]]', B, C)  # Returns 5×3 Mat
```

# Implementation notes
- This is a collective operation; all ranks must call it with compatible arguments
- The function `f` is assumed to be homogeneous (always returns the same type of output)
- For vectors, `f` receives a scalar value per row
- For matrices, `f` receives a view of the row (similar to eachrow)
"""
function map_rows(f::Function, A::Union{Vec{T},Mat{T}}...;
                  prefix::String="",
                  col_partition::Union{Vector{Int},Nothing}=nothing) where T

    # Validate inputs
    isempty(A) && throw(ArgumentError("map_rows requires at least one input"))

    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Use prefix from first argument if not provided
    if isempty(prefix)
        prefix = first(A).obj.prefix
    end

    # Get row partitions and check consistency
    row_partitions = [isa(a, Vec{T}) ? a.obj.row_partition : a.obj.row_partition for a in A]
    @mpiassert all(rp -> rp == row_partitions[1], row_partitions) "All inputs to map_rows must have the same row partition"

    input_row_partition = row_partitions[1]
    m = input_row_partition[end] - 1  # Total number of rows

    # Get local row range
    row_lo = input_row_partition[rank+1]
    row_hi = input_row_partition[rank+2] - 1
    nlocal_rows = row_hi - row_lo + 1

    # Handle empty local partition
    if nlocal_rows == 0
        # Need to determine output type from other ranks
        # For now, we'll handle this after we know the output type
        output_info = nothing
    else
        # Get iterators/arrays for each input
        local_data = []
        for a in A
            if isa(a, Vec{T})
                # For vectors, get local array
                arr = PETSc.unsafe_localarray(a.obj.v; read=true)
                push!(local_data, (arr, :vec))
            else  # Mat{T}
                # For matrices, use eachrow
                push!(local_data, (eachrow(a), :mat))
            end
        end

        # Extract first row to determine output type
        first_row_data = []
        for (data, dtype) in local_data
            if dtype == :vec
                # Wrap scalar in a 1-element view for consistency
                push!(first_row_data, view(data, 1:1))
            else  # :mat
                # Get first row from iterator
                iter_state = iterate(data)
                if iter_state !== nothing
                    push!(first_row_data, iter_state[1])
                end
            end
        end

        # Call f on first row to determine output characteristics
        sample_output = f(first_row_data...)

        # Determine output type and dimensions
        if isa(sample_output, Number)
            # Scalar output → Vec with m rows
            output_type = :vec_scalar
            output_rows = m
            output_cols = 1
        elseif isa(sample_output, LinearAlgebra.Adjoint) && isa(parent(sample_output), AbstractVector)
            # Adjoint vector (row vector) → Mat with m rows
            output_type = :mat
            output_rows = m
            output_cols = length(parent(sample_output))
        elseif isa(sample_output, AbstractVector)
            # Column vector → Vec with m*n rows
            output_type = :vec_vector
            output_rows = m * length(sample_output)
            output_cols = 1
            output_len = length(sample_output)
        else
            throw(ArgumentError("map_rows: function f must return a scalar, Vector, or adjoint Vector, got $(typeof(sample_output))"))
        end

        output_info = (output_type, output_rows, output_cols, sample_output)
    end

    # Broadcast output info to all ranks so ranks with empty partitions know what to create
    # We'll use a simple scheme: rank 0 decides (or first rank with non-empty partition)
    local_has_info = output_info !== nothing ? 1 : 0
    all_has_info = Vector{Int}(undef, nranks)
    MPI.Allgather!(Ref(local_has_info), all_has_info, MPI.COMM_WORLD)

    # Find first rank with info
    first_rank_with_info = findfirst(x -> x == 1, all_has_info)
    if first_rank_with_info === nothing
        throw(ArgumentError("map_rows: all ranks have empty local partitions"))
    end
    first_rank_with_info -= 1  # Convert to 0-based

    # Broadcast output type info from first rank with data
    if output_info !== nothing && rank == first_rank_with_info
        output_type_int = output_info[1] == :vec_scalar ? 1 : (output_info[1] == :vec_vector ? 2 : 3)
        output_rows_val = output_info[2]
        output_cols_val = output_info[3]
    else
        output_type_int = 0
        output_rows_val = 0
        output_cols_val = 0
    end

    # Use MPI.Bcast to share from the rank with info
    output_type_ref = Ref(output_type_int)
    output_rows_ref = Ref(output_rows_val)
    output_cols_ref = Ref(output_cols_val)

    MPI.Bcast!(output_type_ref, first_rank_with_info, MPI.COMM_WORLD)
    MPI.Bcast!(output_rows_ref, first_rank_with_info, MPI.COMM_WORLD)
    MPI.Bcast!(output_cols_ref, first_rank_with_info, MPI.COMM_WORLD)

    output_type = output_type_ref[] == 1 ? :vec_scalar : (output_type_ref[] == 2 ? :vec_vector : :mat)
    output_rows = output_rows_ref[]
    output_cols = output_cols_ref[]

    # Create output based on type
    if output_type == :vec_scalar || output_type == :vec_vector
        # Create Vec
        if output_type == :vec_vector
            # Broadcast output_len from first rank with info
            if output_info !== nothing && rank == first_rank_with_info
                output_len_val = length(output_info[4])
            else
                output_len_val = 0
            end
            output_len_ref = Ref(output_len_val)
            MPI.Bcast!(output_len_ref, first_rank_with_info, MPI.COMM_WORLD)
            output_len = output_len_ref[]

            # Partition for expanded rows
            output_row_partition = default_row_partition(output_rows, nranks)
            local_output_lo = output_row_partition[rank+1]
            local_output_hi = output_row_partition[rank+2] - 1
            local_output_size = local_output_hi - local_output_lo + 1
        else
            output_row_partition = input_row_partition
            local_output_size = nlocal_rows
        end

        # Create result vector
        result_petsc = _vec_create_mpi_for_T(T, local_output_size, output_rows, prefix, output_row_partition)
        result_local = PETSc.unsafe_localarray(result_petsc; read=true, write=true)

    else  # :mat
        # Create Mat
        output_row_partition = input_row_partition
        if col_partition === nothing
            col_partition = default_row_partition(output_cols, nranks)
        end

        # Create result matrix
        local_output_rows = nlocal_rows
        col_lo = col_partition[rank+1]
        col_hi = col_partition[rank+2] - 1
        local_output_cols = col_hi - col_lo + 1

        result_petsc = _mat_create_mpi_for_T(T, local_output_rows, local_output_cols,
                                              output_rows, output_cols, prefix)
    end

    # Now iterate and fill the result
    if nlocal_rows > 0
        # Re-acquire iterators/arrays for each input
        local_data = []
        finalizers = []
        for a in A
            if isa(a, Vec{T})
                arr = PETSc.unsafe_localarray(a.obj.v; read=true)
                push!(local_data, (arr, :vec, nothing))
                push!(finalizers, arr)
            else  # Mat{T}
                iter = eachrow(a)
                push!(local_data, (iter, :mat, iterate(iter)))
            end
        end

        try
            for i in 1:nlocal_rows
                # Extract current row from each input
                row_data = []
                for j in 1:length(local_data)
                    data, dtype, state = local_data[j]
                    if dtype == :vec
                        # Wrap scalar in a 1-element view for type consistency
                        push!(row_data, view(data, i:i))
                    else  # :mat
                        if state !== nothing
                            push!(row_data, state[1])
                            # Advance iterator
                            local_data[j] = (data, dtype, iterate(data, state[2]))
                        end
                    end
                end

                # Apply function
                result = f(row_data...)

                # Store result
                if output_type == :vec_scalar
                    result_local[i] = T(result)
                elseif output_type == :vec_vector
                    # Map local row index to output row indices
                    global_input_row = row_lo + i - 1
                    output_start = (global_input_row - 1) * output_len + 1
                    output_end = output_start + output_len - 1

                    # Check if these rows are in our local partition
                    local_output_lo = output_row_partition[rank+1]
                    local_output_hi = output_row_partition[rank+2] - 1

                    if output_start >= local_output_lo && output_end <= local_output_hi
                        # All output rows are local
                        local_idx_start = output_start - local_output_lo + 1
                        result_local[local_idx_start:local_idx_start+output_len-1] = result
                    else
                        # This shouldn't happen with proper partitioning, but handle it with SetValues
                        indices = collect(output_start:output_end)
                        values = Vector{T}(result)
                        _vec_setvalues!(result_petsc, indices, values, PETSc.INSERT_VALUES)
                    end
                else  # :mat
                    # Store row in matrix
                    global_row = row_lo + i - 1
                    result_vec = isa(result, LinearAlgebra.Adjoint) ? parent(result) : result
                    col_indices = collect(1:output_cols)
                    row_values = Vector{T}(result_vec)
                    _mat_setvalues!(result_petsc, [global_row], col_indices, row_values, PETSc.INSERT_VALUES)
                end
            end
        finally
            # Finalize all vector arrays
            for fin in finalizers
                Base.finalize(fin)
            end
        end
    end

    # Assemble and return result
    if output_type == :vec_scalar || output_type == :vec_vector
        Base.finalize(result_local)
        PETSc.assemble(result_petsc)
        obj = _Vec{T}(result_petsc, output_row_partition, prefix)
        return SafeMPI.DRef(obj)
    else  # :mat
        PETSc.assemble(result_petsc)
        obj = _Mat{T}(result_petsc, output_row_partition, col_partition, prefix)
        return SafeMPI.DRef(obj)
    end
end
