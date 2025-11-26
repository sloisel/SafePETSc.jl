# BlockProduct: Efficient multiplication of block matrices using PETSc reuse

"""
    BlockElement{T,Prefix}

Union type representing valid elements in a block matrix.

Elements can be:
- `Mat{T,Prefix}`: PETSc distributed matrix (must match the BlockProduct's Prefix)
- `Vec{T}`: PETSc distributed vector
- `Adjoint{T, Vec{T}}`: Adjoint of a vector
- `Adjoint{T, Mat{T,Prefix}}`: Adjoint of a matrix (must match the BlockProduct's Prefix)
- Scalar values (Number types), where `0` represents structural zeros
"""
const BlockElement{T,Prefix} = Union{Mat{T,Prefix}, SafeMPI.DRef{_Vec{T}}, Adjoint{T, SafeMPI.DRef{_Vec{T}}}, Adjoint{T, Mat{T,Prefix}}, Number}

"""
    BlockProduct{T,Prefix}

Represents a product of block matrices with pre-allocated storage for efficient recomputation.

A block matrix is a Julia `Matrix` where each element is a `Mat`, `Mat'`, `Vec`, `Vec'`, scalar, or `nothing`.

# Fields
- `prod::Vector{Matrix{BlockElement{T,Prefix}}}`: The sequence of block matrices to multiply
- `result::Union{Matrix{BlockElement{T,Prefix}}, Nothing}`: Pre-allocated result (allocated on first `calculate!` call)
- `intermediates::Vector{Matrix{BlockElement{T,Prefix}}}`: Pre-allocated intermediate results for chained products

# Type Parameters
- `T`: Element type (e.g., Float64)
- `Prefix`: PETSc prefix type (e.g., MPIAIJ, MPIDENSE) - must match all contained objects

# Constructor

    BlockProduct(prod::Vector{Matrix}; Prefix::Type=MPIAIJ)

Validates dimensions and creates a BlockProduct. Actual allocation of result and intermediates
happens lazily on the first call to `calculate!`.

# Example

```julia
# Create block matrices
A = [M1 M2; M3 M4]  # 2x2 block of Mat{Float64,MPIAIJ}
B = [N1 N2; N3 N4]

# Create product (no allocation yet)
bp = BlockProduct([A, B])

# Compute A * B (allocates result on first call)
C = calculate!(bp)

# Subsequent calls reuse allocations
C2 = calculate!(bp)
```
"""
mutable struct BlockProduct{T,Prefix}
    prod::Vector{Matrix{BlockElement{T,Prefix}}}
    result::Union{Matrix{BlockElement{T,Prefix}}, Nothing}
    intermediates::Vector{Matrix{BlockElement{T,Prefix}}}

    # Inner constructor that only validates, doesn't allocate
    function BlockProduct{T,Prefix}(prod::Vector{Matrix{BlockElement{T,Prefix}}}) where {T,Prefix}
        isempty(prod) && error("Product list cannot be empty")

        # Validate dimensions compatibility
        for k in 2:length(prod)
            if size(prod[k-1], 2) != size(prod[k], 1)
                error("Dimension mismatch: block $(k-1) has $(size(prod[k-1], 2)) columns, " *
                      "block $k has $(size(prod[k], 1)) rows")
            end
        end

        # Validate element types (prefix is now a type parameter)
        for (k, block) in enumerate(prod)
            for (i, j) in Iterators.product(axes(block)...)
                elem = block[i, j]
                _validate_element(elem, k, i, j, Prefix)
            end
        end

        new{T,Prefix}(prod, nothing, Matrix{BlockElement{T,Prefix}}[])
    end
end

# Helper to validate individual elements
# Note: Vec elements can have any Prefix (vectors don't need to match matrix Prefix),
# but Mat elements must match the expected Prefix.
function _validate_element(elem, block_idx, i, j, ExpectedPrefix::Type)
    if elem isa Number
        return  # scalar is OK (including 0 for structural zeros)
    elseif elem isa Mat
        # Check that elem is Mat{T,ExpectedPrefix} for some T
        if !(elem isa Mat{<:Any,ExpectedPrefix})
            elem_prefix_type = typeof(elem).parameters[2]
            error("Prefix mismatch at block $block_idx [$i,$j]: expected '$ExpectedPrefix', got '$elem_prefix_type'")
        end
    elseif elem isa Vec
        # Vec can have any Prefix - no validation needed
        # (vectors don't have a meaningful sparse/dense distinction like matrices)
        return
    elseif elem isa Adjoint && elem.parent isa Vec
        # Vec adjoint can have any Prefix - no validation needed
        return
    elseif elem isa Adjoint && elem.parent isa Mat
        # Check that elem.parent is Mat{T,ExpectedPrefix} for some T
        if !(elem.parent isa Mat{<:Any,ExpectedPrefix})
            elem_prefix_type = typeof(elem.parent).parameters[2]
            error("Prefix mismatch at block $block_idx [$i,$j]: expected '$ExpectedPrefix', got '$elem_prefix_type'")
        end
    else
        error("Invalid element type at block $block_idx [$i,$j]: $(typeof(elem))")
    end
end

"""
    BlockProduct(prod::Vector{Matrix}; Prefix::Type=MPIAIJ)

Create a BlockProduct from a vector of block matrices.

The element type `T` is inferred from the first PETSc object encountered.
The prefix type is inferred from the first Mat object, or uses the provided Prefix keyword.
"""
function BlockProduct(prod::Vector{<:Matrix}; Prefix::Type=MPIAIJ)
    # Infer element type from first PETSc object, and prefix from first Mat
    T = nothing
    InferredPrefix = nothing
    for block in prod
        # Use explicit 2D iteration to avoid linear indexing into Adjoint objects
        for i in axes(block, 1), j in axes(block, 2)
            elem = block[i, j]
            if elem isa Mat
                T = eltype(elem.obj.A)
                InferredPrefix = typeof(elem.obj).parameters[2]
                break
            elseif elem isa Vec
                T = eltype(elem.obj.v)
                # Vec has no Prefix, continue looking for a Mat
            elseif elem isa Adjoint && elem.parent isa Vec
                T = eltype(elem.parent.obj.v)
                # Vec has no Prefix, continue looking for a Mat
            elseif elem isa Adjoint && elem.parent isa Mat
                T = eltype(elem.parent.obj.A)
                InferredPrefix = typeof(elem.parent.obj).parameters[2]
                break
            end
        end
        # Only break if we found both T and InferredPrefix, or if we found a Mat
        (T !== nothing && InferredPrefix !== nothing) && break
    end

    if T === nothing
        # All scalars - default to Float64 and use provided Prefix
        T = Float64
        InferredPrefix = Prefix
    end

    # Use the inferred prefix if we found one (overrides keyword argument)
    FinalPrefix = InferredPrefix !== nothing ? InferredPrefix : Prefix

    # Convert input matrices to proper BlockElement{T,FinalPrefix} type
    # This allows Matrix{Float64} to become Matrix{BlockElement{Float64,FinalPrefix}}
    typed_prod = Vector{Matrix{BlockElement{T,FinalPrefix}}}(undef, length(prod))
    for (idx, block) in enumerate(prod)
        typed_block = Matrix{BlockElement{T,FinalPrefix}}(undef, size(block)...)
        for i in axes(block, 1), j in axes(block, 2)
            typed_block[i, j] = block[i, j]
        end
        typed_prod[idx] = typed_block
    end

    bp = BlockProduct{T,FinalPrefix}(typed_prod)
    # Perform initial calculation to cache all PETSc objects
    _calculate_init!(bp)
    return bp
end

"""
    _calculate_init!(bp::BlockProduct)

Initial calculation that allocates PETSc Mat/Vec objects and caches them.

Computes bp.prod[1] * bp.prod[2] * ... * bp.prod[end] using standard `*` operations.
All intermediate Mat/Vec objects created are cached in bp.result and bp.intermediates.

This is called automatically by the constructor. Users should call `calculate!()`
to update cached objects after modifying input matrices/vectors.

Returns the result as a block matrix (Julia Matrix of BlockElements).
"""
function _calculate_init!(bp::BlockProduct{T,Prefix}) where {T,Prefix}
    if length(bp.prod) == 1
        # Single matrix - just return it
        bp.result = bp.prod[1]
        return bp.result
    end

    # Perform left-to-right multiplication
    # Use standard Julia * operator which allocates new PETSc objects

    current = bp.prod[1]
    for k in 2:length(bp.prod)
        current = _block_multiply(current, bp.prod[k])
        # Store each intermediate result for later reuse
        push!(bp.intermediates, current)
    end

    bp.result = current
    return current
end

"""
    calculate!(bp::BlockProduct)

**MPI Collective**

Recompute the product after modifying input matrices/vectors, reusing cached PETSc objects.

After the user modifies entries in bp.prod[k][i,j] matrices or vectors, calling this function
updates bp.result using in-place operations on the cached intermediate results.

This avoids allocating new PETSc Mat/Vec objects.

Returns the updated result as a block matrix (Julia Matrix of BlockElements).

# Example

```julia
# Create and compute initial product
bp = BlockProduct([A, B])
result1 = bp.result

# Modify input matrix entries
# (modify A[1,1] entries here)

# Recompute with cached objects
calculate!(bp)
result2 = bp.result

# result2 has updated values but same PETSc object identity
```
"""
function calculate!(bp::BlockProduct{T,Prefix}) where {T,Prefix}
    if length(bp.prod) == 1
        # Single matrix - result is just that matrix
        return bp.result
    end

    # Recompute each intermediate in-place
    current = bp.prod[1]
    for k in 2:length(bp.prod)
        dest = bp.intermediates[k-1]  # Get cached intermediate
        _block_multiply_inplace!(dest, current, bp.prod[k])
        current = dest
    end

    # Result is the last intermediate
    bp.result = current
    return current
end

"""
    _block_multiply_inplace!(dest::Matrix{BlockElement{T,Prefix}}, A::Matrix{BlockElement{T,Prefix}}, B::Matrix{BlockElement{T,Prefix}})

Multiply two block matrices and store the result in dest, reusing PETSc objects where possible.

For Mat and Vec blocks in dest, uses in-place operations (mul!, VecCopy, MatCopy, VecAXPY, MatAXPY)
to update the values without allocating new PETSc objects.

For scalar blocks, just recomputes the values.
"""
function _block_multiply_inplace!(dest::Matrix{BlockElement{T,Prefix}}, A::Matrix{BlockElement{T,Prefix}}, B::Matrix{BlockElement{T,Prefix}}) where {T,Prefix}
    m, n = size(A, 1), size(B, 2)
    inner = size(A, 2)

    @assert inner == size(B, 1) "Inner dimensions must match"
    @assert size(dest) == (m, n) "Destination size must match result size"

    for i in 1:m, j in 1:n
        # Compute dest[i,j] = sum over k of A[i,k] * B[k,j]
        terms = []

        for k in 1:inner
            a = A[i, k]
            b = B[k, j]

            # Skip structural zeros
            if a isa Number && iszero(a)
                continue
            end
            if b isa Number && iszero(b)
                continue
            end

            # Compute product
            term = _multiply_elements(a, b)
            # Skip if result is zero
            if !(term isa Number && iszero(term))
                push!(terms, term)
            end
        end

        # Update destination
        if isempty(terms)
            dest[i, j] = 0  # structural zero
        elseif length(terms) == 1
            # Single term - use in-place copy if possible
            term = terms[1]
            if dest[i, j] isa Mat && term isa Mat
                _copy_mat_inplace!(dest[i, j], term)
            elseif dest[i, j] isa Vec && term isa Vec
                _copy_vec_inplace!(dest[i, j], term)
            else
                dest[i, j] = term
            end
        else
            # Multiple terms - accumulate into dest
            if dest[i, j] isa Mat && all(t isa Mat for t in terms)
                _accumulate_mats_inplace!(dest[i, j], terms)
            elseif dest[i, j] isa Vec && all(t isa Vec for t in terms)
                _accumulate_vecs_inplace!(dest[i, j], terms)
            else
                # Mixed types or scalars - fall back to allocation
                dest[i, j] = sum(terms)
            end
        end
    end

    return nothing
end

# Helper functions for in-place operations

# _copy_mat_inplace!(dest::Mat, src::Mat)
# Copy src matrix into dest matrix using PETSc MatCopy.
PETSc.@for_libpetsc begin
    function _copy_mat_inplace!(dest::Mat{$PetscScalar,Prefix}, src::Mat{$PetscScalar,Prefix}) where Prefix
        PETSc.@chk ccall(
            (:MatCopy, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat, PETSc.CMat, Cint),
            src.obj.A,
            dest.obj.A,
            Cint(0)  # SAME_NONZERO_PATTERN
        )
        return nothing
    end
end

# _copy_vec_inplace!(dest::Vec, src::Vec)
# Copy src vector into dest vector using PETSc VecCopy.
PETSc.@for_libpetsc begin
    function _copy_vec_inplace!(dest::Vec{$PetscScalar}, src::Vec{$PetscScalar})
        PETSc.@chk ccall(
            (:VecCopy, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CVec, PETSc.CVec),
            src.obj.v,
            dest.obj.v
        )
        return nothing
    end
end

# _accumulate_mats_inplace!(dest::Mat, terms::Vector)
# Zero dest and accumulate all term matrices into it using MatZeroEntries and MatAXPY.
PETSc.@for_libpetsc begin
    function _accumulate_mats_inplace!(dest::Mat{$PetscScalar,Prefix}, terms::Vector) where Prefix
        # Zero out dest
        PETSc.@chk ccall(
            (:MatZeroEntries, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CMat,),
            dest.obj.A
        )

        # Accumulate each term: dest = dest + 1.0 * term
        for term in terms
            PETSc.@chk ccall(
                (:MatAXPY, $libpetsc),
                PETSc.PetscErrorCode,
                (PETSc.CMat, $PetscScalar, PETSc.CMat, Cint),
                dest.obj.A,
                $PetscScalar(1.0),
                term.obj.A,
                Cint(1)  # DIFFERENT_NONZERO_PATTERN
            )
        end

        # Assemble the result
        PETSc.assemble(dest.obj.A)
        return nothing
    end
end

# _accumulate_vecs_inplace!(dest::Vec, terms::Vector)
# Zero dest and accumulate all term vectors into it using VecSet and VecAXPY.
PETSc.@for_libpetsc begin
    function _accumulate_vecs_inplace!(dest::Vec{$PetscScalar}, terms::Vector)
        # Zero out dest
        PETSc.@chk ccall(
            (:VecSet, $libpetsc),
            PETSc.PetscErrorCode,
            (PETSc.CVec, $PetscScalar),
            dest.obj.v,
            $PetscScalar(0.0)
        )

        # Accumulate each term: dest = dest + 1.0 * term
        for term in terms
            PETSc.@chk ccall(
                (:VecAXPY, $libpetsc),
                PETSc.PetscErrorCode,
                (PETSc.CVec, $PetscScalar, PETSc.CVec),
                dest.obj.v,
                $PetscScalar(1.0),
                term.obj.v
            )
        end

        return nothing
    end
end

"""
    _block_multiply(A::Matrix{BlockElement{T,Prefix}}, B::Matrix{BlockElement{T,Prefix}}) -> Matrix{BlockElement{T,Prefix}}

Multiply two block matrices element-wise using standard block matrix multiplication.

C[i,j] = sum_k A[i,k] * B[k,j]

Handles scalars, nothing (structural zeros), Mat, Vec, and Vec' appropriately.
"""
function _block_multiply(A::Matrix{BlockElement{T,Prefix}}, B::Matrix{BlockElement{T,Prefix}}) where {T,Prefix}
    m, n = size(A, 1), size(B, 2)
    inner = size(A, 2)

    @assert inner == size(B, 1) "Inner dimensions must match"

    # Create result matrix with proper BlockElement type
    result = Matrix{BlockElement{T,Prefix}}(undef, m, n)

    for i in 1:m, j in 1:n
        # Compute result[i,j] = sum over k of A[i,k] * B[k,j]
        terms = []

        for k in 1:inner
            a = A[i, k]
            b = B[k, j]

            # Skip structural zeros
            if a isa Number && iszero(a)
                continue
            end
            if b isa Number && iszero(b)
                continue
            end

            # Compute product
            term = _multiply_elements(a, b)
            # Skip if result is zero
            if !(term isa Number && iszero(term))
                push!(terms, term)
            end
        end

        # Sum all terms
        if isempty(terms)
            result[i, j] = 0  # structural zero
        elseif length(terms) == 1
            result[i, j] = terms[1]
        else
            # Need to sum - for now just use + operator
            result[i, j] = sum(terms)
        end
    end

    return result
end

"""
    _multiply_elements(a, b)

Multiply two block matrix elements, handling all type combinations.

Optimizes away multiplication by scalars 0 and 1.
"""
function _multiply_elements(a, b)
    # Handle scalar * scalar
    if a isa Number && b isa Number
        return a * b
    end

    # Handle scalar * object with optimizations
    if a isa Number
        if iszero(a)
            return 0
        elseif isone(a)
            return b
        else
            # scalar * PETSc object
            return a * b
        end
    end

    if b isa Number
        if iszero(b)
            return 0
        elseif isone(b)
            return a
        else
            # PETSc object * scalar
            return a * b
        end
    end

    # PETSc object * PETSc object - use existing * operator
    # This handles Mat*Mat, Mat*Vec, Vec'*Mat, Vec'*Vec automatically
    return a * b
end
