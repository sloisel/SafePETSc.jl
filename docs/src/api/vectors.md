# Vectors API Reference

Distributed vector operations in SafePETSc.

## Type

```@docs
SafePETSc.Vec
```

## Constructors

```@docs
SafePETSc.Vec_uniform
SafePETSc.Vec_sum
```

## Helper Constructors

```@docs
SafePETSc.zeros_like
SafePETSc.ones_like
SafePETSc.fill_like
```

## Partitioning

```@docs
SafePETSc.default_row_partition
```

## Vector Pooling

```@docs
SafePETSc.ENABLE_VEC_POOL
SafePETSc.clear_vec_pool!
SafePETSc.get_vec_pool_stats
```

## Conversion and Display

Convert distributed vectors to Julia arrays for inspection and display:

```@docs
Base.Vector(::SafePETSc.Vec)
```

Display methods (automatically used by `println`, `display`, etc.):
- `show(io::IO, v::Vec)` - Display vector contents
- `show(io::IO, mime::MIME, v::Vec)` - Display with MIME type support

## Utilities

```@docs
SafePETSc.io0
```

## Operations

### Arithmetic

Vectors support standard Julia arithmetic operations via broadcasting:

```julia
y = x .+ 1.0        # Element-wise addition
y = 2.0 .* x        # Scaling
z = x .+ y          # Vector addition
y .= x .+ 1.0       # In-place operation
```

Standard operators are also overloaded:

```julia
z = x + y           # Addition
z = x - y           # Subtraction
z = -x              # Negation
```

### Linear Algebra

```julia
y = A * x                              # Matrix-vector multiplication
LinearAlgebra.mul!(y, A, x)            # In-place multiplication
w = v' * A                             # Adjoint-vector times matrix
LinearAlgebra.mul!(w, v', A)           # In-place
```

### Properties

```julia
T = eltype(v)                          # Element type
n = length(v)                          # Vector length
n = size(v, 1)                         # Size in dimension 1
```
