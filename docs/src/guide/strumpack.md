# STRUMPACK Solver Support

SafePETSc supports both MUMPS (default) and STRUMPACK as sparse direct solvers. STRUMPACK offers GPU acceleration and better MPI scalability for large-scale problems.

## Overview

| Feature | MUMPS (Default) | STRUMPACK |
|---------|-----------------|-----------|
| Included in PETSc_jll | Yes | No (requires build) |
| GPU Acceleration | Limited | CUDA, HIP, SYCL |
| MPI Scalability | Good | Better |
| Compression | BLR | BLR, HSS, HODLR |
| Setup | None | One-time build |

## When to Use STRUMPACK

Consider STRUMPACK when:
- You have access to GPUs (NVIDIA, AMD, or Intel)
- Your problems are large and MPI scalability matters
- You want to use low-rank compression (BLR, HSS) to reduce memory
- You're running on HPC clusters

Stick with MUMPS when:
- You want zero setup (just `using SafePETSc`)
- You're developing locally without GPUs
- Your problems are small to medium sized

## Installation

### Prerequisites

**System MPI is required.** The JLL-provided MPI wrappers have broken paths and cannot be used for building PETSc from source.

```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt-get install libopenmpi-dev
```

Then configure MPI.jl to use system MPI:

```julia
using MPIPreferences

# On macOS (Open MPI from Homebrew)
MPIPreferences.use_system_binary(;
    library_names=["libmpi"],
    extra_paths=["/opt/homebrew/Cellar/open-mpi/5.0.8/lib"],  # Adjust version as needed
    mpiexec="/opt/homebrew/bin/mpiexec"
)

# On Linux (usually auto-detects)
MPIPreferences.use_system_binary()

# Restart Julia after configuring
```

**Important**: MPI.jl must use the **same MPI implementation** as the PETSc build. If PETSc is built with Open MPI, MPI.jl must also use Open MPI. Mixing Open MPI and MPICH causes crashes.

### One-Time Build

STRUMPACK requires building PETSc from source with both STRUMPACK and MUMPS enabled:

```julia
using SafePETSc

# CPU-only build (works on macOS and Linux)
SafePETSc.build_petsc_strumpack()

# Build with GPU support (Linux only, requires SDK installed)
SafePETSc.build_petsc_strumpack(with_cuda=true)   # NVIDIA GPUs
SafePETSc.build_petsc_strumpack(with_hip=true)    # AMD GPUs
SafePETSc.build_petsc_strumpack(with_sycl=true)   # Intel GPUs
```

**GPU Build Requirements** (Linux only):
| GPU | Parameter | SDK Required |
|-----|-----------|--------------|
| NVIDIA | `with_cuda=true` | CUDA Toolkit (`nvcc`) |
| AMD | `with_hip=true` | ROCm (`hipcc`) |
| Intel | `with_sycl=true` | oneAPI (`icpx`) |

GPU SDKs are not available on macOS. The CPU-only build still provides STRUMPACK's
compression features and MPI scalability benefits.

The build function will:
1. Download PETSc source code
2. Compile with STRUMPACK, MUMPS, and dependencies (METIS, ParMETIS, ScaLAPACK)
3. Add GPU support if requested and SDK is available
4. Cache the build in Julia's scratch space
5. Offer to configure your `startup.jl` automatically

### Post-Build Setup

After building, you need to set the `JULIA_PETSC_LIBRARY` environment variable **before PETSc.jl is loaded**. This is because PETSc.jl selects the library at precompile time using `@static if`.

The build function offers to do this automatically by modifying `~/.julia/config/startup.jl`.

If you prefer manual setup, add this to your `startup.jl`:

```julia
ENV["JULIA_PETSC_LIBRARY"] = "/path/to/libpetsc.dylib"  # or .so on Linux
```

Get the path with:
```julia
julia> SafePETSc.petsc_strumpack_library_path()
```

Then:
1. Restart Julia
2. Delete PETSc's compiled cache to force recompilation with the new library:
   ```bash
   rm -rf ~/.julia/compiled/v1.*/PETSc ~/.julia/compiled/v1.*/SafePETSc
   ```
3. Start Julia again - PETSc will recompile with the STRUMPACK library

### HPC Clusters

On clusters with non-shared home directories, either:

1. Build to a shared location (with GPU support if available):
```julia
# On NVIDIA GPU cluster
SafePETSc.build_petsc_strumpack(install_dir="/software/petsc-strumpack", with_cuda=true)

# On AMD GPU cluster
SafePETSc.build_petsc_strumpack(install_dir="/software/petsc-strumpack", with_hip=true)
```

2. Or set the environment variable in your job script:
```bash
export JULIA_PETSC_LIBRARY="/path/to/libpetsc.so"
```

## Verifying Installation

Check if STRUMPACK build is available:

```julia
using SafePETSc

if SafePETSc.petsc_strumpack_available()
    println("STRUMPACK build found at: ", SafePETSc.petsc_strumpack_library_path())
else
    println("STRUMPACK build not found, using default PETSc_jll (MUMPS)")
end
```

## Using STRUMPACK

### Automatic Configuration

When `JULIA_PETSC_LIBRARY` points to a STRUMPACK-enabled build, `SafePETSc.Init()` automatically configures STRUMPACK as the default direct solver. No additional options are needed:

```julia
using SafePETSc
SafePETSc.Init()

# STRUMPACK is automatically used for direct solves
x = A \ b
```

### Manual Configuration

If you want to explicitly control the solver, use PETSc options with the `MPIAIJ_` prefix for sparse matrices:

```julia
using SafePETSc
SafePETSc.Init()

# Explicitly request LU factorization with STRUMPACK for sparse matrices
petsc_options_insert_string("-MPIAIJ_pc_type lu -MPIAIJ_pc_factor_mat_solver_type strumpack")

# Solve as usual
x = A \ b
```

**Note**: STRUMPACK is designed for sparse direct solves and is only configured for `MPIAIJ` matrices. Dense matrices (`MPIDENSE`) use different solvers.

SafePETSc uses prefixed options (`MPIAIJ_`, `MPIDENSE_`) for all matrices. Unprefixed options like `-pc_type` will be ignored.

### STRUMPACK Options

Configure STRUMPACK behavior with these PETSc options:

```julia
# Compression type (BLR recommended for most cases)
petsc_options_insert_string("-mat_strumpack_compression BLR")

# GPU acceleration (requires GPU-enabled build)
petsc_options_insert_string("-mat_strumpack_gpu")

# Verbose output for debugging
petsc_options_insert_string("-mat_strumpack_verbose")

# Compression tolerance (for approximate solvers)
petsc_options_insert_string("-mat_strumpack_compression_rel_tol 1e-4")
```

### Compression Types

| Type | Description | Use Case |
|------|-------------|----------|
| `NONE` | No compression (exact) | Small problems, maximum accuracy |
| `BLR` | Block Low Rank | General purpose, good default |
| `HSS` | Hierarchically Semi-Separable | Structured matrices |
| `HODLR` | Hierarchically Off-Diagonal Low Rank | Requires ButterflyPACK |

BLR is recommended as it works well without additional dependencies.

## GPU Acceleration

STRUMPACK supports multiple GPU backends:

| Backend | Hardware | Libraries |
|---------|----------|-----------|
| CUDA | NVIDIA GPUs | cuBLAS, cuSOLVER |
| HIP | AMD GPUs | rocBLAS, rocSOLVER |
| SYCL | Intel GPUs | oneMKL |

Enable GPU acceleration:

```julia
petsc_options_insert_string("-mat_strumpack_gpu")
```

**Important**: For multi-GPU setups, use 1 MPI process per GPU.

### GPU Notes

- STRUMPACK handles GPU memory management internally
- Input matrices and vectors stay on CPU
- GPU is used for numerical factorization
- No GPU-native matrix types needed (standard MPIAIJ works)

## Performance Comparison

Typical speedups with STRUMPACK GPU:

| Problem Size | CPU Only | GPU (BLR) | Speedup |
|--------------|----------|-----------|---------|
| Small | 1x | 0.8x | Overhead |
| Medium | 1x | 3-5x | Good |
| Large | 1x | 6-14x | Excellent |

GPU acceleration is most beneficial for larger problems where factorization dominates.

## Rebuilding

To rebuild with different options:

```julia
# Force rebuild
SafePETSc.build_petsc_strumpack(rebuild=true)

# Build with debugging symbols
SafePETSc.build_petsc_strumpack(rebuild=true, with_debugging=true)

# Build to custom location
SafePETSc.build_petsc_strumpack(install_dir="/custom/path")
```

## Troubleshooting

### Build Failures

**Missing compilers**: Ensure you have C, C++, and optionally Fortran compilers:
```bash
# macOS
xcode-select --install
brew install gcc

# Ubuntu/Debian
sudo apt-get install build-essential gfortran cmake
```

**Missing MPI**: The build requires system MPI (not the JLL wrappers). Install it and configure MPI.jl:
```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt-get install libopenmpi-dev
```

```julia
using MPIPreferences
MPIPreferences.use_system_binary()
# Restart Julia
```

### Runtime Issues

**"STRUMPACK not found"**: Ensure `JULIA_PETSC_LIBRARY` is set in `startup.jl` and:
1. Julia was restarted
2. PETSc's compiled cache was deleted (`rm -rf ~/.julia/compiled/v1.*/PETSc`)

**MPI mismatch error** ("Application was linked against both Open MPI and MPICH"):
This means MPI.jl is using a different MPI implementation than PETSc was built with. Check which MPI you have:
```julia
using MPI
MPI.identify_implementation()  # Should match what PETSc was built with
```

The STRUMPACK build uses system MPI (usually Open MPI from Homebrew). Configure MPI.jl to match:
```julia
using MPIPreferences
# On macOS with Homebrew Open MPI:
MPIPreferences.use_system_binary(;
    library_names=["libmpi"],
    extra_paths=["/opt/homebrew/Cellar/open-mpi/5.0.8/lib"],
    mpiexec="/opt/homebrew/bin/mpiexec"
)
# Then restart Julia and delete compiled caches
```

**Slow performance**:
- Check GPU is actually being used: `-mat_strumpack_verbose`
- Ensure 1 MPI rank per GPU for multi-GPU
- Try BLR compression for memory-bound problems

**Convergence issues with BLR**:
- Reduce compression tolerance: `-mat_strumpack_compression_rel_tol 1e-6`
- Or disable compression: `-mat_strumpack_compression NONE`

## API Reference

```@docs
build_petsc_strumpack
petsc_strumpack_library_path
petsc_strumpack_available
```

## See Also

- [Linear Solvers](@ref) - General solver documentation
- [PETSc STRUMPACK Documentation](https://petsc.org/release/manualpages/Mat/MATSOLVERSTRUMPACK/)
- [STRUMPACK GitHub](https://github.com/pghysels/STRUMPACK)
