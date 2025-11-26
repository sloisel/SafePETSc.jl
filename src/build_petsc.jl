# Build PETSc with STRUMPACK and MUMPS support
# This module provides functions to download, build, and configure PETSc with
# both STRUMPACK (for GPU acceleration and MPI scalability) and MUMPS (proven direct solver).
#
# IMPORTANT: Requires system MPI (brew install open-mpi or apt-get install libopenmpi-dev)
# The JLL-provided MPI wrappers have broken paths and cannot be used for building.

using Scratch
using Downloads

export build_petsc_strumpack, petsc_strumpack_library_path, petsc_strumpack_available

const PETSC_VERSION = "3.22.0"
const PETSC_TARBALL_URL = "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-$(PETSC_VERSION).tar.gz"

"""
    petsc_library_extension() -> String

Return the platform-specific shared library extension.
"""
function petsc_library_extension()
    if Sys.isapple()
        return "dylib"
    elseif Sys.iswindows()
        return "dll"
    else
        return "so"
    end
end

"""
    petsc_strumpack_library_path() -> String

Return the path to the STRUMPACK-enabled PETSc library in the scratch directory.

This path is consistent across Julia sessions and can be used to set
`JULIA_PETSC_LIBRARY` in your startup.jl file.

# Example
```julia
lib_path = petsc_strumpack_library_path()
# Returns something like: ~/.julia/scratchspaces/uuid/petsc_strumpack/lib/libpetsc.dylib
```
"""
function petsc_strumpack_library_path()
    build_dir = @get_scratch!("petsc_strumpack")
    joinpath(build_dir, "lib", "libpetsc.$(petsc_library_extension())")
end

"""
    petsc_strumpack_available() -> Bool

Check if a STRUMPACK-enabled PETSc build is available in the scratch directory.

# Example
```julia
if petsc_strumpack_available()
    println("STRUMPACK build found at: ", petsc_strumpack_library_path())
end
```
"""
function petsc_strumpack_available()
    isfile(petsc_strumpack_library_path())
end

"""
    has_strumpack() -> Bool

**MPI Non-Collective**

Check if the currently loaded PETSc library has STRUMPACK support.

This function checks if `JULIA_PETSC_LIBRARY` points to our STRUMPACK-enabled build.
It is used by `Init()` to automatically configure STRUMPACK as the default direct solver
when available.

# Example
```julia
if has_strumpack()
    println("STRUMPACK is available in the current PETSc library")
end
```

See also: [`petsc_strumpack_available`](@ref), [`build_petsc_strumpack`](@ref)
"""
function has_strumpack()
    lib_env = get(ENV, "JULIA_PETSC_LIBRARY", "")
    # Check if the environment variable points to our STRUMPACK build
    !isempty(lib_env) && occursin("petsc_strumpack", lib_env)
end

"""
    build_petsc_strumpack(;
        install_dir::Union{String,Nothing}=nothing,
        rebuild::Bool=false,
        with_debugging::Bool=false,
        with_cuda::Bool=false,
        verbose::Bool=true
    ) -> String

Download and build PETSc with both STRUMPACK and MUMPS solvers.

This function downloads PETSc source code and compiles it with both solvers enabled,
giving users the choice of direct solver. The build is cached, so subsequent calls
will skip the build if the library already exists (unless `rebuild=true`).

# Prerequisites
**System MPI is required.** The JLL-provided MPI wrappers have broken paths.

```bash
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt-get install libopenmpi-dev
```

Then configure MPI.jl to use system MPI:
```julia
using MPIPreferences
MPIPreferences.use_system_binary()
# Restart Julia
```

# Arguments
- `install_dir`: Custom installation directory. If `nothing`, uses Julia's Scratch.jl
  managed directory (recommended for most users).
- `rebuild`: Force rebuild even if library already exists.
- `with_debugging`: Build with debugging symbols (slower but useful for development).
- `with_cuda`: Enable NVIDIA GPU support (requires CUDA Toolkit, Linux only).
- `verbose`: Print progress messages during build.

# Returns
The path to the built PETSc library.

# Build Time
First build typically takes 30-60 minutes depending on your system.

# Post-Build Setup
After building, you need to set the `JULIA_PETSC_LIBRARY` environment variable
and restart Julia. The function will offer to set this up automatically.

# Example
```julia
using SafePETSc

# CPU-only build (works everywhere)
SafePETSc.build_petsc_strumpack()

# Build with NVIDIA GPU support (Linux only)
SafePETSc.build_petsc_strumpack(with_cuda=true)

# Follow the prompts to configure your startup.jl, then restart Julia
```

# Solver Selection
After setup, choose your solver via PETSc options:
```julia
# STRUMPACK (GPU acceleration, better MPI scalability)
petsc_options_insert_string("-pc_factor_mat_solver_type strumpack")
petsc_options_insert_string("-mat_strumpack_compression BLR")
petsc_options_insert_string("-mat_strumpack_gpu")  # Enable GPU if built with GPU support

# MUMPS (proven direct solver)
petsc_options_insert_string("-pc_factor_mat_solver_type mumps")
```

# HPC Clusters
On clusters with non-shared home directories, specify a shared location:
```julia
SafePETSc.build_petsc_strumpack(install_dir="/software/petsc-strumpack", with_cuda=true)
```
Then set `JULIA_PETSC_LIBRARY` in your job submission script or module file.

See also: [`petsc_strumpack_library_path`](@ref), [`petsc_strumpack_available`](@ref)
"""
function build_petsc_strumpack(;
    install_dir::Union{String,Nothing}=nothing,
    rebuild::Bool=false,
    with_debugging::Bool=false,
    with_cuda::Bool=false,
    verbose::Bool=true
)
    # Determine install location
    if install_dir === nothing
        install_dir = @get_scratch!("petsc_strumpack")
    end

    # Validate GPU options (CUDA SDK is Linux-only)
    if with_cuda && Sys.isapple()
        error("""
        CUDA support is not available on macOS.

        NVIDIA dropped macOS CUDA support in 2019.

        Use the default CPU-only build on macOS:
            SafePETSc.build_petsc_strumpack()
        """)
    end

    # Check CUDA SDK availability
    if with_cuda && !_check_cuda_available()
        error("""
        CUDA not found! To build with NVIDIA GPU support, install CUDA Toolkit first.

        Ubuntu: sudo apt install nvidia-cuda-toolkit
        Or download from: https://developer.nvidia.com/cuda-downloads
        """)
    end

    lib_path = joinpath(install_dir, "lib", "libpetsc.$(petsc_library_extension())")

    # Skip if already built (unless rebuild requested)
    if !rebuild && isfile(lib_path)
        if verbose
            @info "PETSc with STRUMPACK already built at: $lib_path"
        end
        _print_success_message(lib_path)
        return lib_path
    end

    if verbose
        @info """
        ╔══════════════════════════════════════════════════════════════════╗
        ║  Building PETSc with STRUMPACK                                   ║
        ║  This may take 30-60 minutes...                                  ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
    end

    # Create install directory
    mkpath(install_dir)

    # Download and extract PETSc source
    src_dir = _download_and_extract_petsc(install_dir, verbose)

    # Build PETSc with STRUMPACK
    _build_petsc_with_strumpack(src_dir, install_dir, with_debugging, with_cuda, verbose)

    # Verify the build succeeded
    if !isfile(lib_path)
        error("Build failed: library not found at $lib_path")
    end

    _print_success_message(lib_path)
    return lib_path
end

function _download_and_extract_petsc(install_dir::String, verbose::Bool)
    tarball_path = joinpath(install_dir, "petsc-$(PETSC_VERSION).tar.gz")
    src_dir = joinpath(install_dir, "petsc-$(PETSC_VERSION)")

    # Skip download if source already exists
    if isdir(src_dir)
        if verbose
            @info "PETSc source already exists at: $src_dir"
        end
        return src_dir
    end

    # Download tarball
    if !isfile(tarball_path)
        if verbose
            @info "Downloading PETSc $(PETSC_VERSION)..."
        end
        Downloads.download(PETSC_TARBALL_URL, tarball_path)
    end

    # Extract tarball
    if verbose
        @info "Extracting PETSc source..."
    end
    run(`tar -xzf $tarball_path -C $install_dir`)

    return src_dir
end

"""
    _check_system_mpi() -> Bool

Check if system MPI wrappers (mpicc, mpicxx) are available and working.
Returns false for broken JLL wrappers (which have /workspace/destdir paths).
"""
function _check_system_mpi()
    try
        # Check if mpicc exists and can show its config
        result = read(`mpicc -show`, String)
        # Reject broken JLL wrappers that have build-time paths
        !occursin("/workspace/destdir", result) && !isempty(result)
    catch
        false
    end
end

"""
    _check_cuda_available() -> Bool

Check if NVIDIA CUDA toolkit (nvcc) is available.
"""
function _check_cuda_available()
    try
        !isempty(read(`nvcc --version`, String))
    catch
        false
    end
end


function _build_petsc_with_strumpack(src_dir::String, install_dir::String, with_debugging::Bool,
                                     with_cuda::Bool, verbose::Bool)
    # Check for system MPI (required)
    if !_check_system_mpi()
        error("""
        No working system MPI found!

        The JLL-provided MPI wrappers have broken paths and cannot be used for building.
        You must install a system MPI first:

          macOS:  brew install open-mpi
          Ubuntu: sudo apt-get install libopenmpi-dev

        Then configure MPI.jl to use system MPI:

          julia> using MPIPreferences
          julia> MPIPreferences.use_system_binary()
          # Restart Julia

        After that, run build_petsc_strumpack() again.
        """)
    end

    # Check for Fortran MPI wrapper
    has_mpif90 = try
        success(`which mpif90`)
    catch
        false
    end

    # Build configuration flags - includes both STRUMPACK and MUMPS
    configure_flags = [
        "--prefix=$install_dir",
        "--with-cc=mpicc",
        "--with-cxx=mpicxx",
        has_mpif90 ? "--with-fc=mpif90" : "--with-fc=0",
        "--with-debugging=$(with_debugging ? 1 : 0)",
        "--with-shared-libraries=1",
    ]
    if verbose
        @info "Using system MPI compilers (mpicc, mpicxx)"
    end

    # Add common dependencies
    append!(configure_flags, [
        # Solvers
        "--download-strumpack",
        "--download-mumps",
        # Dependencies
        "--download-metis",
        "--download-parmetis",
        "--download-scalapack",
        "--download-ptscotch",
        "--download-bison",
        "--download-fblaslapack",
        "--download-cmake",
        # Disable features that conflict with Julia
        "--with-hdf5=0",
        "--with-curl=0",
        "--with-x=0",
    ])

    # Add CUDA GPU backend flags
    if with_cuda
        push!(configure_flags, "--with-cuda")
        if verbose
            @info "Enabling CUDA support for NVIDIA GPUs"
        end
        # SLATE is required for GPU-accelerated STRUMPACK (GPU-enabled ScaLAPACK alternative)
        # SLATE requires OpenMP support
        push!(configure_flags, "--with-openmp", "--download-slate")
        if verbose
            @info "Adding OpenMP and SLATE for GPU-accelerated ScaLAPACK"
        end
    end

    # Run configure
    if verbose
        @info "Configuring PETSc with STRUMPACK..."
        @info "Configure flags: $(join(configure_flags, " "))"
    end

    configure_cmd = Cmd(`python3 ./configure $(configure_flags)`; dir=src_dir)
    run(configure_cmd)

    # Run make
    if verbose
        @info "Building PETSc (this will take a while)..."
    end

    # Get number of CPU cores for parallel build
    ncores = max(1, Sys.CPU_THREADS ÷ 2)
    make_cmd = Cmd(`make -j$ncores all`; dir=src_dir)
    run(make_cmd)

    # Run make install
    if verbose
        @info "Installing PETSc..."
    end

    install_cmd = Cmd(`make install`; dir=src_dir)
    run(install_cmd)
end

function _setup_startup_jl(lib_path::String)
    startup_dir = joinpath(homedir(), ".julia", "config")
    startup_file = joinpath(startup_dir, "startup.jl")
    env_line = "ENV[\"JULIA_PETSC_LIBRARY\"] = \"$lib_path\""

    # Check if already configured
    if isfile(startup_file)
        content = read(startup_file, String)
        if occursin("JULIA_PETSC_LIBRARY", content)
            @info "startup.jl already contains JULIA_PETSC_LIBRARY setting"
            println("\nIf you want to update it, manually edit: $startup_file")
            return
        end
    end

    # Ask for confirmation
    println("\nWould you like to automatically add this to $startup_file?")
    println("    $env_line")
    print("Confirm (y/n): ")
    response = lowercase(strip(readline()))

    if response == "y" || response == "yes"
        mkpath(startup_dir)
        open(startup_file, "a") do f
            write(f, "\n# Added by SafePETSc.build_petsc_strumpack()\n")
            write(f, env_line, "\n")
        end
        @info """
        Added to $startup_file

        Please restart Julia for changes to take effect.
        """
    else
        @info """
        To enable STRUMPACK manually, add this line to $startup_file:

            $env_line

        Then restart Julia.
        """
    end
end

function _print_success_message(lib_path::String)
    @info """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PETSc with STRUMPACK + MUMPS is ready!                          ║
    ╚══════════════════════════════════════════════════════════════════╝

    Library path: $lib_path
    """

    _setup_startup_jl(lib_path)

    println("""
    ─────────────────────────────────────────────────────────────────────
    On HPC clusters with non-shared home directories, instead set
    JULIA_PETSC_LIBRARY in your job submission script or module file:

        export JULIA_PETSC_LIBRARY="$lib_path"
    ─────────────────────────────────────────────────────────────────────

    Solver Selection (after restart):

      STRUMPACK (GPU acceleration, better MPI scalability):
        petsc_options_insert_string("-pc_factor_mat_solver_type strumpack")
        petsc_options_insert_string("-mat_strumpack_compression BLR")
        petsc_options_insert_string("-mat_strumpack_gpu")

      MUMPS (proven direct solver):
        petsc_options_insert_string("-pc_factor_mat_solver_type mumps")
    """)
end
