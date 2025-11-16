using Documenter
using SafePETSc
using SparseArrays
using Pkg

# Compute version dynamically
version = string(pkgversion(SafePETSc))

makedocs(;
    modules=[SafePETSc, SafePETSc.SafeMPI],
    sitename="SafePETSc.jl $version",
    remotes=nothing,  # Disable remote source links
    warnonly=[:missing_docs, :cross_references, :docs_block],  # Don't fail on warnings
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sloisel.github.io/SafePETSc.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Vectors" => "guide/vectors.md",
            "Matrices" => "guide/matrices.md",
            "Linear Solvers" => "guide/solvers.md",
            "Input/Output and Display" => "guide/io.md",
            "Distributed Reference Management" => "guide/distributed_refs.md",
        ],
        "API Reference" => [
            "SafeMPI" => "api/safempi.md",
            "Vectors" => "api/vectors.md",
            "Matrices" => "api/matrices.md",
            "Solvers" => "api/solvers.md",
        ],
        "Developer Guide" => "developer.md",
    ],
)

deploydocs(;
    repo="github.com/sloisel/SafePETSc.jl",
    devbranch="main",
)
