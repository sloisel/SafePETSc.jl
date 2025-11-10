using Documenter
using SafePETSc

makedocs(;
    modules=[SafePETSc, SafePETSc.SafeMPI],
    sitename="SafePETSc.jl",
    remotes=nothing,  # Disable remote source links
    warnonly=[:missing_docs, :cross_references, :docs_block],  # Don't fail on warnings
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yourusername.github.io/SafePETSc.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Distributed Reference Management" => "guide/distributed_refs.md",
            "Vectors" => "guide/vectors.md",
            "Matrices" => "guide/matrices.md",
            "Linear Solvers" => "guide/solvers.md",
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

# Uncomment and configure when you have a GitHub repository:
# deploydocs(;
#     repo="github.com/yourusername/SafePETSc.jl",
#     devbranch="main",
# )
