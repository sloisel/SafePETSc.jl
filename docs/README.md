# SafePETSc Documentation

This directory contains the documentation for SafePETSc.jl built with Documenter.jl.

## Building the Documentation

### Prerequisites

Install documentation dependencies from the package root:

```bash
# From the SafePETSc root directory
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

### Build Locally

```bash
# From the SafePETSc root directory
julia --project=docs docs/make.jl
```

The generated HTML documentation will be in `docs/build/`.

### View Locally

Open `docs/build/index.html` in your web browser, or use a local server:

```bash
# Python 3
cd docs/build && python3 -m http.server

# Then open http://localhost:8000 in your browser
```

## Documentation Structure

```
docs/
├── Project.toml          # Documentation dependencies
├── make.jl               # Build script
├── src/
│   ├── index.md          # Landing page
│   ├── getting_started.md
│   ├── guide/            # User guides
│   │   ├── distributed_refs.md
│   │   ├── vectors.md
│   │   ├── matrices.md
│   │   └── solvers.md
│   ├── api/              # API reference
│   │   ├── safempi.md
│   │   ├── vectors.md
│   │   ├── matrices.md
│   │   └── solvers.md
│   └── developer.md      # Developer guide
└── build/                # Generated documentation (gitignored)
```

## Updating Documentation

### Adding Pages

1. Create a new `.md` file in `docs/src/`
2. Add it to the `pages` array in `docs/make.jl`
3. Rebuild the documentation

### Adding Docstrings

Docstrings from the source code are automatically included via `@docs` blocks in the API reference pages. To add or update docstrings:

1. Edit the source code in `src/`
2. Add docstrings using Julia's triple-quoted string format
3. Reference them in API pages with ` ```@docs FunctionName``` `
4. Rebuild the documentation

## Deployment

To deploy to GitHub Pages, configure the `deploydocs` section in `make.jl` with your repository URL, then set up GitHub Actions or use Documenter's deployment features.

See [Documenter.jl documentation](https://documenter.juliadocs.org/) for more details.
