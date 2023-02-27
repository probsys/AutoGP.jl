# AutoGP

_Warning: This is rapidly evolving research software._

This package contains the reference implementation of AutoGP, a method
for automatically learning the structure of time series data using
Gaussian process models.

## Installing

This software is tested on Debian 5.19.11-1rodete1 (2022-10-31) x86_64 GNU/Linux.

1. Obtain [Julia 1.8](https://julialang.org/downloads/) or later.
2. Clone this repository.
3. Set environment variable: `export JULIA_PROJECT=/path/to/AutoGP.jl`
4. Instantiate dependencies: `julia -e 'using Pkg; Pkg.instantiate()'`

## Documentation and Tutorials

1. First build the documentation

        $ JULIA_PROJECT=. julia docs/make.jl

2. Start a web server in `docs/build/`, e.g.,

        $ python3 -m http.server --directory docs/build/ --bind localhost 9090

3. Navigate to `localhost:9090` in your browser.
