# AutoGP.jl

This package contains the Julia reference implementation of AutoGP, a
method for automatically discovering models of 1D time series data using
Gaussian processes, as described in

> _Sequential Monte Carlo Learning for Time Series Structure Discovery_.<br/>
> Saad, F A; Patton, B J; Hoffmann, M D.; Saurous, R A; Mansinghka, V K.<br/>
> ICML 2023: Proc. The 40th International Conference on Machine Learning.

## Installing

1. Obtain [Julia 1.8](https://julialang.org/downloads/) or later.
2. Clone this repository.
3. Set environment variable: `export JULIA_PROJECT=/path/to/AutoGP.jl`
4. Instantiate dependencies: `julia -e 'using Pkg; Pkg.instantiate()'`
5. Build PyCall: `PYTHON= julia -e 'using Pkg; Pkg.build("PyCall")'`
6. Verify import works: `julia -e 'import AutoGP; import PyPlot; println("success!")'`

## Documentation and Tutorials

1. Build the documentation

        $ julia docs/make.jl

2. Start a web server in `docs/build/`, e.g.,

        $ python3 -m http.server --directory docs/build/ --bind localhost 9090

3. Navigate to `http://localhost:9090` in your browser. Instructions for running
the tutorials are in `http://localhost:9090/tutorials.html`

## Citation

```bibtex
@inproceedings{saad2023icml,
title        = {Sequential Monte Carlo Learning for Time Series Structure Discovery},
author       = {Saad, Feras A. and Patton, Brian J. and Hoffmann, Matthew D. and Saurous, Rif A. and Mansinghka, V. K.},
booktitle    = {Proceedings of the 40th International Conference on Machine Learning},
series       = {Proceedings of Machine Learning Research},
fvolume      = {},
fpages       = {},
year         = {2023},
publisher    = {PMLR},
}
```
