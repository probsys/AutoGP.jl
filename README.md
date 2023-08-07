# AutoGP.jl

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://fsaad.github.io/AutoGP.jl/)
[![GitHub Action Status](https://github.com/fsaad/AutoGP.jl/workflows/Documentation/badge.svg)](https://github.com/fsaad/AutoGP.jl/actions/workflows/Documentation.yml)
[![GitHub Action Status](https://github.com/fsaad/AutoGP.jl/workflows/Tutorials/badge.svg)](https://github.com/fsaad/AutoGP.jl/actions/workflows/Tutorials.yml)
[![Git Tag](https://img.shields.io/github/v/tag/fsaad/AutoGP.jl)](https://github.com/fsaad/AutoGP.jl/tags)
[![License](https://img.shields.io/github/license/fsaad/AutoGP.jl?color=lightgrey)](https://github.com/fsaad/AutoGP.jl/blob/main/LICENSE.txt)


This package contains the Julia reference implementation of AutoGP, a
method for automatically discovering Gaussian process models of univariate
time series data, as described in

> _Sequential Monte Carlo Learning for Time Series Structure Discovery_.<br/>
> Saad, F A; Patton, B J; Hoffmann, M D.; Saurous, R A; Mansinghka, V K.<br/>
> ICML 2023: Proc. The 40th International Conference on Machine Learning.<br/>
> Proceedings of Machine Learning Research vol. 202, pages 29473-29489, 2023.

Whereas traditional Gaussian process software packages focus on inferring
the numeric parameters for a fixed (user-specified) covariance kernel
function, AutoGP learns both covariance kernel functions and numeric
parameters for a given dataset. The plots below show two examples of online
time series structure discovery using AutoGP, which discovers periodic
components, trends, and smoothly-varying temporal components.

<img style="float: left" src="./docs/src/assets/tsdl.161.gif" width="49%"/> <img style="float: center" src="./docs/src/assets/tsdl.533.gif" width="49%"/>

## Installing

AutoGP can be installed using the Julia package manager. From the Julia
REPL (version 1.8+), type `]` to enter the Pkg REPL mode and run

```
pkg> add AutoGP
```

Alternatively, use the terminal command
`julia -e 'import Pkg; Pkg.add("AutoGP")'`.

## Tutorials

Please see https://fsaad.github.io/AutoGP.jl


## Developer Notes

#### Building Documentation

```
$ julia --project=. docs/make.jl
$ python3 -m http.server --directory docs/build/ --bind localhost 9090
```

#### Building From Clone

1. Obtain [Julia 1.8](https://julialang.org/downloads/) or later.
2. Clone this repository.
3. Set environment variable: `export JULIA_PROJECT=/path/to/AutoGP.jl`
4. Instantiate dependencies: `julia -e 'using Pkg; Pkg.instantiate()'`
5. Build PyCall: `PYTHON= julia -e 'using Pkg; Pkg.build("PyCall")'`
6. Verify import works: `julia -e 'import AutoGP; import PyPlot; println("success!")'`

## Citation

```bibtex
@inproceedings{saad2023icml,
title        = {Sequential {Monte} {Carlo} Learning for Time Series Structure Discovery},
author       = {Saad, Feras A. and Patton, Brian J. and Hoffmann, Matthew D. and Saurous, Rif A. and Mansinghka, V. K.},
booktitle    = {Proceedings of the 40th International Conference on Machine Learning},
series       = {Proceedings of Machine Learning Research},
volume       = {202},
pages        = {29473--29489},
year         = {2023},
publisher    = {PMLR},
}
```
