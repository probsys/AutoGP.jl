# AutoGP.jl

A Julia package for learning the covariance structure of Gaussian process time
series models.

[png](assets/tsdl.161.gif)


***

## Installation

First, obtain Julia 1.8 or later, available [here](https://julialang.org/downloads/).

The AutoGP package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and then run:

```julia
pkg> add AutoGP
```

## Tutorials

Please refer to the [Tutorials](./tutorials.html) page.

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


## License

AutoGP.jl is licensed under the Apache License, Version 2.0; refer to
[LICENSE](https://github.com/probsys/AutoGP.jl/LICENSE.txt).
