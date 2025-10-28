# Gaussian Process Library

```@contents
Pages = ["gp.md"]
```

This section describes a library for Gaussian process time series models. A
technical overview of key concepts can be found in the following
references.

> Roberts S, Osborne M, Ebden M, Reece S, Gibson N, Aigrain S. 2013.
> Gaussian processes for time-series modelling.
> Phil Trans R Soc A 371: 20110550.
> [http://dx.doi.org/10.1098/rsta.2011.0550](http://dx.doi.org/10.1098/rsta.2011.0550)

> Rasmussen C, Williams C. 2006.
> Gaussian Processes for Machine Learning.
> MIT Press, Cambridge, MA.
> [http://gaussianprocess.org/gpml/chapters/](http://gaussianprocess.org/gpml/chapters/)


```@docs
AutoGP.GP
```

## [Covariance Kernels](@id gp_cov_kernel)

```@docs
AutoGP.GP.Node
AutoGP.GP.LeafNode
AutoGP.GP.BinaryOpNode
AutoGP.GP.pretty
AutoGP.GP.size
AutoGP.GP.eval_cov
AutoGP.GP.compute_cov_matrix
AutoGP.GP.compute_cov_matrix_vectorized
AutoGP.GP.extract_kernel
AutoGP.GP.split_kernel_sop
AutoGP.GP.reparameterize
AutoGP.GP.rescale
```

## [Primitive Kernels](@id gp_cov_kernel_prim)

**Notation**. In this section, generic parameters (e.g., ``\theta``,
``\theta_1``, ``\theta_2``), are used to denote fieldnames of the
corresponding Julia structs in the same order as they appear in the
constructors.

```@docs
AutoGP.GP.WhiteNoise
AutoGP.GP.Constant
AutoGP.GP.Linear
AutoGP.GP.SquaredExponential
AutoGP.GP.GammaExponential
AutoGP.GP.Periodic
```

## [Composite Kernels](@id gp_cov_kernel_comp)

```@docs
AutoGP.GP.Times
AutoGP.GP.Plus
AutoGP.GP.ChangePoint
```

## Prediction Utilities

```@docs
AutoGP.Distributions.MvNormal
AutoGP.Distributions.quantile
```


## Prior Configuration

```@docs
AutoGP.GP.GPConfig
```
