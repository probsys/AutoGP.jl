# AutoGP

The purpose of this package is to automatically infer Gaussian process (GP)
models of time series data, hence the name AutoGP.

At a high level, a Gaussian process $\{X(t) \mid t \in T \}$ is a family of
random variables indexed by some set $T$. For time series data, the index set
$T = \mathbb{R}$ is typically (a subset) of the real numbers.
For any list of $n$ time points $[t_1, \dots, t_n]$, the prior distribution
of the random vector $[X(t_1), \dots, X(t_n)]$ is a multivariate Gaussian,

```math
\begin{aligned}
    \begin{bmatrix}
        X(t_1)\\
        \vdots \\
        X(t_n)\\
    \end{bmatrix}
\sim \mathrm{MultivariteNormal} \left(
    \begin{bmatrix}
        m(t_1)\\
        \vdots \\
        m(t_n)\\
    \end{bmatrix},
    \begin{bmatrix}
        k_\theta(t_1, t_1) & \dots & k_\theta(t_1, t_n) \\
        \vdots & \ddots & \vdots \\
        k_\theta(t_n, t_1) & \dots & k_\theta(t_n, t_n) \\
    \end{bmatrix}
    \right).
\end{aligned}
```
In the equation above $m : T \to \mathbb{R}$ is the _mean
function_ and ``k_\theta: T \times T \to \mathbb{R}_{\ge 0}`` is the
_covariance (kernel) function_ parameterized by $\theta$. We typically
assume (without loss of generality) that $m(t) = 0$ for all $t$ and focus
our modeling efforts on the covariance kernel $k_\theta$. The structure
of the kernel dictates the qualitative properties of $X$, such as the existence
of linear trends, seasonal components, changepoints, etc.; refer to
the [kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/) for
an overview of these concepts.

Gaussian processes can be used as priors in Bayesian nonparametric models
of time series data $[Y(t_1), \dots, Y(t_n)]$ as follows. We assume that
each $Y(t_i) = g(X(t_i), \epsilon(t_i))$ for some known function $g$ and
noise process $\epsilon(t_i)$. A common choice is to let $Y$ be a copy of
$X$ corrupted with i.i.d. Gaussian innovations, which gives $$Y(t_i) =
X(t_i) + \epsilon(t_i)$$ where $\epsilon(t_i) \sim \mathrm{Normal}(0,
\eta)$ for some variance $\eta > 0$.

Writing out the full model
```math
\begin{aligned}
k
    &\sim \mathrm{PCFG}
    &&\textrm{(prior over covariance kernel)} \\
[\theta_1, \dots, \theta_{d(k)}]
    &\sim p_{k}
    && \textrm{(parameter prior)}\\
\eta
    &\sim p_{\eta}
    &&\textrm{(noise prior)} \\
[X(t_1), \dots, X(t_n)]
    &\sim \mathrm{MultivariteNormal}(\mathbf{0}, [k_\theta(t_i,t_j)]_{i,j=1}^n)
    &&\textrm{(Gaussian process)} \\
Y(t_i)
    &\sim \mathrm{Normal}(X(t_i), \eta), i=1,\dots,n
    &&\textrm{(noisy observations)}
\end{aligned}
```

where PCFG denotes a [probabilistic-context free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)
that defines a language of covariance kernel expressions $k$,
```math
B      &:= \textsc{Linear} \mid \textsc{Periodic} \mid \textsc{GammaExponential} \mid \dots \\
\oplus &:= \textsc{+} \mid \textsc{*} \mid \textsc{ChangePoint} \\
k      &:= B \mid \textsf{(}k_1 \oplus k_2\textsf{)}.
```
and $d(k)$ is the number of parameters in expression $k$.
Given data ``\{(t_i,y_i)\}_{i=1}^n``, AutoGP [infers](@ref end_to_end_model_fitting)
likely values of the symbolic structure of the covariance kernel $k$,
the kernel parameters $\theta$,
and the observation noise variance $\eta$.

The ability to automatically synthesize covariance kernels in AutoGP is
in contrast to existing Gaussian process libraries such as
[`GaussianProcess.jl`](https://github.com/STOR-i/GaussianProcesses.jl/)
[`sklearn.gaussian_process`](https://scikit-learn.org/stable/modules/gaussian_process.html),
[`GPy`](https://gpy.readthedocs.io/en/deploy/), and
[`GPyTorch`](https://gpytorch.ai/), which all require users to manually
specify $k$.

After model fitting, users can [query the models](@ref model_querying)
to generate forecasts, compute probabilities, and inspect qualitative structure.

```@docs
AutoGP
```

## Model Initialization

```@docs
AutoGP.GPModel
AutoGP.num_particles
AutoGP.seed!
AutoGP.IndexType
```

## [End-to-End Model Fitting](@id end_to_end_model_fitting)

```@docs
AutoGP.fit_smc!
AutoGP.fit_mcmc!
AutoGP.fit_greedy!
```

## Incremental Model Fitting

```@docs
AutoGP.add_data!
AutoGP.maybe_resample!
AutoGP.mcmc_structure!
AutoGP.mcmc_parameters!
```

## [Model Querying](@id model_querying)

```@docs
AutoGP.predict
AutoGP.predict_proba
AutoGP.predict_mvn
AutoGP.log_marginal_likelihood_estimate
AutoGP.particle_weights
AutoGP.covariance_kernels
```
