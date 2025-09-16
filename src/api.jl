# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import DataFrames
import Distributions
import Random

import Gen

using Dates
using Match

using Distributions: MixtureModel
using Distributions: MvNormal
using Distributions: Normal

"""
    seed!(seed)
Set the random seed of the global random number generator.
"""
function seed!(seed)
    Random.seed!(seed)
    Gen.seed!(seed)
end

"""
    IndexType = Union{Vector{<:Real}, Vector{<:Date}, Vector{<:DateTime}}

Permitted Julia types for Gaussian process time points.
`Real` numbers are ingested directly, treated as time points.
Instances of the `Dates` types are converted to numeric time points by using
[`Dates.datetime2unix`](https://docs.julialang.org/en/v1/stdlib/Dates/#Dates.datetime2unix).
"""
const IndexType = Union{Vector{<:Real}, Vector{<:Date}, Vector{<:DateTime}}

to_numeric(t::DateTime) = datetime2unix(t)
to_numeric(t::Date) = to_numeric(DateTime(t))
to_numeric(t::Real) = t

"""
    struct GPModel

A `GPModel` contains covariance kernel structures and parameters for modeling data.

# Fields

- `pf_state::Gen.ParticleFilterState`: Internal particle set.
- `config::GP.GPConfig=GP.GPConfig()`: User-specific customization, refer to [`GP.GPConfig`](@ref).
- `ds::IndexType`: Observed time points.
- `y::Vector{<:Real}`: Observed time series values.
- `ds_transform::Transforms.LinearTransform`: Transformation of time to direct space.
- `y_transform::Transforms.LinearTransform`: Transformation of observations to direct space.

# Constructors

    model = GPModel(
        ds::IndexType,
        y::Vector{<:Real};
        n_particles::Integer=8,
        config::GP.GPConfig=GP.GPConfig())

# See also

To perform learning given the data, refer to
- [`AutoGP.fit_smc!`](@ref)
- [`AutoGP.fit_mcmc!`](@ref)
- [`AutoGP.fit_greedy!`](@ref)

"""
mutable struct GPModel
    pf_state::Gen.ParticleFilterState
    config::GP.GPConfig
    ds::IndexType
    y::Vector{<:Real}
    ds_transform::Transforms.LinearTransform
    y_transform::Transforms.LinearTransform
end

function GPModel(
        ds::IndexType,
        y::Vector{<:Real};
        n_particles::Integer=Threads.nthreads(),
        config::GP.GPConfig=GP.GPConfig())
    # Save the transformations.
    ds_transform = Transforms.LinearTransform(to_numeric.(ds), 0, 1)
    y_transform = Transforms.LinearTransform(Vector{Float64}(y), 1)
    # Transform the data.
    ds_numeric = Transforms.apply(ds_transform, to_numeric.(ds))
    y_numeric = Transforms.apply(y_transform, y)
    # Initialize the particle filter.
    observations = Gen.choicemap((:xs, y_numeric))
    if !isnothing(config.noise)
        observations[:noise] = Model.untransform_param(:noise, config.noise)
    end
    pf_state = Gen.initialize_particle_filter(
        Model.model, (ds_numeric, config), observations, n_particles)
    # Return the state.
    return GPModel(pf_state, config, collect(ds), collect(y), ds_transform, y_transform)
end

"""
    particle_weights(model::GPModel)
Return vector of normalized particle weights.
"""
particle_weights(model::GPModel) = Inference.compute_particle_weights(model.pf_state)

"""
    effective_sample_size(model::GPModel)
Return effective sample size (ESS) of weighted particle collection.
"""
effective_sample_size(model::GPModel) = Inference.effective_sample_size(model.pf_state)

"""
    log_marginal_likelihood_estimate(model::GPModel)
Return estimate of marginal likelihood of data (in log space).
"""
log_marginal_likelihood_estimate(model::GPModel) = model.pf_state.log_ml_est

"""
    num_particles(model::GPModel)
Return the number of particles.
"""
num_particles(model::GPModel) = length(model.pf_state.traces)

"""
    covariance_kernels(model::GPModel; reparameterize::Bool=true)
Return [Gaussian process covariance kernels](@ref gp_cov_kernel) in `model`.
If `reparameterize` is `true` (default), then the kernel parameters are
given in the original data space (more interpretable); otherwise they are
given in the transformed space over which parameter inference is performed
(useful for debugging).
"""
function covariance_kernels(model::GPModel; reparameterize::Bool=true)
    kernels = [trace[] for trace in model.pf_state.traces]
    if reparameterize
        y_transform_inv = Transforms.invert(model.y_transform)
        kernels = GP.reparameterize.(kernels, [model.ds_transform])
        kernels = GP.rescale.(kernels, [y_transform_inv])
    end
    return kernels
end

"""
    observation_noise_variances(model::GPModel; reparameterize::Bool=true)
Return list of observation noise variances for each particle in `model`.
If `reparameterize` is `true` (default), then the kernel parameters are
given in the original data space (more interpretable); otherwise they are
given in the transformed space over which parameter inference is performed
(useful for debugging).
"""
function observation_noise_variances(model::GPModel; reparameterize::Bool=true)
    noises = [t[:noise] for t in model.pf_state.traces]
    noises = Model.transform_param.(:noise, noises) .+ AutoGP.Model.JITTER
    if reparameterize
        noises = Transforms.unapply_var.([model.y_transform], noises)
    end
    return noises
end

"""
    fit_smc!(
        model::GPModel;
        schedule::Vector{<:Integer},
        n_mcmc::Int,
        n_hmc::Int,
        shuffle::Bool=true,
        biased::Bool=false,
        adaptive_resampling::Bool=true,
        adaptive_rejuvenation::Bool=false,
        hmc_config::Dict=Dict(),
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)

Infer the structure and parameters of an appropriate
[Gaussian process covariance kernel](@ref gp_cov_kernel) for modeling
the observed data. Inference is performed using sequential Monte Carlo.

# Arguments
- `model::GPModel`: Instance of the `GPModel` to use.
- `schedule::Vector{<:Integer}`: Schedule for incorporating data for SMC, refer to [`Schedule`](@ref).
- `n_mcmc::Union{Integer,Vector{<:Integer}}`: Number of involutive MCMC rejuvenation steps. If vector, must have same length as `schedule`.
- `n_hmc::Union{Integer,Vector{<:Integer}}`: Number of HMC steps per accepted involutive MCMC step. If vector, must have same length as `schedule`.
- `biased::Bool`:   Whether to bias the proposal to produce "short" structures.
- `shuffle::Bool=true`: Whether to shuffle indexes `ds` or incorporate data in the given order.
- `adaptive_resampling::Bool=true`: If `true` resamples based on ESS threshold, else at each step.
- `adaptive_rejuvenation::Bool=false`: If `true` rejuvenates only if resampled, else at each step.
- `hmc_config::Dict`: Configuration for HMC inference on numeric parameters. Allowable keys are:
    - `n_exit::Integer=1`: Number of successive rejections after which HMC loop is terminated.
    - `L_param::Integer=10` Number of leapfrog steps for kernel parameters.
    - `L_noise::Integer=10`: Number of leapfrog steps for noise parameter.
    - `eps_param::Float64=0.02`: Step size for kernel parameters.
    - `eps_noise::Float64=0.02`: Step size for noise parameter.
- `verbose::Bool=false`: Report progress to stdout.
- `check::Bool=false`: Perform dynamic correctness checks during inference.
- `config::GP.GPConfig=GP.GPConfig()`: User-specific customization, refer to [`GP.GPConfig`](@ref).
- `callback_fn`: A callback for monitoring inference, must be generated by [`AutoGP.Callbacks.make_smc_callback`](@ref).
"""
function fit_smc!(
        model::GPModel;
        schedule::Vector{<:Integer},
        n_mcmc::Union{Integer,Vector{<:Integer}},
        n_hmc::Union{Integer,Vector{<:Integer}},
        biased::Bool=false,
        shuffle::Bool=true,
        adaptive_resampling::Bool=true,
        adaptive_rejuvenation::Bool=false,
        hmc_config=(n_exit=1,),
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)
    if Threads.nthreads() < num_particles(model)
        @warn "Using more particles than available threads."
    end
    # Obtain observed data.
    n = length(model.ds)
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(model.ds))
    y_numeric = Transforms.apply(model.y_transform, model.y)
    permutation = shuffle ? Random.randperm(n) : collect(1:n)
    # Run SMC.
    model.pf_state = Inference.run_smc_anneal_data(
        ds_numeric, y_numeric;
        config=model.config,
        biased=biased,
        n_particles=num_particles(model),
        n_mcmc=n_mcmc,
        n_hmc=n_hmc,
        hmc_config=hmc_config,
        permutation=permutation,
        schedule=schedule,
        adaptive_resampling=adaptive_resampling,
        adaptive_rejuvenation=adaptive_rejuvenation,
        verbose=verbose,
        check=check,
        callback_fn=callback_fn)
end

"""
    fit_mcmc!(
        model::GPModel;
        n_mcmc::Integer,
        n_hmc::Integer,
        biased::Bool=false,
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)

Perform `n_mcmc` steps of involutive MCMC on the structure, with `n_hmc`
steps of Hamiltonian Monte Carlo sampling on the parameters per accepted
involutive MCMC move.

A `callback_fn` can be provided to monitor the progress each MCMC step for
which at least one particle (i.e, chain) accepted a transition. Its
signature must contain a single varargs specifier, which will be populated
with keys `:model`, `:step`, `:elapsed`.

!!! warning

    The `callback_fn` imposes a roughly 2x runtime overhead as compared to
    the equivalent [`mcmc_structure!`](@ref) method, because parallel
    execution must be synchronized across the particles to invoke the
    callback at each step. The `:elapsed` variable provided to the callback
    function will still reflect an accurate estimate of the inference
    runtime without this overhead. If no callback is required, use
    [`mcmc_structure!`](@ref) instead.
"""
function fit_mcmc!(
        model::GPModel;
        n_mcmc::Integer,
        n_hmc::Integer,
        hmc_config=(n_exit=2,),
        biased::Bool=false,
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)
    n_particles = num_particles(model)
    accepted = Vector{Bool}(undef, n_particles)
    elapsed = zeros(n_particles)
    for step=1:n_mcmc
        Threads.@threads for i=1:n_particles
            TimeIt.@timeit elapsed[i] begin
                trace = model.pf_state.traces[i]
                observations = Inference.get_observations_choicemap(trace)
                trace, stats = Inference.rejuvenate_particle_structure(
                            trace, 1, n_hmc, biased;
                            hmc_config=hmc_config, verbose=verbose,
                            check=check, observations=observations)
                model.pf_state.traces[i] = trace
                accepted[i] = (stats[:mh] > 0)
            end
        end
        any(accepted) && callback_fn(model=model, step=step, elapsed=elapsed)
    end
end

"""
    function fit_greedy!(
            model::GPModel;
            max_depth::Integer=model.config.max_depth,
            verbose::Bool=false,
            check::Bool=false,
            callback_fn::Function=(; kwargs...) -> nothing)

Infer the structure and parameters of an appropriate
[Gaussian process covariance kernel](@ref gp_cov_kernel) for modeling
the observed data. Inference is performed using greedy search, as described
in [Algorithm 2 of Kim and Teh, 2018](https://arxiv.org/pdf/1706.02524.pdf#page=11.pdf).
It is an error if `max_depth` is not a finite positive number.

A `callback_fn` can be provided to monitor the search progress at each stage.
Its signature must contain a single varargs specifier, which will be populated
with keys `:model`, `:step`, `:elapsed` at each step of the greedy search.
"""
function fit_greedy!(
        model::GPModel;
        max_depth::Integer=model.config.max_depth,
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)
    # Error checking.
    (num_particles(model) == 1) || error("AutoGP.fit_greedy! requires exactly 1 particle.")
    !model.config.changepoints || error("AutoGP.fit_greedy! does not support changepoint operators.")
    (1 <= max_depth <= (model.config.max_depth == -1 ? Inf : model.config.max_depth)) || error("AutoGP.fit_greedy! requires positive and finite max_depth.")
    # Prepare observations.
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(model.ds))
    y_numeric = Transforms.apply(model.y_transform, model.y)
    # Helper function for creating intermediate models for callback.
    make_greedy_submodel = (trace) -> begin
        pf_state = Gen.initialize_particle_filter(
            Model.model, (ds_numeric, model.config), Gen.get_choices(trace), 1)
        GPModel(
            pf_state,
            model.config,
            model.ds,
            model.y,
            model.ds_transform,
            model.y_transform)
    end
    # Initialize search at depth 1.
    elapsed = 0.
    TimeIt.@timeit elapsed begin
        trace, aic, observations = Greedy.greedy_search_initialize(
            ds_numeric, y_numeric, model.config; check=check)
    end
    submodel = make_greedy_submodel(trace)
    callback_fn(model=submodel, step=1, aic=aic, elapsed=elapsed)
    # Continue search up to max depth.
    for depth=2:max_depth
        TimeIt.@timeit elapsed begin
            (trace, new_aic, accepted) = Greedy.greedy_search_extend(
                trace, aic;
                verbose=verbose,
                check=check,
                observations=observations)
        end
        @assert new_aic <= aic
        aic = new_aic
        submodel = make_greedy_submodel(trace)
        callback_fn(model=submodel, step=depth, aic=aic, elapsed=elapsed)
    end
    # Update the model.
    pf_state = Gen.initialize_particle_filter(
        Model.model, (ds_numeric, model.config), Gen.get_choices(trace), 1)
    model.pf_state = pf_state
end

"""
    mcmc_parameters!(model::GPModel, n_hmc::Integer; verbose::Bool=false, check::Bool=false)

Perform `n_hmc` steps of Hamiltonian Monte Carlo sampling on the parameters.
"""
function mcmc_parameters!(model::GPModel, n_hmc::Integer; verbose::Bool=false, check::Bool=false)
    Threads.@threads for i=1:num_particles(model)
        trace = model.pf_state.traces[i]
        observations = Inference.get_observations_choicemap(trace)
        trace, n_accept = Inference.rejuvenate_particle_parameters(
            trace, n_hmc; verbose=verbose, check=check, observations=observations)
        model.pf_state.traces[i] = trace
    end
end

"""
    mcmc_structure!(model::GPModel, n_mcmc::Integer, n_hmc::Integer;
        biased::Bool=false, verbose::Bool=false, check::Bool=false)

Perform `n_mcmc` steps of involutive MCMC on the structure, with `n_hmc`
steps of Hamiltonian Monte Carlo sampling on the parameters per accepted
involutive MCMC move.
"""
function mcmc_structure!(
        model::GPModel,
        n_mcmc::Integer,
        n_hmc::Integer;
        hmc_config=(n_exit=2,),
        biased::Bool=false,
        verbose::Bool=false,
        check::Bool=false)
    Threads.@threads for i=1:num_particles(model)
        trace = model.pf_state.traces[i]
        observations = Inference.get_observations_choicemap(trace)
        trace, stats = Inference.rejuvenate_particle_structure(
                    trace, n_mcmc, n_hmc, biased;
                    hmc_config=hmc_config, verbose=verbose,
                    check=check, observations=observations)
        model.pf_state.traces[i] = trace
    end
end

"""
    add_data!(model::GPModel, ds::IndexType, y::Vector{<:Real})
Incorporate new observations `(ds, y)` into `model`.
"""
function add_data!(model::GPModel, ds::IndexType, y::Vector{<:Real})
    if !(eltype(ds) <: eltype(model.ds))
        error("Invalid time $(ds), expected $(eltype(model.ds))")
    end
    # Append the data.
    append!(model.ds, ds)
    append!(model.y, y)
    # Convert to numeric.
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(model.ds))
    y_numeric = Transforms.apply(model.y_transform, model.y)
    # Prepare observations.
    observations = Gen.choicemap((:xs, y_numeric))
    !isnothing(model.config.noise) && (observations[:noise] = trace[:noise])
    # Run SMC step.
    Inference.smc_step!(model.pf_state, (ds_numeric, model.config), observations)
end

"""
    remove_data!(model::GPModel, ds::IndexType, y::Vector{<:Real})
Remove existing observations `ds` from `model`.
"""
function remove_data!(model::GPModel, ds::IndexType)
    # Find the data point.
    indexes = findall(x->x in ds, model.ds)
    if length(indexes) == 0
        error("No such time points $(ds).")
    end
    # Append the data.
    deleteat!(model.ds, indexes)
    deleteat!(model.y, indexes)
    # Convert to numeric.
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(model.ds))
    y_numeric = Transforms.apply(model.y_transform, model.y)
    # Prepare observations.
    observations = Gen.choicemap((:xs, y_numeric))
    !isnothing(model.config.noise) && (observations[:noise] = trace[:noise])
    # Run SMC step.
    Inference.smc_step!(model.pf_state, (ds_numeric, model.config), observations)
end

"""
    maybe_resample!(model::GPModel, ess_threshold::Real)
Resample the particle collection in `model` if ESS is below `ess_threshold`.
Setting `ess_threshold = AutoGP.num_particles(model) + 1` will ensure
that resampling always takes place, since the ESS is upper bounded by
the number of particles.
"""
function maybe_resample!(model::GPModel, ess_threshold::Real)
    return Gen.maybe_resample!(model.pf_state, ess_threshold=ess_threshold)
end

"""
    dist = predict_mvn(model::GPModel, ds::IndexType; noise_pred::Union{Nothing,Float64}=nothing)

Return an instance of [`Distributions.MixtureModel`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.MixtureModel)
representing the overall posterior predictive distribution for data at index points `ds`.
By default, the `noise_pred` of the new data is equal to the inferred `noise` of the observed
data within each particle in `model`; using `noise_pred=0.` returns the posterior
distribution over the noiseless function values.

The returned `dist` has precisely [`num_particles`](@ref)`(model)` components, each
of type [`Distributions.MvNormal`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormal),
with weights [`particle_weights`](@ref)`(model)`. These objects can be retrieved using
[`Distributions.components`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.components-Tuple{AbstractMixtureModel})
and
[`Distributions.probs`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.probs-Tuple{AbstractMixtureModel}), respectively.
"""
function predict_mvn(
        model::GPModel,
        ds::IndexType;
        noise_pred::Union{Nothing,Float64}=nothing)
    if !(eltype(ds) <: eltype(model.ds))
        error("Invalid time $(ds), expected $(eltype(model.ds))")
    end
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(ds))
    n_particles = num_particles(model)
    weights = particle_weights(model)
    distributions = Vector{MvNormal}(undef, n_particles)
    Threads.@threads for i=1:n_particles
        dist = Inference.predict_mvn(
                    model.pf_state.traces[i],
                    ds_numeric;
                    noise_pred=noise_pred)
        mu = Distributions.mean(dist)
        cov = Distributions.cov(dist)
        # Since model.y_transform is linear, data remains normal.
        # The same is not true for if model.y_transform is log, in which
        # case we would need to return Distributions.MvLogNormal.
        mu, cov = Transforms.unapply_mean_var(model.y_transform, mu, cov)
        distributions[i] = MvNormal(mu, cov)
    end
    return MixtureModel(distributions, weights)
end

"""
    (x::Vector, success::Bool) = predict_quantile(
        model::GPModel, ds::IndexType, q::Real;
        noise_pred::Union{Nothing,Float64}=nothing, tol=1e-5, max_iter=1e6)

Evaluates the inverse cumulative distribution function (CDF) of the
multivariate Gaussian mixture model returned by [`predict_mvn`](@ref) at
`q` (between 0 and 1, exclusive) separately for each dimension. The
returned vector `x` has the same length as the index points `ds`.

# Note

The inverse CDF is numerically estimated using a binary search algorithm.
The keyword arguments `tol` and `max_iter`  correspond to the desired
absolute tolerance of the estimate and the maximum number of binary search
iterations, respectively. The returned Boolean variable `success` indicates
whether the returned value `x` has been located to within the specified
error tolerance.

# See also

- [`predict_mvn`](@ref)
"""
function predict_quantile(
        model::GPModel,
        ds::IndexType,
        q::Real;
        noise_pred::Union{Nothing,Float64}=nothing,
        tol=1e-5,
        max_iter=1e6)
    (0 < q < 1) || error("Quantile must be in (0,1).")
    mvn = predict_mvn(model, ds; noise_pred=noise_pred)
    components = Distributions.components(mvn)
    means = hcat(Distributions.mean.(components)...)
    vars = hcat(Distributions.var.(components)...)
    mixtures = [Normal.(m, sqrt.(v)) for (m, v) in zip(eachrow(means), eachrow(vars))]
    weights = Distributions.probs(mvn)
    mixture = MixtureModel.(mixtures, Ref(weights))
    @assert length(mixture) == length(ds)
    @assert all(length.(mixtures) .== num_particles(model))
    x = zeros(length(mixture))
    iter = 0
    x_max = repeat([Inf], length(mixture))
    x_min = repeat([-Inf], length(mixture))
    success = false
    while iter < max_iter
        epsilon = @. Distributions.cdf(mixture, x) - q
        if all(abs.(epsilon) .< tol)
            success = true
            break
        end
        x_max = ifelse.(epsilon .> 0, x, x_max)
        x_min = ifelse.(epsilon .< 0, x, x_min)
        x_hi = min.(x_max, (@. 2^sign(x)*x + (x == 0)))
        x_lo = max.(x_min, (@. 2^-sign(x)*x - (x == 0)))
        x_hi_mid = Distributions.mean([x, x_hi])
        x_lo_mid = Distributions.mean([x, x_lo])
        x = ifelse.(
                abs.(epsilon) .< tol,
                x,
                ifelse.(
                    epsilon .< 0,
                    x_hi_mid,
                    x_lo_mid))
        iter += 1
    end
    return (x, success)
end

"""
    predictions = predict(
        model::GPModel,
        ds::IndexType;
        quantiles::Vector{Float64}=Float64[],
        noise_pred::Union{Nothing,Float64}=nothing)

Return predictions for new index point `ds`, and optionally quantiles
corresponding to the provided `quantiles` (numbers between 0 and 1, inclusive).
By default, the `noise_pred` of the new data is equal to the inferred `noise` of the observed
data within each particle in `model`; using `noise_pred=0.` returns the posterior
distribution over the noiseless function values.

The returned [`DataFrames.DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame)
has columns `["ds", "particle", "weight", "y_mean"]`, as well as any
additional columns for the requested `quantiles`.

# Example
```
julia> ds = [Dates.Date(2020,1,1), Dates.Date(2020,1,2)] # Dates to query
julia> GPModel.predict(model, ds; quantiles=[.025, 0.975])
16×6 DataFrame
 Row │ ds          particle  weight       y_0.025    y_0.975    y_mean
     │ Date        Int64     Float64      Float64    Float64    Float64
─────┼────────────────────────────────────────────────────────────────────
   1 │ 2020-01-01         1  4.97761e-22  -13510.0   14070.6      280.299
   2 │ 2020-01-02         1  4.97761e-22  -13511.0   14071.6      280.299
   3 │ 2020-01-01         2  0.279887       4504.73   8211.43    6358.08
   4 │ 2020-01-02         2  0.279887       4448.06   8154.3     6301.18
   5 │ 2020-01-01         3  0.0748059    -43638.6   65083.0    10722.2
   6 │ 2020-01-02         3  0.0748059    -43662.0   65074.7    10706.4
   7 │ 2020-01-01         4  0.60809      -17582.2   30762.4     6590.06
   8 │ 2020-01-02         4  0.60809      -17588.0   30771.5     6591.78
```
"""
function predict(
        model::GPModel,
        ds::IndexType;
        quantiles::Vector{Float64}=Float64[],
        noise_pred::Union{Nothing,Float64}=nothing)
    if !(eltype(ds) <: eltype(model.ds))
        error("Invalid time $(ds), expected $(eltype(model.ds))")
    end
    ds_numeric = Transforms.apply(model.ds_transform, to_numeric.(ds))
    weights = particle_weights(model)
    n_particles = num_particles(model)
    frames = Vector(undef, n_particles)
    Threads.@threads for i=1:n_particles
        y_mean, y_bounds = Inference.predict(
            model.pf_state.traces[i], ds_numeric;
            quantiles=quantiles, noise_pred=noise_pred)
        records = Dict(
                "ds"        => ds,
                "y_mean"    => Transforms.unapply(model.y_transform, y_mean),
                "particle"  => repeat([i], length(ds)),
                "weight"    => repeat([weights[i]], length(ds)))
        for (j, q) in enumerate(quantiles)
            key = "y_$(q)"
            records[key] = Transforms.unapply(model.y_transform, y_bounds[:,j])
        end
        frames[i] = DataFrames.DataFrame(records)
    end
    frame = vcat(frames...)
end

"""
    function predict_proba(model::GPModel, ds::IndexType, y::Vector{<:Real})

Compute predictive probability of data `y` at time points `ds` under `model`.

# Example
```
julia> ds = [Dates.Date(2020,1,1), Dates.Date(2020,1,2)] # Dates to query
julia> y = [0.1, .0.5] # values to query
julia> GPModel.predict(model, ds, y)
7×3 DataFrame
 Row │ particle  weight       logp
     │ Int64     Float64      Float64
─────┼─────────────────────────────────
   1 │        1  0.0287155    -64.2388
   2 │        2  0.0437349    -59.7672
   3 │        3  0.576247     -62.6499
   4 │        4  0.00164846   -59.5311
   5 │        5  0.215255     -61.066
   6 │        6  0.134198     -64.4041
   7 │        7  0.000201078  -68.462
```
"""
function predict_proba(model::GPModel, ds::IndexType, y::Vector{<:Real})
    dist = predict_mvn(model, ds)
    components = Distributions.components(dist)
    probs = Distributions.probs(dist)
    n_particles = num_particles(model)
    logps = @match y begin
        []  => repeat([0.], n_particles)
        _   => Distributions.logpdf.(components, [y])
    end
    return DataFrames.DataFrame(
        "particle" => 1:n_particles,
        "weight"   => probs,
        "logp"     => logps)
end

# Hack for tutorials.ipynb -> tutorials.md conversion.
function Base.show(df::DataFrames.DataFrame)
    return Base.show(df::DataFrames.DataFrame;
        summary=false, header_crayon=DataFrames.PrettyTables.Crayons.Crayon(),
        eltypes=false, rowlabel=Symbol())
end

"""
    function decompose(model::GPModel)

Decompose each particle within `model` into its constituent kernels.
Supposing that [`num_particles`](@ref)`(model)` equals ``k``, the return
value `models::Vector{GPModel}` of `decompose` is a length-``k`` vector of
[`GPModel`](@ref) instances.

Therefore, `models[i]` is a [`GPModel`](@ref) that represents the
decomposition of particle `i` in `model` into its constituent kernels. Each
particle in `models[i]` corresponds to a kernel fragment in the covariance
for particle `i` of `model` (i.e., one particle for each [`GP.Node`](@ref)
in the covariance kernel).

The weights of `models[i]` are arbitrary and have no meaningful value.

This function is particularly useful for visualizing the individual time
series structures that make up each particle of `model`.
"""
function decompose(model::GPModel)
    kernels = covariance_kernels(model)
    unrolled = map(GP.unroll, kernels)
    @assert length(kernels) == num_particles(model)
    models = Vector{GPModel}(undef, length(kernels))
    # ERROR: type GPConfig has no field WhiteNoise
    # noises = Model.transform_param.(
    #         :noise,
    #         [trace[:noise] for trace in model.pf_state.traces],)
    #     .+ AutoGP.Model.JITTER
    for (i, kernel_list::Vector{GP.Node}) in enumerate(unrolled)
        # ERROR: type GPConfig has no field WhiteNoise
        # Add observation noise as a WhiteNoise kernel.
        # typeof(kernel_list)
        # push!(kernel_list, GP.WhiteNoise(noises[i]))
        # Initialize new GPModel.
        models[i] = GPModel(
                model.ds, model.y;
                n_particles=length(kernel_list), config=model.config)
        # -- Copy transforms, since add_data! may have been called on model.
        models[i].ds_transform = model.ds_transform
        models[i].y_transform = model.y_transform
        # Force update each particle to match the kernel fragment.
        for (j, trace) in enumerate(models[i].pf_state.traces)
            models[i].pf_state.traces[j] = Inference.node_to_trace(
                kernel_list[j], model.pf_state.traces[i])
        end
        # Weights are arbitrary.
        models[i].pf_state.log_weights = zeros(length(models[i].pf_state.traces))
    end
    return models
end
