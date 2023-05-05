# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Julia 1.8.2
#     language: julia
#     name: julia-1.8
# ---

# <!--
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -->

# # Overview

# This tutorial demonstrates the basic capabilities of the [`AutoGP`](@ref) package.

import AutoGP

# +
import CSV
import Dates
import DataFrames

using PyPlot: plt
# -

# ## Loading Data

# The first step is to load a dataset from disk. The [`tsdl.161.csv`](assets/tsdl.161.csv) file, obtained from the [Time Series Data Library](https://pkg.yangzhuoranyang.com/tsdl/), has two columns:
#
# - `ds` indicates time stamps.
# - `y` indicates measured time series values.
#
# In the call to `CSV.File`, we explicitly set the type of the `ds` column to `Dates.Date`, permitted types for time indexes are types `T <: Real` and `T < :Dates.TimeType`, see [`AutoGP.IndexType`](@ref).

data = CSV.File("assets/tsdl.161.csv"; header=[:ds, :y], types=Dict(:ds=>Dates.Date, :y=>Float64));
df = DataFrames.DataFrame(data)
show(df)

# We next split the data into a training set and test set.

# +
n_test = 18
n_train = DataFrames.nrow(df) - n_test
df_train = df[1:end-n_test, :]
df_test = df[end-n_test+1:end, :]

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")
ax.legend();
# -

# ## Creating an AutoGP Model

# Julia natively supports multiprocessing, which greatly improves performance for embarrassingly parallel computations in `AutoGP`. The number of threads available to Julia can be set using the `JULIA_NUM_THREADS=[nthreads]` environment variable or invoking `julia -t [nthreads]` from the command line.

Threads.nthreads()

# We next initialize a [`AutoGP.GPModel`](@ref), which will enable us to automatically discover an ensemble of [Gaussian process covariance kernels](@ref gp_cov_kernel) for modeling the time series data. Initially, the model structures and parameters are sampled from the prior.  The `n_particles` argument is optional and specifices the number of particles for [sequential Monte Carlo inference](#Fitting-the-Model-using-Sequential-Monte-Carlo).

model = AutoGP.GPModel(df_train.ds, df_train.y; n_particles=6);

# ## Generating Prior Forecasts

# Calling [`AutoGP.covariance_kernels`](@ref) returns the ensemble of covariance kernel structures and parameters, whose weights are given by [`AutoGP.particle_weights`](@ref). These model structures have not yet been fitted to the data, so we are essentially importance sampling the posterior over structures and parameters given data by using the prior distribution as the proposal.

weights = AutoGP.particle_weights(model)
kernels = AutoGP.covariance_kernels(model)
for (i, (k, w)) in enumerate(zip(kernels, weights))
    println("Model $(i), Weight $(w)")
    Base.display(k)
end

# Forecasts are obtained using [`AutoGP.predict`](@ref), which takes in a `model`, a list of time points `ds` (which we specify to be the observed time points, the test time points, and 36 months of future time points). We also specify a list of `quantiles` for obtaining prediction intervals. The return value is a [`DataFrames.DataFrame`](https://dataframes.juliadata.org/stable/lib/types/#DataFrames.DataFrame) object that show the particle id, particle weight, and predictions from each of the particles in `model`.

ds_future = range(start=df.ds[end]+Dates.Month(1), step=Dates.Month(1), length=36)
ds_query = vcat(df_train.ds, df_test.ds, ds_future)
forecasts = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975])
show(forecasts)

# Let us visualize the forecasts before model fitting. The model clearly underfits the data.

# +
fig, ax = plt.subplots(figsize=(10,5))

ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")

for i=1:AutoGP.num_particles(model)
    subdf = forecasts[forecasts.particle.==i,:]
    ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.5)
    ax.fill_between(
        subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        color="tab:blue", alpha=0.05)
end
# -

# ## Model Fitting via SMC

# The next step is to fit the model to the observed data. There are [three fitting algorithms](@ref end_to_end_model_fitting) available.  We will use [`AutoGP.fit_smc!`](@ref) which leverages sequential Monte Carlo structure learning to infer the covariance kernel structures and parameters.
#
# The annealing schedule below adds roughly 10% of the observed data at each step, with 100 MCMC rejuvenation steps over the structure and 10 Hamiltonian Monte Carlo steps for the parameters. Using `verbose=true` will print some statistics about the acceptance rates of difference MCMC and HMC moves that are performed within the SMC learning algorithm.

AutoGP.seed!(6)
AutoGP.fit_smc!(model; schedule=AutoGP.Schedule.linear_schedule(n_train, .10), n_mcmc=75, n_hmc=10, verbose=true);

# ## Generating Posterior Forecasts

# Having the fit data, we can now inspect the ensemble of posterior structures, parameters, and predictions.

ds_future = range(start=df_test.ds[end]+Dates.Month(1), step=Dates.Month(1), length=36)
ds_query = vcat(df_train.ds, df_test.ds, ds_future)
forecasts = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975]);
show(forecasts)

# The plot below reflects posterior uncertainty as to whether the linear componenet will persist or the data will revert to the mean.

# +
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")

for i=1:AutoGP.num_particles(model)
    subdf = forecasts[forecasts.particle.==i,:]
    ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.5)
    ax.fill_between(
        subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        color="tab:blue", alpha=0.05)
end
# -

# We can also inspect the discovered kernel structures and their weights.

weights = AutoGP.particle_weights(model)
kernels = AutoGP.covariance_kernels(model)
for (i, (k, w)) in enumerate(zip(kernels, weights))
    println("Model $(i), Weight $(w)")
    display(k)
end

# ## Computing Predictive Probabilities

# In addition to generating forecasts, the predictive probability of new data can be computed by using [`AutoGP.predict_proba`](@ref). The table below shows that the particles in our collection are able to predict the future data with varying accuracy, illustrating the benefits of maintaining an ensemble of learned structures.

logps = AutoGP.predict_proba(model, df_test.ds, df_test.y);
show(logps)

# It is also possible to directly access the underlying predictive distribution of new data at arbitrary time series values by using [`AutoGP.predict_mvn`](@ref), which returns an instance of [`Distributions.MixtureModel`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.MixtureModel). The [`Distributions.MvNormal`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormal) object corresponding to each of the 7 particles in the mixture can be extracted using [`Distributions.components`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.components-Tuple{AbstractMixtureModel}) and the weights extracted using [`Distributions.probs`](https://juliastats.org/Distributions.jl/stable/mixture/#Distributions.probs-Tuple{AbstractMixtureModel}).
#
# Each `MvNormal` in the mixture has 18 dimensions corresponding to the lenght of `df_test.ds`.

mvn = AutoGP.predict_mvn(model, df_test.ds)

# Operations from the [`Distributions`](https://juliastats.org/Distributions.jl) package can now be applied to the `mvn` object.

# ## Incorporating New Data

# Online learning is supported by using [`AutoGP.add_data!`](@ref), which lets us incorporate a new batch of observations.  Each particle's weight will be updated based on how well it predicts the new data (technically, the predictive probability it assigns to the new observations). Before adding new data, let us first look at the current particle weights.

AutoGP.particle_weights(model)

# Here are the forecasts and predictive probabilities of the test data under each particle in `model`.

using Printf: @sprintf

# +
forecasts = AutoGP.predict(model, df_test.ds; quantiles=[0.025, 0.975])

fig, axes = plt.subplots(ncols=6)
for i=1:AutoGP.num_particles(model)
    axes[i].scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")
    subdf = forecasts[forecasts.particle.==i,:]
    axes[i].plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.5)
    axes[i].fill_between(
        subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        color="tab:blue", alpha=0.1)
    axes[i].set_title("$(i), logp $(@sprintf "%1.2f" logps[i,:logp])")
    axes[i].set_yticks([])
    axes[i].set_ylim([0.8*minimum(df_test.y), 1.1*maximum(df_test.y)])
    axes[i].tick_params(axis="x", labelrotation=90)
fig.set_size_inches(12, 5)
fig.set_tight_layout(true)
end
# -

# Now let's incorporate the data and see what happens to the particle weights.

AutoGP.add_data!(model, df_test.ds, df_test.y)
AutoGP.particle_weights(model)

# The particle weights have changed to reflect the fact that some particles are able to predict the new data better than others, which indicates they are able to better capture the underlying data generating process.
# The particles can be resampled using [`AutoGP.maybe_resample!`](@ref), we will use an effective sample size of `num_particles(model)/2` as the resampling criterion.

AutoGP.maybe_resample!(model, AutoGP.num_particles(model)/2)

# Because the resampling critereon was met, the particles were resampled and now have equal weights.

AutoGP.particle_weights(model)

# The estimate of the marginal likelihood can be computed using [`AutoGP.log_marginal_likelihood_estimate`](@ref).

AutoGP.log_marginal_likelihood_estimate(model)

# Since we have added new data, we can update the particle structures and parameters by using [`AutoGP.mcmc_structure!`](@ref). Note that this "particle rejuvenation" operation does not impact the weights. 

AutoGP.mcmc_structure!(model, 100, 10; verbose=true)

# Let's generate and plot the forecasts over the 36 month period again now that we have observed all the data. The prediction intervals are markedly narrower and it is more likely that the linear trend will persist rather than revert.

# +
forecasts = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975]);

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df.ds, df.y, marker=".", color="k", label="Observed Data")
for i=1:AutoGP.num_particles(model)
    subdf = forecasts[forecasts.particle.==i,:]
    ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.5)
    ax.fill_between(
        subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        color="tab:blue", alpha=0.05)
end
