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

# # Insurance Claims

# This tutorial uses AutoGP to discover time series models of weekly insurance claims data.

# +
import AutoGP
import CSV
import Dates
import DataFrames

using PyPlot: plt
# -

# We first load the [`iclaims.csv`](assets/iclaims.csv) dataset from disk.  Since the data is positive we apply a log transform and perform all modeling in this transformed space.

data = CSV.File("assets/iclaims.csv"; header=[:ds, :y], types=Dict(:ds=>Dates.Date, :y=>Float64));
df = DataFrames.DataFrame(data)
df[:,"y"] = log.(df[:,"y"])
show(df)

# Let's hold out the final 100 weeks of observations to serve as test data.

# +
n_test = 100
n_train = DataFrames.nrow(df) - n_test
df_train = df[1:end-n_test, :]
df_test = df[end-n_test+1:end, :]

fig, ax = plt.subplots()
ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")
ax.legend()

fig.set_size_inches((20, 10))
# -

# The next step is to initialize an [`AutoGP.GPModel`](@ref) instance and fit the model using sequential Monte Carlo structure learning.

model = AutoGP.GPModel(df_train.ds, df_train.y; n_particles=8);

AutoGP.seed!(10)
schedule = AutoGP.Schedule.linear_schedule(n_train, .20)
AutoGP.fit_smc!(model; schedule=schedule, n_mcmc=50, n_hmc=10, shuffle=true, adaptive_resampling=false, verbose=true);

# Plotting the forecasts from each particle reflects the structural uncertainty.  7/8 particles have inferred a periodic component ([`AutoGP.GP.Periodic`](@ref)) with additive linear trend [`AutoGP.GP.Linear`](@ref). 1/8 of the particles has inferred a sum of a periodic kernel and gamma exponential ([`AutoGP.GP.GammaExponential`](@ref)) kernel, which is stationary but not "smooth" (formally, not mean-square differentiable).

# +
# Generate in-sample and future predictions.
ds_future = range(start=df_test.ds[end]+Dates.Week(1), step=Dates.Week(1), length=54*10)
ds_query = vcat(df_train.ds, df_test.ds, ds_future)
forecasts = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975]);
weights = AutoGP.particle_weights(model)

# Plot the data.
fig, ax = plt.subplots()
ax.scatter(df_train.ds, df_train.y, marker=".", color="k", label="Observed Data")
ax.scatter(df_test.ds, df_test.y, marker=".", color="r", label="Test Data")

# Plot the forecasts from each particle.
for i=1:AutoGP.num_particles(model)
    subdf = forecasts[forecasts.particle.==i,:]
    ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="k", linewidth=.1)
    ax.fill_between(
        subdf.ds, subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        color="tab:blue", alpha=0.025)
end

# Plot the grand mean.
mvn = AutoGP.predict_mvn(model, ds_query)
ax.plot(ds_query, AutoGP.Distributions.mean(mvn), color="k");

fig.set_size_inches((20, 10))
# -

for (w, k) in zip(AutoGP.particle_weights(model), AutoGP.covariance_kernels(model))
    println("Particle weight $(w)")
    display(k)
end

# !!! note
#
#     Mean forecasts, quantile forecasts, and probability densities values obtained via [`AutoGP.predict`](@ref) and [`AutoGP.predict_proba`](@ref) are all in the transformed (log space).  Only quantile forecasts can be transformed back to direct space via `exp`.  Converting mean forecasts and probability densities can be performed by using the [`Distributions.MvLogNormal`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvLogNormal) constructor, as demonstrated below.

import Distributions
log_mvn_components = [Distributions.MvLogNormal(d) for d in Distributions.components(mvn)]
log_mvn_weights = Distributions.probs(mvn)
log_mvn = Distributions.MixtureModel(log_mvn_components, log_mvn_weights);

fig, ax = plt.subplots()
ax.plot(ds_query, Distributions.mean(log_mvn), color="tab:blue", label="Correct Mean Forecasts in Direct Space")
ax.plot(ds_query, exp.(Distributions.mean(mvn)), color="k", label="Incorrect Mean Forecasts in Direct Space")
ax.axvline(df_test.ds[end], color="k", linestyle="--")
ax.legend()

# The difference between the blue and black curves is too small to observe on the scale above; let us plot the bias that arises from doing a naive transformation.

fig, ax = plt.subplots()
ax.plot(ds_query, Distributions.mean(log_mvn) - exp.(Distributions.mean(mvn)));
