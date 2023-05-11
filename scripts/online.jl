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

import AutoGP
import CSV
import Dates
import DataFrames
import Distributions
import Gen
import Printf
import Random

using Match
using PyPlot: plt

plt.ioff()


function fn_callback(; ds_test, y_test, log_dir, kwargs...)
    model       = kwargs[:model]        # Inferred AutoGP.GPModel so far.
    ds_next     = kwargs[:ds_next]      # Future SMC data (time).
    y_next      = kwargs[:y_next]       # Future SMC data (observations).
    step        = kwargs[:step]         # Current SMC step
    rejuvenated = kwargs[:rejuvenated]  # Rejuvenated at current step?
    resampled   = kwargs[:resampled]    # Resampled at current step?
    elapsed     = kwargs[:elapsed]      # Total runtime elapsed.
    verbose     = kwargs[:verbose]      # Verbose setting?

    # Generate predictions on observed data, next data, and future data.
    ds_query = vcat(model.ds, ds_next, ds_test)
    y_true = vcat(model.y, y_next, y_test)
    predictions = AutoGP.predict(model, ds_query; quantiles=[0.025, 0.975])

    # Add column for the type of ds.
    predictions[!, :ds_type] .= ""
    predictions[predictions.ds .∈ [model.ds], :ds_type] .= "ds_obs"
    predictions[predictions.ds .∈ [ds_next], :ds_type] .= "ds_next"
    predictions[predictions.ds .∈ [ds_test], :ds_type] .= "ds_test"

    # Add column for elapsed and step.
    predictions[!, :elapsed] .= kwargs[:elapsed]
    predictions[!, :step] .= step
    predictions[!, :resampled] .= resampled
    predictions[!, :rejuvenated] .= rejuvenated

    # Compute log probability of test data.
    test_mask = (!isnan).(y_test)
    logps = AutoGP.predict_proba(model, ds_test[test_mask], y_test[test_mask])

    # Add particle-specific statistics.
    predictions[!, :logp_test] .= 0.
    predictions[!, :parent] .= 0
    for i=1:AutoGP.num_particles(model)
        mask = predictions.particle .== i
        predictions[mask, :logp_test] .= logps[logps.particle.==i, :logp]
        predictions[mask, :parent] .= model.pf_state.parents[i]
    end

    # Add y_true column.
    df_true = DataFrames.DataFrame("ds" => ds_query, "y_true" => y_true)
    DataFrames.leftjoin!(predictions, df_true, on=:ds)

    # Save predictions.
    fname = joinpath(log_dir, join(["gp", Printf.@sprintf("%03d", step)], "."))
    CSV.write(fname, predictions)
    println(fname)

    # Plot the observed data so far.
    idx_sort = sortperm(ds_query)
    weights = AutoGP.particle_weights(model)
    fig, ax = plt.subplots()
    ax.scatter(model.ds, model.y, marker="o", color="k", s=20, label="Observed Data")
    ax.scatter(ds_next, y_next, label="Training Data (Future)", marker="o", s=20, alpha=.1, color="gray")
    ax.scatter(ds_test, y_test, label="Test Data", marker="o", s=20, color="r")
    # Plot predictions on future data.
    for i=1:AutoGP.num_particles(model)
        subdf = predictions[predictions.particle.==i,:]
        ax.plot(
            subdf[idx_sort,:ds],
            subdf[idx_sort,:y_mean],
            linewidth=.5*weights[i],
            color="k")
        ax.fill_between(
            subdf[idx_sort,:ds],
            subdf[idx_sort,"y_0.025"],
            subdf[idx_sort,"y_0.975"],
            color="tab:green",
            alpha=.05)
    end
    ys = filter(!isnan, vcat(model.y, y_next, y_test))
    ymin = 0.5 * minimum(ys)
    ymax = 1.5 * maximum(ys)
    ax.set_ylim([ymin, ymax])
    ax.legend(loc="upper left")
    ax.set_title("Step $(step) Elapsed $(kwargs[:elapsed])")
    ax.legend(loc="upper right", framealpha=1)
    fig.set_tight_layout(true)
    fig.set_size_inches((10, 8))
    plotname = joinpath(log_dir, "overlay.$(Printf.@sprintf "%03.d" step).png")
    fig.savefig(plotname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    println(plotname)
end


function load_data(path::String, n_test::Integer=0, time_type::String="NUMERIC")
    time_type = @match time_type begin
        "NUMERIC"  => Float64
        "DATE"     => Dates.Date
        "DATETIME" => Dates.DateTime
        _          => error("Unknown TIME $(time)")
    end
    data = CSV.File(path; header=[:ds, :y], types=Dict(:ds=>time_type, :y=>Float64))
    n_train = length(data.ds) - n_test
    @assert 1 <= n_train <= length(data.ds)
    return (data.ds, data.y, n_train)
end


function extend_data(ds, n_future::Integer; freq=nothing)
    if isnothing(freq)
        freq_list = ds[2:end] - ds[1:end-1]
        freq = freq_list[end]
        all(freq .== freq_list) || error("Cannot determine freq; use FREQ=")
    end
    ds_future = range(start=ds[end]+freq, step=freq, length=n_future)
    y_future = fill(NaN, n_future)
    return (ds_future, y_future)
end


function plot_data(ds, y, n_train)
    fig, ax = plt.subplots()
    ax.scatter(ds[1:n_train], y[1:n_train], color="k")
    ax.scatter(ds[n_train+1:end], y[n_train+1:end], color="r")
    plt.show(block=true)
end


function make_log_dir(config, schedule)
    basename = splitpath(config.DATASET)[end]
    timestamp = Dates.format(Dates.now(), "YYYYmmdd-HHMMSS")
    log_dir = joinpath("logs", "$(timestamp).$(basename)")
    config_path = joinpath(log_dir, "config")
    mkdir(log_dir)
    open(config_path, "w") do f
        for key in keys(config)
            println(f, "$(key) $(config[key])")
        end
        println(f, "SCHEDULE $(schedule)")
    end
    config.VERBOSE && println(log_dir)
    return log_dir
end


config = (
    DATASET               = ENV["DATASET"],
    TIMETYPE              = get(ENV, "TIMETYPE", "DATE"),
    N_PARTICLES           = parse(Int, get(ENV, "N_PARTICLES", "6")),
    SEED                  = parse(Int, get(ENV, "SEED", "$(rand(1:10000))")),
    N_MCMC                = parse(Int, get(ENV, "N_MCMC", "200")),
    N_HMC                 = parse(Int, get(ENV, "N_HMC", "10")),
    SAVE                  = parse(Bool, get(ENV, "SAVE", "true")),
    ADAPTIVE_RESAMPLING   = parse(Bool, get(ENV, "ADAPTIVE_RESAMPLING", "false")),
    ADAPTIVE_REJUVENATION = parse(Bool, get(ENV, "ADAPTIVE_REJUVENATION", "false")),
    N_HMC_EXIT            = parse(Int, get(ENV, "N_HMC_EXIT", "10")),
    MAX_DEPTH             = parse(Int, get(ENV, "MAX_DEPTH", "-1")),
    NOISE                 = eval(Meta.parse(get(ENV, "NOISE", "nothing"))),
    SHUFFLE               = parse(Bool, get(ENV, "SHUFFLE", "false")),
    N_TEST                = parse(Int, get(ENV, "N_TEST", "0")),
    N_FUTURE              = parse(Int, get(ENV, "N_FUTURE", "100")),
    BIASED                = parse(Bool, get(ENV, "BIASED", "false")),
    CHANGEPOINTS          = parse(Bool, get(ENV, "CHANGEPOINTS", "false")),
    PLOT                  = parse(Bool, get(ENV, "PLOT", "false")),
    VERBOSE               = parse(Bool, get(ENV, "VERBOSE", "false")),
    CHECK                 = parse(Bool, get(ENV, "CHECK", "false")),
    FREQ                  = eval(Meta.parse(get(ENV, "FREQ", "nothing"))),
)

# Set seed.
AutoGP.seed!(config.SEED)

# Prepare data.
(ds, y, n_train) = load_data(config.DATASET, config.N_TEST, config.TIMETYPE)
config.PLOT && plot_data(ds, y, n_train)

# TODO: Implement a schedule based on Dates semantics.
schedule = Vector{Integer}(1:n_train)
config.VERBOSE && println(schedule)

# Prepare the data.
ds_train = ds[1:n_train]
y_train  = y[1:n_train]
ds_test  = ds[n_train+1:end]
y_test   = y[n_train+1:end]

# Create the GP model.
model = AutoGP.GPModel(
    ds_train, y_train;
    n_particles=config.N_PARTICLES,
    config=AutoGP.GP.GPConfig(
        changepoints=config.CHANGEPOINTS,
        noise=config.NOISE,
        max_depth=config.MAX_DEPTH))

# Prepare the callback.
callback_fn = @match config.SAVE begin
    false   => (; kwargs...) -> nothing
    true    => begin
        log_dir = make_log_dir(config, schedule)
        (ds_future, y_future) = extend_data(ds, config.N_FUTURE; freq=config.FREQ)
        AutoGP.Callbacks.make_smc_callback(
                fn_callback, model;
                ds_test=vcat(ds_test, ds_future),
                y_test=vcat(y_test, y_future),
                log_dir=log_dir)
    end
end

AutoGP.fit_smc!(
    model;
    schedule=schedule,
    n_mcmc=config.N_MCMC,
    n_hmc=config.N_HMC,
    biased=config.BIASED,
    shuffle=config.SHUFFLE,
    adaptive_resampling=config.ADAPTIVE_RESAMPLING,
    adaptive_rejuvenation=config.ADAPTIVE_REJUVENATION,
    hmc_config=(n_exit = config.N_HMC_EXIT,),
    verbose=config.VERBOSE,
    check=config.CHECK,
    callback_fn=callback_fn)

predictions = AutoGP.predict(model, ds, quantiles=[.025, 0.975])
display(predictions)
weights = AutoGP.particle_weights(model)
fig, ax = plt.subplots()
for i=1:AutoGP.num_particles(model)
    subdf = predictions[predictions.particle .== i, :]
    ax.scatter(ds_train, y_train, marker="o", color="black")
    ax.scatter(ds_test, y_test, marker="o", color="red")
    ax.plot(subdf[!,"ds"], subdf[!,"y_mean"], color="black", linewidth=10*weights[i])
    ax.fill_between(subdf[!,"ds"], subdf[!,"y_0.025"], subdf[!,"y_0.975"];
        alpha=.05, color="tab:blue")
end
plt.show(block=true)
