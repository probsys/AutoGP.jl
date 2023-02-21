import Distributions
import Gen

import AutoGP: Model
import AutoGP: Inference

using AutoGP.GP
using PyPlot: plt
using LinearAlgebra: diag

import Random

plt.ioff()

Gen.seed!(15)
Random.seed!(15)

"""Plot forecast of posterior predictive distribution."""
function plot_posterior_forecast(trace, ts_obs, ts_test, xs_obs, xs_test)
    node = trace[]
    noise = Model.transform_param(:noise, trace[:noise])
    dist = Distributions.MvNormal(node, noise + Model.JITTER, ts_obs, xs_obs, ts_test)
    mu = Distributions.mean(dist)
    bounds = Distributions.quantile(dist, [[.1, .9]])

    fig, ax = plt.subplots()
    ax.axvline(0, color="gray")
    ax.axhline(0, color="gray")
    ax.plot(ts_obs, xs_obs, color="k", label="Observed Data")
    ax.plot(ts_test, xs_test, color="b", label="Held-Out Data")
    ax.plot(ts_test, mu, color="red", label="Posterior Mean")
    ax.fill_between(ts_test, bounds[:,1], bounds[:,2], alpha=.1, color="red")
    ax.legend(loc="upper right", framealpha=1)
    return fig, ax
end

"""Compute unnormalized posterior for given address, fixing other constraints."""
function compute_unnormalized_posteriors(trace, address)
    constraints = Gen.choicemap(Gen.get_choices(trace))
    params = range(-10, 10, length=1000)
    scores = Vector{Float64}(undef, 1000)
    for (i, p) in enumerate(params)
        constraints[address] = p
        tr, w = Gen.generate(
            Gen.get_gen_fn(trace),
            Gen.get_args(trace),
            constraints)
        scores[i] = Gen.get_score(tr)
        @assert isapprox(w, scores[i])
    end
    return (params, scores)
end

"""Compute unnormalized posterior for all parameters, fixing other constraints."""
function compute_unnormalized_posteriors(trace)
    addresses = Inference.get_gp_parameter_addresses(trace)
    return Dict(a => compute_unnormalized_posteriors(trace, a) for a in addresses)
end

"""Plot unnormalized posterior for given parameter address."""
function plot_unnormalized_posterior(constraints, address, params, scores)
    fig, ax = plt.subplots()
    ax.set_title("Unnormalized Posterior $(address)")
    ax.axvline(constraints[address], color="red", label="True Value")
    ax.plot(params, scores)
    return fig, ax
end

"""Plot unnormalized posterior for all parameter addresses."""
function plot_unnormalized_posteriors(trace)
    posteriors = compute_unnormalized_posteriors(trace)
    constraints = Gen.get_choices(trace)
    results = Dict()
    for (address, (params, scores)) in posteriors
        fig, ax = plot_unnormalized_posterior(constraints, address, params, scores)
        results[address] = ax
    end
    return results
end

"""Test constrained generation of covariance tree."""
function test_constrain_structure(config)
    ts = Vector{Float64}(range(0, 3, length=300))
    for i=1:100
        trace, = Gen.generate(Model.model, (ts, config))
        choices = Gen.get_choices(trace)
        tree = Gen.get_submap(choices, :tree)
        constraints = Gen.choicemap()
        Gen.set_submap!(constraints, :tree, tree)
        trace_constrained, = Gen.generate(Model.model, (ts, config), constraints)
        @assert isapprox(trace_constrained[], trace[])
    end
end

"""Test that predictive likelihood agrees."""
function test_predictive_likelihood_agrees(config, constraints, ts_obs, xs_obs, ts_test, xs_test)
    # Compute using Bayes rule.
    ts = vcat(ts_obs, ts_test)
    xs = vcat(xs_obs, xs_test)
    constraints_joint = Gen.choicemap(constraints)
    constraints_joint[:xs] = xs
    trace_joint, weight_joint = Gen.generate(Model.model, (ts, config), constraints_joint)
    # Compute using custom inference.
    constraints_obs = Gen.choicemap(constraints)
    constraints_obs[:xs] = xs_obs
    trace_obs, weight_obs = Gen.generate(Model.model, (ts_obs, config), constraints_obs)
    # Ensure agreement.
    noise = Model.transform_param(:noise, trace_obs[:noise]) + Model.JITTER
    dist = Distributions.MvNormal(trace_obs[], noise, ts_obs, xs_obs, ts_test)
    lp_test_ll = Distributions.logpdf(dist, xs_test)
    lp_test_bayes = weight_joint - weight_obs
    println((lp_test_ll, lp_test_bayes))
    @assert isapprox(lp_test_ll, lp_test_bayes)
    @assert isapprox(trace_joint[], trace_obs[])
    # Return the simulated traces.
    return (trace_joint, trace_obs, lp_test_bayes)
end

"""Create ground-truth trace."""
function initialize_trace_true(ts, config, n_obs, node, xi)
    @assert 1 <= n_obs < length(ts)
    # Make constraints with true structure.
    constraints = Gen.choicemap()
    constraints[:noise] = xi
    Gen.set_submap!(constraints, :tree, Inference.node_to_choicemap(node, config))
    # Simulate the trace.
    trace, = Gen.generate(Model.model, (ts, config), constraints)
    @assert isapprox(trace[], node)
    @assert isapprox(trace[:noise], xi)
    xs = trace[:xs]
    ts_obs = ts[1:n_obs]
    xs_obs = xs[1:n_obs]
    ts_test = ts[n_obs+1:end]
    xs_test = xs[n_obs+1:end]
    # Return trace and test/train data.
    return (trace, ts_obs, xs_obs, ts_test, xs_test)
end

"""Create an initial trace for HMC."""
function initialize_trace_infer(trace_true, ts_obs, xs_obs)
    (_, config) = Gen.get_args(trace_true)
    choices = Gen.get_choices(trace_true)
    constraints_hmc = Gen.choicemap(choices)
    constraints_hmc[:xs] = xs_obs
    # Use Pathfinder U[-2,2] initialization for all numeric parameters.
    for addr in Inference.get_gp_parameter_addresses(trace_true)
        constraints_hmc[addr] = Gen.uniform(-2,2)
    end
    trace_infer, = Gen.generate(Model.model, (ts_obs, config), constraints_hmc)
    return trace_infer
end

"""Report metrics from the hmc trace."""
function compute_inference_metrics(trace_infer)
    noise = Model.transform_param(:noise, trace_infer[:noise])
    state = (GP.pretty(trace_infer[]), noise)
    score = Gen.get_score(trace_infer)
    dist = Distributions.MvNormal(trace_infer[], noise + Model.JITTER, ts_obs, xs_obs, ts_test)
    lp_test = Distributions.logpdf(dist, xs_test)
    return (state, score, lp_test)
end

# Set of benchmark kernels and noise.
BENCHMARKS = Dict(
    1 => (SquaredExponential(2), 0.01),
    2 => (Plus(Linear(.5), Periodic(2, 1)), 0.05),
    3 => (ChangePoint(Linear(.5), Linear(1.5), 1, .001), 0.001)
)

config = GP.GPConfig(changepoints=true)

# Run a quick test.
test_constrain_structure(config)

# Select benchmark.
(node_true, noise_true) = BENCHMARKS[2]
xi_true = Model.untransform_param(:noise, noise_true)
@assert isapprox(noise_true, Model.transform_param(:noise, xi_true))

# Simulate ground-truth trace.
(n, n_obs) = (1000, 200)
ts = Vector{Float64}(range(0, 10, length=n))
trace_true, ts_obs, xs_obs, ts_test, xs_test = initialize_trace_true(
    ts, config, n_obs, node_true, xi_true)
trace_joint, trace_obs, lp_test_true = test_predictive_likelihood_agrees(
    config, Gen.get_choices(trace_true), ts_obs, xs_obs, ts_test, xs_test)
println("True structure: $(GP.pretty(trace_true[])) $(noise_true)")

# Plot observed data, test data, and forecasts under true model.
fig, ax = plot_posterior_forecast(trace_obs, ts_obs, ts_test, xs_obs, xs_test)
ax.set_title("Predictive Distribution for True Trace")

# Plot unnormalized posterior surfaces for all parameters.
unnormalized_posterior_plots = plot_unnormalized_posteriors(trace_obs)

# Sample HMC trace.
trace_infer = initialize_trace_infer(trace_true, ts_obs, xs_obs)
test_predictive_likelihood_agrees(config, Gen.get_choices(trace_infer), ts_obs, xs_obs, ts_test, xs_test)

# Plot predictive from prior trace.
fig, ax = plot_posterior_forecast(trace_infer, ts_obs, ts_test, xs_obs, xs_test)
ax.set_title("Predictive Distribution for Prior Trace")

# Compute initial metrics.
leaf_addrs = Inference.get_gp_parameter_addresses(trace_infer)
(state_hmc, lp_test_hmc, score_hmc) = compute_inference_metrics(trace_infer)
println((0, state_hmc, score_hmc, lp_test_hmc, lp_test_true))
stats = [(state_hmc, score_hmc, lp_test_hmc)]

# Add initial point to the unnormalized posterior plots.
for address in leaf_addrs
    local ax = unnormalized_posterior_plots[address]
    ax.axvline(trace_infer[address], color="gray", label="Initial Value")
end

# Run HMC.
runtime = 0
n_steps = 100
for i in 1:100
    global trace_infer
    t = @timed trace_infer, accepted = Gen.hmc(trace_infer, Gen.select(leaf_addrs...))
    global runtime += t.time
    (state_i, score_i, lp_test_i) = compute_inference_metrics(trace_infer)
    println((i, accepted, state_i, score_i, lp_test_i, lp_test_true))
    push!(stats, (state_i, score_i, lp_test_i))
end

println("Completed $(n_steps) in $(runtime) secs (avg $(runtime/n_steps))")

println("TRUE TRACE")
display(Inference.get_gp_parameter_values(trace_true))

println("INFERRED TRACE")
display(Inference.get_gp_parameter_values(trace_infer))

# Add end points to the unnormalized posterior plots.
for address in leaf_addrs
    local ax = unnormalized_posterior_plots[address]
    ax.axvline(trace_infer[address], color="blue", label="End Value")
    ax.legend(loc="upper left")
end

# Plot the progress.
steps = Vector(1:length(stats))
scores = [s[2] for s in stats]
lps = [s[3] for s in stats]

fig, ax = plt.subplots(nrows=2)
ax[1].plot(steps, scores)
ax[1].set_yscale("symlog")
ax[1].set_ylabel("Unnormalized Posterior P(params, xi)")
ax[2].plot(steps, lps)
ax[2].set_ylabel("Predictive Likelihood of Held-Out Data")
ax[2].set_yscale("symlog")
ax[2].set_xlabel("HMC Steps")

fig, ax = plot_posterior_forecast(trace_infer, ts_obs, ts_test, xs_obs, xs_test)
ax.set_title("Predictive Distribution for Inferred Trace")

plt.show(block=true)
