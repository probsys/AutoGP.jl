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

# included by ./Inference.jl

import Distributions

using Printf: @sprintf

# Return normalized particle weights
function compute_particle_weights(state::Gen.ParticleFilterState)
    log_normalized_weights = Gen.normalize_weights(state.log_weights)[2]
    return exp.(log_normalized_weights)
end

# Return normalized particle weights
function effective_sample_size(state::Gen.ParticleFilterState)
    log_normalized_weights = Gen.normalize_weights(state.log_weights)[2]
    return Gen.effective_sample_size(log_normalized_weights)
end

function rejuvenate_particle_parameters(
        trace::Gen.Trace,
        n_hmc::Integer;
        hmc_config=Dict(),
        verbose::Bool=false,
        check::Bool=false,
        observations::ChoiceMap=EmptyChoicemap())
    # Obtain addresses of numeric parameters.
    choices = Gen.get_choices(trace)
    submap = Gen.get_submap(choices, :tree)
    leaf_addrs = Any[
        (:tree => address)
        for (address, value) in Gen.get_values_shallow(submap)
        if address[2] != :node_type
    ]
    # Infer noise?
    infer_noise = !Gen.has_value(observations, :noise)
    # HMC configuration.
    L_param   = get(hmc_config, :L_param, 10)
    eps_param = get(hmc_config, :eps_param, .02)
    L_noise   = get(hmc_config, :L_noise, 10)
    eps_noise = get(hmc_config, :eps_noise, .02)
    n_exit    = get(hmc_config, :n_exit, n_hmc)
    # HMC on numeric parameters.
    n_accept = 0
    n_reject = 0
    n_trial = 0
    @assert length(leaf_addrs) > 0
    selection = Gen.select(leaf_addrs...)
    for i=1:n_hmc
        trace, accepted = Gen.hmc(trace, selection;
            L=L_param, eps=eps_param, check=check, observations=observations)
        if infer_noise
            trace, = Gen.hmc(trace, Gen.select(:noise);
                L=L_noise, eps=eps_noise, check=check, observations=observations)
        end
        n_trial += 1
        n_accept += Integer(accepted)
        n_reject = accepted ? 0 : n_reject + 1
        (n_reject == n_exit) && break
    end
    verbose && println("accepted HMC[$(n_accept)/$(n_trial)]")
    return trace, n_accept, n_trial
end

function rejuvenate_particle_structure(
        trace::Gen.Trace,
        n_mcmc::Integer,
        n_hmc::Integer,
        biased::Bool;
        hmc_config=Dict(),
        verbose::Bool=false,
        check::Bool=false,
        observations::ChoiceMap=EmptyChoicemap())

    stats = Dict(:mh=>0, :hmc=>0, :hmc_trials=> 0)
    for iter=1:n_mcmc
        trace, accepted = Gen.metropolis_hastings(
            trace,
            tree_rejuvenation_proposal,
            (biased,),
            tree_rejuvenation_involution;
            check=check,
            observations=observations)
        stats[:mh] += Integer(accepted)

        if accepted
            trace, n_accept, trials = rejuvenate_particle_parameters(
                trace, n_hmc;
                verbose=false,
                check=check,
                observations=observations,
                hmc_config=hmc_config)
            stats[:hmc] += n_accept
            stats[:hmc_trials] += trials
        end
    end

    # Report statistics
    if verbose
        print("accepted MCMC[$(stats[:mh])/$(n_mcmc)]"
            * " HMC[$(stats[:hmc])/$(stats[:hmc_trials])]"
            * "\n")
    end

    return trace, stats
end

# Custom implementation of particle_filter_step! in Gen
# that removes the error on !isempty(discard)
# In this case, the sequence of targets is p_n(C) = P(C | x[1:n])
# When performing an "update the target" move
#   simulate C_n ~ \delta(C_{n-1})
#   incremental weight p_n(C) / p_{n-1}(C) = p(x[n] | C, x[1:n-1])
function smc_step!(
        state::Gen.ParticleFilterState{U},
        (ts, config)::Tuple,
        observations::Gen.ChoiceMap) where {U}
    argdiffs = (Gen.UnknownChange(), Gen.NoChange())
    num_particles = length(state.traces)
    Threads.@threads for i=1:num_particles
        (state.new_traces[i], weight_diff, _, discard) = Gen.update(
            state.traces[i], (ts, config), argdiffs, observations)
        state.log_weights[i] += weight_diff
    end
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

function run_smc_anneal_data(
        ts::Vector{Float64},
        xs::Vector{Float64};
        config::GPConfig=GPConfig(),
        biased::Bool=false,
        n_particles::Int=4,
        n_mcmc::Int=10,
        n_hmc::Int=10,
        hmc_config=Dict(),
        permutation::Vector{<:Integer}=collect(1:length(ts)),
        schedule::Vector{<:Integer}=range(1:length(ts)),
        adaptive_resampling::Bool=true,
        adaptive_rejuvenation::Bool=false,
        verbose::Bool=false,
        check::Bool=false,
        callback_fn::Function=(; kwargs...) -> nothing)

    # Initialize elapsed runtime.
    elapsed = 0.

    # Apply the permutation
    @assert sort(permutation) == 1:length(ts)
    ts = ts[permutation]
    xs = xs[permutation]

    # Determine inference schedule.
    @assert 1 <= schedule[1]
    @assert schedule[end] == length(ts)
    @assert all((schedule[2:end] .- schedule[1:end-1]) .> 0)

    # Initialize SMC particles from prior.
    @timeit elapsed begin
        observations = Gen.choicemap()
        if !isnothing(config.noise)
            observations[:noise] = Model.untransform_param(:noise, config.noise)
        end
        state = Gen.initialize_particle_filter(
                    model,
                    (Float64[], config),
                    observations,
                    n_particles)
    end

    # Invoke callback for step zero.
    callback_fn(
        state=state,
        ts=ts,
        xs=xs,
        permutation=permutation,
        schedule=schedule,
        step=0,
        elapsed=elapsed,
        rejuvenated=false,
        resampled=false,
        verbose=verbose)

    # Run inference.
    for step in schedule
        verbose && println("Running SMC round $(step)/$(schedule[end])")

        @timeit elapsed begin

            # Obtain dataset.
            ts_obs = ts[1:step]
            xs_obs = xs[1:step]
            observations[:xs] = xs_obs

            # Reweight step.
            smc_step!(state, (ts_obs, config), observations)

            # Report weights.
            verbose && begin
                w = compute_particle_weights(state)
                wstr = replace(repr(map((x->(@sprintf "%1.2e" x)), w)),"\""=>"")
                println("weights $(wstr)")
            end

            # Resample step.
            resampled = false
            if step < schedule[end]
                ess_threshold = adaptive_resampling ? n_particles/2 : n_particles
                resampled = Gen.maybe_resample!(state, ess_threshold=ess_threshold)
                verbose && println("resampled $(resampled)")
            end

            # Rejuvenate step.
            rejuvenated = false
            if !adaptive_rejuvenation || resampled
                rejuvenated = true
                Threads.@threads for i=1:n_particles
                    local trace = state.traces[i]
                    trace, = rejuvenate_particle_structure(
                        trace, n_mcmc, n_hmc, biased;
                        hmc_config=hmc_config, verbose=verbose, check=check,
                        observations=observations)
                    state.traces[i] = trace
                end
            end

        end

        # Invoke callback for current step.
        callback_fn(
            state=state,
            ts=ts,
            xs=xs,
            permutation=permutation,
            schedule=schedule,
            step=step,
            elapsed=elapsed,
            rejuvenated=rejuvenated,
            resampled=resampled,
            verbose=verbose)
    end

    # Return final traces.
    return state
end
