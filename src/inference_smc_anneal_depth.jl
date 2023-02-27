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

function smc_step(
            model_trace::Gen.Trace,
            model_args::Tuple,
            argdiffs::Tuple,
            proposal_fwd::Gen.GenerativeFunction,
            proposal_fwd_args::Tuple,
            proposal_bwd::Gen.GenerativeFunction,
            proposal_bwd_args::Tuple,
            transform_fwd::Function,
            transform_bwd::Function;
            check::Bool=false,
            observations::Gen.ChoiceMap=EmptyChoicemap())

    # Simulate forward proposal.
    proposal_fwd_trace = Gen.simulate(proposal_fwd, (model_trace, proposal_fwd_args...))
    proposal_fwd_retval = get_retval(proposal_fwd_trace)
    proposal_fwd_choices = get_choices(proposal_fwd_trace)
    proposal_fwd_weight = get_score(proposal_fwd_trace)

    # Obtain  new model choices and corresponding backward proposal choices.
    model_choices_new, proposal_bwd_choices =
        transform_fwd(
            model_trace,
            proposal_fwd_choices,
            proposal_fwd_retval,
            proposal_fwd_args)

    # Generate new model trace and incremental weight.
    (model_trace_new, model_weight_diff, retdiff, discard) = Gen.update(
        model_trace,
        model_args,
        argdiffs,
        model_choices_new)

    # Compute backward proposal weight.
    proposal_bwd_weight, proposal_bwd_retval = Gen.assess(
        proposal_bwd,
        (model_trace_new, proposal_bwd_args...),
        proposal_bwd_choices)
    proposal_weight_diff = proposal_bwd_weight - proposal_fwd_weight

    # Dynamic checks.
    if check
        # Observations unchanged.
        Gen.check_observations(get_choices(model_trace_new), observations)

        # Obtain the round trip model choices and forward choices.
        model_choices_rt, proposal_fwd_choices_rt =
            transform_bwd(
                model_trace_new,
                proposal_bwd_choices,
                proposal_bwd_retval,
                proposal_bwd_args)

        # Generate round-trip model trace.
        (model_trace_rt, _) = Gen.update(
            model_trace_new,
            get_args(model_trace),
            argdiffs,
            model_choices_rt)

        # Generate round-trip proposal trace.
        proposal_fwd_trace_rt, w = Gen.generate(
            proposal_fwd,
            (model_trace_rt, proposal_fwd_args...),
            proposal_fwd_choices_rt)
        @assert isapprox(get_score(proposal_fwd_trace_rt), w)

        Gen.check_round_trip(model_trace, model_trace_rt)
        Gen.check_round_trip(proposal_fwd_trace, proposal_fwd_trace_rt)
    end
    # Compute the incremental weight.
    weight_diff = model_weight_diff + proposal_weight_diff
    return model_trace_new, weight_diff
end

function smc_step!(
            state::Gen.ParticleFilterState{U},
            model_args::Tuple,
            argdiffs::Tuple,
            proposal_fwd::Gen.GenerativeFunction,
            proposal_fwd_args::Tuple,
            proposal_bwd::Gen.GenerativeFunction,
            proposal_bwd_args::Tuple,
            transform_fwd::Function,
            transform_bwd::Function;
            check::Bool=false,
            observations::Gen.ChoiceMap=EmptyChoicemap()) where {U}

    num_particles = length(state.traces)
    Threads.@threads for i=1:num_particles
        (state.new_traces[i], weight_diff) = smc_step(
                state.traces[i],
                model_args,
                argdiffs,
                proposal_fwd,
                proposal_fwd_args,
                proposal_bwd,
                proposal_bwd_args,
                transform_fwd,
                transform_bwd;
                check=check,
                observations=observations)
        state.log_weights[i] += weight_diff

    end

    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

function run_smc_anneal_depth(
        ts::Vector{Float64},
        xs::Vector{Float64},
        schedule::Vector{<:Integer};
        n_train::Integer=length(ts),
        biased::Bool=false,
        changepoints::Bool=false,
        noise::Union{Nothing,Float64}=nothing,
        n_particles::Int=50,
        n_mcmc::Int=10,
        n_hmc::Int=10,
        adaptive_resampling::Bool=true,
        adaptive_rejuvenation::Bool=false,
        verbose::Bool=false,
        log_dir::Union{Nothing,String}=true,
        check::Bool=false)

    # Confirm valid schedule.
    @assert @. all((schedule[2:end] - schedule[1:end-1]) > 0)

    # Prepare data.
    ts_train = ts[1:n_train]
    xs_train = xs[1:n_train]
    ts_test = ts[n_train+1:end]
    xs_test = xs[n_train+1:end]

    # Prepare constraints.
    observations = Gen.choicemap()
    observations[:xs] = xs_train
    if !isnothing(noise)
        observations[:noise] = Model.untransform_param(:noise, noise)
    end

    # Initialize state.
    state = nothing

    # Run inference.
    for depth in schedule
        verbose && println("Running SMC round $(depth)/$(schedule[end])")
        config = GPConfig(max_depth=depth, changepoints=changepoints)

        # Run SMC step.
        if depth == schedule[1]
            # Initialize step.
            state = Gen.initialize_particle_filter(
                        model,
                        (ts_train, config),
                        observations,
                        n_particles)
        else
            # Increment depth step.
            smc_step!(
                    state,
                    (ts_train, config),
                    (NoChange(), UnknownChange()),
                    detach_attach_proposal__propose_attach,
                    (biased, false, depth),
                    detach_attach_proposal__propose_detach,
                    (biased, false),
                    detach_attach_involution__apply_attach,
                    detach_attach_involution__apply_detach;
                    check=check,
                    observations=observations)
        end

        # Debug.
        verbose && println(compute_particle_weights(state))

        # Resample step.
        resampled = false
        if depth < schedule[end]
            ess_threshold = adaptive_resampling ? n_particles/2 : n_particles
            resampled = Gen.maybe_resample!(state, ess_threshold=ess_threshold)
        end

        # Rejuvenate step.
        rejuvenated = false
        if !adaptive_rejuvenation || resampled
            rejuvenated = true
            Threads.@threads for i=1:n_particles
                local trace = state.traces[i]
                state.traces[i] = rejuvenate_particle(
                    trace, n_mcmc, n_hmc, biased;
                    verbose=verbose, check=check, observations=observations)
            end
        end

        if !isnothing(log_dir)
            ts_train_fut = Float64[]
            xs_train_fut = Float64[]
            save_smc_predictions(
                state, depth, resampled, rejuvenated,
                ts_train_fut, xs_train_fut,
                ts_test, xs_test, log_dir; verbose=verbose)
        end
    end

    # Return final traces.
    return state
end
