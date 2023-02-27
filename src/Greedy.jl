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

module Greedy

import ..GP
import ..Inference
import ..Model

import Distributions
import Gen
import LinearAlgebra

using Match

function compute_data_likelihood(trace)
    model = Gen.get_gen_fn(trace)
    model_args = Gen.get_args(trace)
    model_choices = Gen.get_choices(trace)

    logp_joint = Gen.get_score(trace)

    choices_latent = Gen.choicemap()
    choices_latent[:noise] = model_choices[:noise]
    Gen.set_submap!(choices_latent, :tree, Gen.get_submap(model_choices, :tree))
    logp_latents = Gen.generate(model, model_args, choices_latent)[2]
    logp_data_est = logp_joint - logp_latents

    return logp_data_est
end

function compute_aic(trace; k=nothing)
    if isnothing(k)
        leaf_addrs = Inference.get_gp_parameter_addresses(trace)
        k = length(leaf_addrs)
    end
    logp_data = compute_data_likelihood(trace)
    return 2*(k - logp_data)
end

# ==============================================================================
# MCMC / GREEDY SEARCH HYBRID
# Algorithm 0
# ==============================================================================

const MAX_OPT=10

function greedy_mcmc_rejuvenate(
        trace::Gen.Trace,
        aic::Float64,
        biased::Bool;
        verbose::Bool=false,
        check::Bool=false,
        observations::Gen.ChoiceMap=Gen.EmptyChoiceMap())

    model = Gen.get_gen_fn(trace)
    model_args = Gen.get_args(trace)

    # if rand((1,2)) == 1
    fn_proposal = Inference.subtree_replace_proposal
    fn_involution = Inference.subtree_replace_involution
    proposal_args = (trace, biased,)
    # else
    #     fn_proposal = Inference.detach_attach_proposal
    #     fn_involution = Inference.detach_attach_involution
    #     proposal_args = (trace, biased, true)
    # end

    # Simulate proposal trace.
    proposal_fwd_trace = Gen.simulate(fn_proposal, proposal_args)
    proposal_retval = Gen.get_retval(proposal_fwd_trace)
    proposal_choices = Gen.get_choices(proposal_fwd_trace)
    proposal_fwd_weight = Gen.get_score(proposal_fwd_trace)

    # Obtain proposed trace.
    trace_prop, _ = fn_involution(trace, proposal_choices, proposal_retval, proposal_args)
    check && Gen.check_observations(Gen.get_choices(trace_prop), observations)

    # Optimize parameters of proposed trace.
    leaf_addrs = Inference.get_gp_parameter_addresses(trace_prop)
    selection = Gen.select(leaf_addrs...)
    prev_score = Gen.get_score(trace_prop)
    for i=1:MAX_OPT
        trace_prop = Gen.map_optimize(trace_prop, selection; verbose=verbose)
        new_score = Gen.get_score(trace_prop)
        @match (new_score == prev_score) begin
                true  => (break)
                false => (prev_score = new_score)
                end
    end
    check && Gen.check_observations(Gen.get_choices(trace_prop), observations)

    # Compute the new AIC.
    aic_prop = compute_aic(trace_prop; k=length(leaf_addrs))

    return @match (aic_prop < aic) begin
        true  => (trace_prop, aic_prop, true)
        false => (trace, aic, false)
        end
end

function greedy_mcmc_rejuvenate(
        n_parallel::Integer,
        trace::Gen.Trace,
        aic::Float64,
        biased::Bool;
        verbose::Bool=false,
        check::Bool=false,
        observations::Gen.ChoiceMap=Gen.EmptyChoiceMap())

    results = Vector{Any}(undef, n_parallel)
    for i=1:n_parallel
        x = greedy_mcmc_rejuvenate(trace, aic, biased;
            verbose=verbose, check=check, observations=observations)
        results[i] = x
    end
    (new_trace, new_aic, accepted) = argmin(x -> x[2], results)
    return (new_trace, new_aic, accepted)
end


# ==============================================================================
# Algorithm 2 from CKS
# https://arxiv.org/pdf/1706.02524.pdf
# ==============================================================================

# ===============
# MOVE TYPE UTILS
# ===============

function get_node_dist(config::GP.GPConfig)
    return @match config.changepoints begin
        true => config.node_dist_cp
        false => config.node_dist_nocp
    end
end

function get_leaf_node_types(config::GP.GPConfig)
    return findall(!iszero, config.node_dist_leaf .> 0)
end

function get_op_node_types(config::GP.GPConfig)
    node_dist = get_node_dist(config)
    n_leaf = length(config.node_dist_leaf)
    n_oper = length(node_dist) - n_leaf
    mask = vcat(zeros(n_leaf), ones(n_oper))
    p_oper = mask .* node_dist
    return findall(!iszero, p_oper .> 0)
end

get_leaf_node_types(trace::Gen.Trace) = get_leaf_node_types(Gen.get_args(trace)[2])
get_op_node_types(trace::Gen.Trace) = get_op_node_types(Gen.get_args(trace)[2])

function is_leaf_node_type(config::GP.GPConfig, node_type::Integer)
    return 1 <= node_type <= length(config.node_dist_leaf)
end

function is_op_node_type(config::GP.GPConfig, node_type::Integer)
    node_dist = config.changepoints ? config.node_dist_cp : config.node_dist_nocp
    return length(config.node_dist_leaf) < node_type <= length(node_dist)
end

function filter_node_type_indexes(trace::Gen.Trace, f::Function)
    choices = Gen.get_choices(trace)
    submap = Gen.get_submap(choices, :tree)
    return [
        address[1] for (address, value) in Gen.get_values_shallow(submap)
        if (address[2] == :node_type) && f(value)
    ]
end

"""Return list of numeric indexes of leaf nodes in the :tree."""
function get_leaf_node_indexes(trace::Gen.Trace)
    config = Gen.get_args(trace)[2]
    f = (value) -> is_leaf_node_type(config, value)
    return filter_node_type_indexes(trace, f)
end

"""Return list of numeric indexes of op nodes in the :tree."""
function get_op_node_indexes(trace::Gen.Trace)
    config = Gen.get_args(trace)[2]
    f = (value) -> is_op_node_type(config, value)
    return filter_node_type_indexes(trace, f)
end

function get_all_node_indexes(trace::Gen.Trace)
    config = Gen.get_args(trace)[2]
    f = (value) -> true
    return filter_node_type_indexes(trace, f)
end

"""Make a choice map for a base kernel with the given node_type."""
function make_base_kernel_choicemap(
        node_idx::Integer, node_type::Integer, config::GP.GPConfig)
    @assert is_leaf_node_type(config, node_type)
    @assert config.node_dist_leaf[node_type] > 0
    choices = Gen.choicemap()
    choices[(node_idx, :node_type)] = node_type
    NodeType = config.index_to_node[node_type]
    params = []
    dist = Distributions.Normal(0,1)
    for field in fieldnames(NodeType)
        log_param = rand(dist)
        choices[(node_idx, field)] = log_param
    end
    return choices
end

# ===========================================
# MOVE TYPE 1, REPLACE LEAF WITH ANOTHER LEAF
# ===========================================

"""Replace leaf node at node_idx with base kernel of node_type."""
function get_choices_replace_leaf(trace::Gen.Trace, node_idx::Integer, node_type::Integer)
    config = Gen.get_args(trace)[2]
    choices = Gen.get_choices(trace)
    tree = Gen.get_submap(choices, :tree)
    leaf_addr = :tree => (node_idx, :node_type)
    @assert Gen.has_value(choices, leaf_addr)
    @assert is_leaf_node_type(config, node_type)
    current_node_type = choices[leaf_addr]
    @assert current_node_type != node_type
    # Make the new tree.
    new_tree = Gen.choicemap()
    for (address, value) in Gen.get_values_shallow(tree)
        if address[1] != node_idx
            new_tree[address] = value
        end
    end
    # Add the base kernel.
    new_base_kernel = make_base_kernel_choicemap(node_idx, node_type, config)
    for (address, value) in Gen.get_values_shallow(new_base_kernel)
        @assert !Gen.has_value(new_tree, address)
        new_tree[address] = value
    end
    # Make overall new choices.
    new_choices = Gen.choicemap(choices)
    Gen.set_submap!(new_choices, :tree, new_tree)
    return new_choices
end

"""Replace leaf node at node_idx with all possible base kernels."""
function get_choices_replace_leaf(trace::Gen.Trace, node_idx::Integer)
    config = Gen.get_args(trace)[2]
    leaf_type = trace[:tree => (node_idx, :node_type)]
    new_choices_list = Gen.ChoiceMap[]
    leaf_node_types = get_leaf_node_types(trace)
    for node_type in leaf_node_types
        @assert config.node_dist_leaf[node_type] > 0
        if node_type != leaf_type
            new_choices = get_choices_replace_leaf(trace, node_idx, node_type)
            push!(new_choices_list, new_choices)
        end
    end
    return new_choices_list
end

# Replace all leaf nodes in the tree with all possible base kernels."""
function get_choices_replace_leaf(trace::Gen.Trace)
    leaf_node_indexes = get_leaf_node_indexes(trace)
    new_choices = [get_choices_replace_leaf(trace, n) for n in leaf_node_indexes]
    return vcat(new_choices...)
end

# ======================================
# MOVE TYPE 2, INSERT OPEATOR AT SUBTREE
# ======================================

# Inset op_type at subtree_idx with base kernel of leaf_type.
function get_choices_insert_oper_at_subtree(trace::Gen.Trace,
        subtree_idx::Integer, op_type::Integer, leaf_type::Integer)
    config = Gen.get_args(trace)[2]
    @assert !config.changepoints
    @assert is_op_node_type(config, op_type)
    @assert config.node_dist_nocp[op_type] > 0
    subtree_addr = :tree => (subtree_idx, :node_type)
    # Obtain current choices and tree.
    choices = Gen.get_choices(trace)
    @assert Gen.has_value(choices, subtree_addr)
    tree = Gen.get_submap(choices, :tree)
    idx_left = Gen.get_child(subtree_idx, 1, config.max_branch)
    idx_right = Gen.get_child(subtree_idx, 2, config.max_branch)
    # The subtree may be a leaf.
    # @assert Gen.has_value(choices, :tree => (idx_left, :node_type))
    # @assert Gen.has_value(choices, :tree => (idx_right, :node_type))
    # Reindex the current subtree to start at idx_left.
    new_subtree = Inference.extract_subtree_choices(
        tree, subtree_idx, config; idx_subtree_new=idx_left)
    # Obtain new base kernel and put into index 3.
    new_base_kernel = make_base_kernel_choicemap(idx_right, leaf_type, config)
    for (address, value) in Gen.get_values_shallow(new_base_kernel)
        @assert !Gen.has_value(new_subtree, address)
        new_subtree[address] = value
    end
    # Add operator at subtree index.
    @assert !Gen.has_value(new_subtree, (subtree_idx, :node_type))
    new_subtree[(subtree_idx, :node_type)] = op_type
    # New overall tree.
    new_tree = Gen.choicemap()
    for (address, value) in Gen.get_values_shallow(new_subtree)
        new_tree[address] = value
    end
    for (address, value) in Gen.get_values_shallow(tree)
        node_idx = address[1]
        if !Gen.has_value(new_subtree, (node_idx, :node_type))
            new_tree[address] = value
        end
    end
    # Make overall new choices.
    new_choices = Gen.choicemap(choices)
    Gen.set_submap!(new_choices, :tree, new_tree)
    return new_choices
end

# Insert op_type at subtree_idx with all possible base kernels.
function get_choices_insert_oper_at_subtree(trace::Gen.Trace,
        subtree_idx::Integer, op_type::Integer)
    config = Gen.get_args(trace)[2]
    leaf_node_types = get_leaf_node_types(trace)
    new_choices_list = Gen.ChoiceMap[]
    for node_type in leaf_node_types
        @assert config.node_dist_leaf[node_type] > 0
        new_choices = get_choices_insert_oper_at_subtree(
            trace, subtree_idx, op_type, node_type)
        push!(new_choices_list, new_choices)
    end
    return new_choices_list
end

# Insert all possible op_type at subtree_idx with all possible base kernels.
# Replace all leaf nodes in the tree with all possible base kernels."""
function get_choices_insert_oper_at_subtree(trace::Gen.Trace)
    op_node_types = sort(get_op_node_types(trace))
    subtree_idxs = sort(get_all_node_indexes(trace))
    new_choices = [
        get_choices_insert_oper_at_subtree(trace, s, o)
        for s in subtree_idxs
        for o in op_node_types
    ]
    return vcat(new_choices...)
end

# ==============
# Overall Search
# ==============

function update_and_optimize_structure(
        trace::Gen.Trace,
        new_choices::Gen.ChoiceMap;
        check::Bool=false,
        observations::Gen.ChoiceMap=Gen.EmptyChoiceMap())
    MAX_OPT = 500
    trace_prop, = Gen.generate(Gen.get_gen_fn(trace), Gen.get_args(trace), new_choices)
    # Optimize parameters of proposed trace.
    leaf_addrs = Inference.get_gp_parameter_addresses(trace_prop)
    selection = Gen.select(leaf_addrs...)
    prev_score = Gen.get_score(trace_prop)
    for i=1:MAX_OPT
        trace_prop = Gen.map_optimize(trace_prop, selection)
        new_score = Gen.get_score(trace_prop)
        @assert prev_score <= new_score
        @match (new_score == prev_score) begin
                true  => (break)
                false => (prev_score = new_score)
                end
    end
    check && Gen.check_observations(Gen.get_choices(trace_prop), observations)
    aic_prop = compute_aic(trace_prop; k=length(leaf_addrs))
    return (trace_prop, aic_prop)
end

function enumerate_next_structures(trace::Gen.Trace)
    new_choices_replace_leaf = get_choices_replace_leaf(trace)
    new_choices_insert_oper = get_choices_insert_oper_at_subtree(trace)
    return vcat(new_choices_replace_leaf, new_choices_insert_oper)
end

function greedy_search_initialize(
        ts::Vector{Float64},
        xs::Vector{Float64},
        config::GP.GPConfig;
        check::Bool=false)
    leaf_node_types = get_leaf_node_types(config)
    observations = Gen.choicemap((:xs, xs))
    if !isnothing(config.noise)
        observations[:noise] = Model.untransform_param(:noise, config.noise)
    end
    trace, = Gen.generate(Model.model, (ts, config,), observations)
    # Optimize each base kernel.
    results = Vector{Any}(undef, length(leaf_node_types))
    Threads.@threads for i=1:length(leaf_node_types)
        node_type = leaf_node_types[i]
        new_tree = make_base_kernel_choicemap(1, node_type, config)
        new_choices = Gen.choicemap(observations)
        Gen.set_submap!(new_choices, :tree, new_tree)
        (trace_prop, aic_prop) = update_and_optimize_structure(
            trace, new_choices; check=check, observations=observations)
        results[i] = (trace_prop, aic_prop)
    end
    # Select base kernel with lowest AIC.
    (new_trace, new_aic) = argmin(x -> x[2], results)
    return (new_trace, new_aic, observations)
end

function greedy_search_extend(
        trace::Gen.Trace,
        aic::Float64;
        verbose::Bool=false,
        check::Bool=false,
        observations::Gen.ChoiceMap=Gen.EmptyChoiceMap())
    new_choices_list = enumerate_next_structures(trace)
    verbose && println("Proposals: $(length(new_choices_list))")

    # Optimize each new choice.
    results = Vector{Any}(undef, length(new_choices_list))
    Threads.@threads for i=1:length(new_choices_list)
        new_choices = new_choices_list[i]
        (trace_prop, aic_prop) = update_and_optimize_structure(
            trace, new_choices; check=check, observations=observations)
        results[i] = (trace_prop, aic_prop)
    end

    # Select new structure with lowest AIC.
    (trace_prop, aic_prop) = argmin(x -> x[2], results)

    return @match (aic_prop < aic) begin
        true  => (trace_prop, aic_prop, true)
        false => (trace, aic, false)
    end
end

end # module Greedy
