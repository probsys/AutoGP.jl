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

# Tree rejuvenation via Attach-Detach move.

function get_node_dist_attach_detach(
        idx::Int,
        path_to_hole::Dict{Int,Bool},
        force_cp::Bool,
        config::GPConfig)

    node_dist = collect(Model.get_node_dist(idx, config))

    if !haskey(path_to_hole, idx)
        # Node is not on path to hole, probabilities from Model.
        return node_dist
    elseif path_to_hole[idx] == true
        # Node is a hole, no choices.
        return nothing
    else
        # Internal node on path to hole must be a branch.
        if force_cp
            # Dirac at ChangePoint.
            @assert config.changepoints
            node_dist = zeros(length(node_dist))
            node_dist[config.ChangePoint] = 1.
        else
            # Uniform over SUM, PRODUCT, and maybe CHANGEPOINT.
            node_dist[1:length(config.node_dist_leaf)] .= 0
            node_dist ./= sum(node_dist)
        end
        return node_dist
    end
end

@gen function covariance_proposal_attach_detach(
        idx::Int,
        path_to_hole::Dict{Int,Bool},
        force_cp::Bool,
        config::GPConfig)

    node_dist = get_node_dist_attach_detach(idx, path_to_hole, force_cp, config)
    isnothing(node_dist) && return
    node_type = {(idx, :node_type)} ~ categorical(node_dist)

    # LeafNode
    if node_type in [
            config.Constant,
            config.Linear,
            config.SquaredExponential,
            config.GammaExponential,
            config.Periodic,
            ]
        NodeType = config.index_to_node[node_type]
        for field in fieldnames(NodeType)
            {(idx, field)} ~ normal(0, 1)
        end
    # BinaryOpNode
    elseif node_type in [config.Plus, config.Times]
        config = GPConfig(config;
            changepoints=config.changepoints && (!config.changepoints_at_root))
        idx_l = Gen.get_child(idx, 1, config.max_branch)
        idx_r = Gen.get_child(idx, 2, config.max_branch)
        {*} ~ covariance_proposal_attach_detach(idx_l, path_to_hole, force_cp, config)
        {*} ~ covariance_proposal_attach_detach(idx_r, path_to_hole, force_cp, config)
    # ChangePoint
    elseif node_type == config.ChangePoint
        @assert config.changepoints
        idx_l = Gen.get_child(idx, 1, config.max_branch)
        idx_r = Gen.get_child(idx, 2, config.max_branch)
        {(idx, :location)} ~ normal(0, 1)
        {*} ~ covariance_proposal_attach_detach(idx_l, path_to_hole, force_cp, config)
        {*} ~ covariance_proposal_attach_detach(idx_r, path_to_hole, force_cp, config)
    else
        error("Invalid node type $(node_type)")
    end
end

"""Proposal for the detach_attach IMCMC kernel."""
@gen function detach_attach_proposal(trace::Gen.Trace, biased::Bool, noroot::Bool)
    # Extract the covariance tree.
    config = Gen.get_args(trace)[2]
    tree = Gen.get_retval(trace)
    tree_depth = GP.depth(tree)
    tree_size = size(tree)

    # Fail.
    if (tree_size == 1) && (tree_depth == config.max_depth)
        error("Cannot apply ATTACH-DETACH with config.max_depth = 1.")
    end

    # Decide whether to run DETACH or ATTACH move.
    p_detach = (tree_size == 1) ? 0. : .5
    detach_move ~ bernoulli(p_detach)

    # Simulate the selected move.
    if detach_move
        return {*} ~ detach_attach_proposal__propose_detach(trace, biased, noroot)
    else
        return {*} ~ detach_attach_proposal__propose_attach(trace, biased, noroot, config.max_depth)
    end
end

"""Proposal for the detach move."""
@gen function detach_attach_proposal__propose_detach(
            trace::Gen.Trace, biased::Bool, noroot::Bool)
    config = Gen.get_args(trace)[2]
    tree = Gen.get_retval(trace)
    # Sample the root of subtree that detach will occur with respect to.
    (subtree_node_a, subtree_idx_a, subtree_depth_a) =
        {:pick_node_a} ~ pick_random_node(
            tree, 1, 1, biased, false, false, config.max_branch)
    # Sample the root of the subtree to detach.
    (subtree_node_b, subtree_idx_b, subtree_depth_b) =
        {:pick_node_b} ~ pick_random_node(
            subtree_node_a, subtree_idx_a, subtree_depth_a,
            biased, false, noroot, config.max_branch)
    # @assert subtree_idx > 1
    return [
        (subtree_node_a, subtree_idx_a, subtree_depth_a),
        (subtree_node_b, subtree_idx_b, subtree_depth_b),
    ]
end

"""Proposal for the attach move."""
@gen function detach_attach_proposal__propose_attach(
            trace, biased::Bool, noroot::Bool, max_depth::Integer)
    config = Gen.get_args(trace)[2]
    tree = Gen.get_retval(trace)
    tree_depth = GP.depth(tree)
    # Sample the root of subtree that attach will occur with respect to.
    subtree_node_a, subtree_idx_a, subtree_depth_a =
        {:pick_node_a} ~ pick_random_node(
            tree, 1, 1, biased, false, false, config.max_branch)
    # Sample aux tree structure.
    subtree_height_a = GP.depth(subtree_node_a)
    max_depth_aux = (max_depth == -1) ? -1 : max_depth - (subtree_height_a - 1)
    path_to_hole = Dict{Int,Bool}()
    subtree_idx_b = {:pick_node_b} ~ generate_random_path(
                        subtree_idx_a, subtree_depth_a, max_depth_aux,
                        noroot, config.max_branch, path_to_hole)
    # @assert subtree_idx_b > 1
    # @assert path_to_hole[subtree_idx_b] == true
    root_type = trace[:tree => (subtree_idx_a, :node_type)]
    force_cp = (root_type == config.ChangePoint) && config.changepoints_at_root
    aux_tree ~ covariance_proposal_attach_detach(
                    subtree_idx_a, path_to_hole, force_cp,
                    GPConfig(config; max_depth=max_depth))
    return (subtree_idx_a, subtree_idx_b)
end

"""Involution for the detach_attach IMCMC kernel."""
function detach_attach_involution(
        model_trace::Gen.Trace,
        proposal_choices_in::Gen.ChoiceMap,
        proposal_retval,
        proposal_args::Tuple)

    if proposal_choices_in[:detach_move]
        # DETACH MOVE.
        model_choices_out, proposal_choices_out =
            detach_attach_involution__apply_detach(
                model_trace,
                proposal_choices_in,
                proposal_retval,
                proposal_args)
        proposal_choices_out[:detach_move] = false
    else
        # ATTACH MOVE.
        model_choices_out, proposal_choices_out =
            detach_attach_involution__apply_attach(
                model_trace,
                proposal_choices_in,
                proposal_retval,
                proposal_args)
        proposal_choices_out[:detach_move] = true
    end

    # Compute the new trace and weight.
    (model_trace_out, weight, retdiff, discard) = update(model_trace, model_choices_out)

    return (model_trace_out, proposal_choices_out, weight)
end

"""Involution for the detach_attach IMCMC kernel (applies DETACH move)."""
function detach_attach_involution__apply_detach(
        model_trace::Gen.Trace,
        proposal_choices_in::Gen.ChoiceMap,
        proposal_retval,
        proposal_args::Tuple)

    # Extract model arguments.
    ts, config = Gen.get_args(model_trace)

    # Extract choices of model trace.
    model_choices_in = Gen.get_choices(model_trace)

    # Obtain detached subtree from proposal.
    (subtree_node_a, subtree_idx_a, subtree_depth_a) = proposal_retval[1]
    (subtree_node_b, subtree_idx_b, subtree_depth_b) = proposal_retval[2]

    # Write the new model.
    model_choices_out = choicemap()
    current_tree = get_submap(model_choices_in, :tree)
    subtree_choices = extract_subtree_choices(
        current_tree, subtree_idx_b, config; idx_subtree_new=subtree_idx_a)
    new_choices, discard = replace_subtree_choices(current_tree, subtree_choices, subtree_idx_a, config)
    set_submap!(model_choices_out, :tree, new_choices)

    # Write the backward proposal choices.
    proposal_choices_out = choicemap()
    proposal_choices_out[:detach_move] = false

    # -- Write to :pick_node.
    set_submap!(proposal_choices_out, :pick_node_a, get_submap(proposal_choices_in, :pick_node_a))
    set_submap!(proposal_choices_out, :pick_node_b, get_submap(proposal_choices_in, :pick_node_b))

    # -- Write to :aux_tree
    ignore = Set(subtree_idx_b)
    for (address, value) in discard
        idx = address[1]
        if idx in ignore
            for i=1:config.max_branch
                push!(ignore, Gen.get_child(idx, i, config.max_branch))
            end
        else
            proposal_choices_out[:aux_tree => address] = value
        end
    end

    # Return the new choices.
    return model_choices_out, proposal_choices_out
end

"""Involution for the detach_attach IMCMC kernel (applies ATTACH move)."""
function detach_attach_involution__apply_attach(
        model_trace::Gen.Trace,
        proposal_choices_in::Gen.ChoiceMap,
        proposal_retval,
        proposal_args::Tuple)

    # Extract model arguments.
    ts, config = get_args(model_trace)
    # Extract choices of model trace.
    model_choices_in = get_choices(model_trace)

    # Obtain index of the hole in the auxiliary tree.
    subtree_idx_a = proposal_retval[1]
    subtree_idx_b = proposal_retval[2]

    # Write the new model.
    model_choices_out = choicemap()
    current_tree = get_submap(model_choices_in, :tree)
    subtree_choices_current = extract_subtree_choices(
        current_tree, subtree_idx_a, config; idx_subtree_new=subtree_idx_b)
    subtree_choices_aux = get_submap(proposal_choices_in, :aux_tree)
    subtree_choices_new = merge(subtree_choices_current, subtree_choices_aux)
    new_tree, discard = replace_subtree_choices(
        current_tree, subtree_choices_new, subtree_idx_a, config)
    set_submap!(model_choices_out, :tree, new_tree)

    # Write the backward proposal choices.
    proposal_choices_out = choicemap()

    # -- Write to :pick_node.
    set_submap!(proposal_choices_out, :pick_node_b, get_submap(proposal_choices_in, :pick_node_b))
    set_submap!(proposal_choices_out, :pick_node_a, get_submap(proposal_choices_in, :pick_node_a))

    # Return the new choices.
    return model_choices_out, proposal_choices_out
end
