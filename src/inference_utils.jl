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

get_p_done(node::LeafNode, biased::Bool, leaf::Bool, noroot::Bool) =
    noroot ? error("Impossible pick_random_node call.") : 1.

get_p_done(node::BinaryOpNode, biased::Bool, leaf::Bool, noroot::Bool) =
    noroot ? 0. : leaf ? 0. : biased ? .5 : 1. / size(node)

get_p_recurse_left(node::BinaryOpNode, biased::Bool) =
    biased ? .5 : size(node.left) / (size(node) - 1)

"""Select a node at random from the tree; uniformly or biased to leaves."""
@gen function pick_random_node(
            node::Node,         # current node
            idx::Int,           # index of the current node
            depth::Int,         # depth of the current node
            biased::Bool,       # biased or uniform sampling
            leaf::Bool,         # leaves only
            noroot::Bool,       # disallow selecting root
            max_branch::Int     # maximum branch factor
            )

    # Terminate?
    p_done = get_p_done(node, biased, leaf, noroot)
    if ({:done => depth} ~ bernoulli(p_done))
        return (node, idx, depth)
    end

    # Probability of recursing to left child.
    p_recurse_left = get_p_recurse_left(node, biased)

    # Recurse to left child.
    if ({:recurse_left => idx} ~ bernoulli(p_recurse_left))
        idx_l = Gen.get_child(idx, 1, max_branch)
        return {*} ~ pick_random_node(
            node.left, idx_l, depth + 1, biased, leaf, false, max_branch)

    # Recurse to right child.
    else
        idx_r = Gen.get_child(idx, 2, max_branch)
        return {*} ~ pick_random_node(
            node.right, idx_r, depth + 1, biased, leaf, false, max_branch)
    end

end

"""Generate a random path in a binary tree."""
@gen function generate_random_path(
            idx::Int,                       # index of the current node
            depth::Int,                     # depth of the current node
            max_depth::Int,                 # maximum depth
            noroot::Bool,                   # disallow generating path to root
            max_branch::Int,                # maximum branch factor
            path::Dict{Int,Bool}            # current paths maps index to halt?
            )
    @assert (max_depth == -1) || (1 <= depth <= max_depth)
    @assert (max_depth == -1) || !noroot || (1 < max_depth)
    @assert !haskey(path, idx)
    # Terminate?
    p_done = noroot ? 0. : depth == max_depth ? 1. : .5
    if ({:done => depth} ~ bernoulli(p_done))
        path[idx] = true
        return idx
    end
    path[idx] = false
    # Recurse to left child.
    if ({:recurse_left => idx} ~ bernoulli(.5))
        idx_l = Gen.get_child(idx, 1, max_branch)
        return {*} ~ generate_random_path(idx_l, depth + 1, max_depth, false, max_branch, path)
    # Recurse to right child.
    else
        idx_r = Gen.get_child(idx, 2, max_branch)
        return {*} ~ generate_random_path(idx_r, depth + 1, max_depth, false, max_branch, path)
    end
end

"""Return trace addresses of numeric parameters."""
function get_gp_parameter_addresses(trace::Gen.Trace)
    config = get_args(trace)[2]
    noise_addrs = isnothing(config.noise) ? [:noise] : []
    choices = Gen.get_choices(trace)
    submap = Gen.get_submap(choices, :tree)
    leaf_addrs = [
        :tree => address
        for (address, value) in Gen.get_values_shallow(submap)
            if address[2] != :node_type
    ]
    return vcat(noise_addrs, leaf_addrs)
end

"""Return trace address and values of numeric parameters."""
function get_gp_parameter_values(trace::Gen.Trace)
    addresses = get_gp_parameter_addresses(trace)
    choices = choicemap()
    for address in addresses
        choices[address] = trace[address]
    end
    return choices
end

"""Extract choices of a subtree at idx_subtree; use idx_subtree_new to reindex."""
function extract_subtree_choices(
        choices::Gen.ChoiceMap,
        idx_subtree::Integer,
        config::GPConfig;
        idx_subtree_new=idx_subtree)
    subtree = Gen.choicemap()
    @assert Gen.has_value(choices, (idx_subtree, :node_type))
    stack = [(idx_subtree, idx_subtree_new)]
    while length(stack) > 0
        # Copy addresses at current index.
        idx_in, idx_out = popfirst!(stack)
        node_type = choices[(idx_in, :node_type)]
        subtree[(idx_out, :node_type)] = node_type
        for f in fieldnames(config.index_to_node[node_type])
            if Gen.has_value(choices, (idx_in, f))
                subtree[(idx_out, f)] = choices[(idx_in, f)]
            end
        end
        # Recurse.
        idx_left_in = Gen.get_child(idx_in, 1, config.max_branch)
        if Gen.has_value(choices, (idx_left_in, :node_type))
            idx_left_out = Gen.get_child(idx_out, 1, config.max_branch)
            idx_right_in = Gen.get_child(idx_in, 2, config.max_branch)
            idx_right_out = Gen.get_child(idx_out, 2, config.max_branch)
            @assert Gen.has_value(choices, (idx_right_in, :node_type))
            push!(stack, (idx_left_in, idx_left_out))
            push!(stack, (idx_right_in, idx_right_out))
        end
    end
    return subtree
end

"""Replace all choices rooted at idx_subtree with choices_subtree."""
function replace_subtree_choices(
        choices::Gen.ChoiceMap,
        choices_subtree::Gen.ChoiceMap,
        idx_subtree::Integer,
        config::GPConfig)
    choices_no_subtree = Gen.choicemap()
    @assert Gen.has_value(choices, (idx_subtree, :node_type))
    @assert Gen.has_value(choices_subtree, (idx_subtree, :node_type))
    ignore = Set(idx_subtree)
    discard = []
    for (address, value) in sort(collect(Gen.get_values_shallow(choices)))
        idx = address[1]
        if idx in ignore
            push!(discard, (address, value))
            for i=1:config.max_branch
                push!(ignore, Gen.get_child(idx, i, config.max_branch))
            end
        else
            choices_no_subtree[address] = value
        end
    end
    return merge(choices_no_subtree, choices_subtree), discard
end

"""Return MvNormal predictive distribution at new indexes ts."""
function predict_mvn(
        trace::Gen.Trace,
        ts::Vector{Float64};
        noise_pred::Union{Nothing, Float64}=nothing)
    ts_train = Gen.get_args(trace)[1]
    xs_train = trace[:xs]
    cov_fn = trace[]
    noise = Model.transform_param(:noise, trace[:noise]) + Model.JITTER
    return Distributions.MvNormal(cov_fn, noise, ts_train, xs_train, ts; noise_pred=noise_pred)
end

"""Return the posterior predictive mean and quantiles at new indexes ts."""
function predict(
        trace::Gen.Trace,
        ts::Vector{Float64};
        quantiles::Vector{Float64}=Float64[],
        noise_pred::Union{Nothing,Float64}=nothing)
    all(0 .<= quantiles .<= 1) || error("Quantiles must be in [0,1]")
    dist = predict_mvn(trace, ts; noise_pred=noise_pred)
    y_mean = Distributions.mean(dist)
    y_bounds = Distributions.quantile(dist, [quantiles])
    return y_mean, y_bounds
end

"""Convert Node to the integer code."""
function node_to_integer(node::Node, config::GPConfig)
    nodename = split(string(typeof(node)), ".")[end]
    field = Symbol(nodename)
    return getfield(config, field)
end

"""Convert Node to Gen.ChoiceMap."""
function node_to_choicemap end

function node_to_choicemap(node::Node, config::GPConfig; params=nothing)
    return node_to_choicemap(node, 1, config; params=params)
end

function node_to_choicemap(node::LeafNode, idx::Int, config::GPConfig; params=nothing)
    NodeType = typeof(node)
    choices = Gen.choicemap()
    choices[(idx, :node_type)] = node_to_integer(node, config)
    if isnothing(params) || params
        for field in fieldnames(NodeType)
            param = getfield(node, field)
            choices[(idx, field)] = Model.untransform_param(field, param)
        end
    end
    return choices
end

function node_to_choicemap(node::Union{Plus,Times}, idx::Int, config::GPConfig; params=nothing)
    choices = Gen.choicemap()
    choices[(idx, :node_type)] = node_to_integer(node, config)
    idx_l = Gen.get_child(idx, 1, config.max_branch)
    idx_r = Gen.get_child(idx, 2, config.max_branch)
    c_l = node_to_choicemap(node.left, idx_l, config; params=params)
    c_r = node_to_choicemap(node.right, idx_r, config; params=params)
    return merge(choices, c_l, c_r)
end

function node_to_choicemap(node::ChangePoint, idx::Int, config::GPConfig; params=nothing)
    choices = Gen.choicemap()
    choices[(idx, :node_type)] = node_to_integer(node, config)
    if isnothing(params) || params
        choices[(idx, :location)] = Model.untransform_param(:location, node.location)
    end
    idx_l = Gen.get_child(idx, 1, config.max_branch)
    idx_r = Gen.get_child(idx, 2, config.max_branch)
    c_l = node_to_choicemap(node.left, idx_l, config; params=params)
    c_r = node_to_choicemap(node.right, idx_r, config; params=params)
    return merge(choices, c_l, c_r)
end

function get_observations_choicemap(trace::Gen.Trace)
    config = Gen.get_args(trace)[2]
    observations = Gen.choicemap((:xs, trace[:xs]))
    if !isnothing(config.noise)
        observations[:noise] = trace[:noise]
    end
    return observations
end

function node_to_trace(node::Node, trace::Gen.Trace)
    config = Gen.get_args(trace)[2]
    choicemap_obs = get_observations_choicemap(trace)
    choicemap_node = Gen.choicemap()
    Gen.set_submap!(choicemap_node, :tree, node_to_choicemap(node, config))
    constraints = merge(choicemap_node, choicemap_obs)
    constraints[:noise] = trace[:noise]
    return Gen.generate(
        Gen.get_gen_fn(trace),
        Gen.get_args(trace),
        constraints,
        )[1]
end
