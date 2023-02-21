module Model

using ..GP
using Gen

using Match

JITTER = 1e-5

transform_log_normal(z::Real, mu::Real, sigma::Real) = exp(mu + sigma*z)
untransform_log_normal(param::Real, mu::Real, sigma::Real) = (log(param) - mu) / sigma

function transform_logit_normal(z::Real, scale::Real, mu::Real, sigma::Real)
    return scale * 1 / (1 + exp(-(mu + sigma*z)))
end

function untransform_logit_normal(param::Real, scale::Real, mu::Real, sigma::Real)
    return (log(param / (scale - param)) - mu) / sigma
end

transform_param(field::Symbol, z::Real) = @match field begin
    :gamma  => transform_logit_normal(z, 2, 0, 1)
    _       => transform_log_normal(z, -1.5, 1.)
end

untransform_param(field::Symbol, param::Real) = @match field begin
    :gamma  => untransform_logit_normal(param, 2, 0, 1)
    _       => untransform_log_normal(param, -1.5, 1.)
end

"""Return distribution over node types at a given index."""
function get_node_dist(idx::Int, config::GPConfig)
    depth = GP.idx_to_depth(idx)
    @assert config.max_depth == -1 || 1 <= depth <= config.max_depth
    if depth == config.max_depth
        return config.node_dist_leaf
    elseif config.changepoints
        return config.node_dist_cp
    else
        return config.node_dist_nocp
    end
end

@gen function covariance_prior(idx::Int, config::GPConfig)
    node_dist = get_node_dist(idx, config)
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
        params = []
        for field in fieldnames(NodeType)
            log_param  = {(idx, field)} ~ normal(0, 1)
            param = transform_param(field, log_param)
            push!(params, param)
        end
        node = NodeType(params...)

    # BinaryOpNode
    elseif node_type in [config.Plus, config.Times]
        idx_l = Gen.get_child(idx, 1, config.max_branch)
        idx_r = Gen.get_child(idx, 2, config.max_branch)
        config = GPConfig(config; changepoints=false)
        left_node = {*} ~ covariance_prior(idx_l, config)
        right_node = {*} ~ covariance_prior(idx_r, config)
        NodeType = config.index_to_node[node_type]
        node = NodeType(left_node, right_node)

    # ChangePoint
    elseif node_type == config.ChangePoint
        # @assert config.changepoints
        # It is impossible to be here if the assertion is false,
        # but we should allow such traces to have probability zero
        # (for inference) rather than force an assertion error.
        location = {(idx, :location)} ~ normal(0, 1)
        param = transform_param(:location, location)
        child1 = Gen.get_child(idx, 1, config.max_branch)
        child2 = Gen.get_child(idx, 2, config.max_branch)
        left_node = {*} ~ covariance_prior(child1, config)
        right_node = {*} ~ covariance_prior(child2, config)
        node = ChangePoint(left_node, right_node, param, .001)

    else
        error("Unknown node type: $(node_type)")
    end

    return node
end

@gen function model(ts::Vector{Float64}, config::GPConfig)
    n = length(ts)
    covariance_fn = {:tree} ~ covariance_prior(1, config)
    noise ~ normal(0, 1)
    noise = transform_param(:noise, noise) + JITTER
    cov_matrix = GP.compute_cov_matrix_vectorized(covariance_fn, noise, ts)
    xs ~ mvnormal(zeros(n), cov_matrix)
    return covariance_fn
end

export covariance_prior
export model

end # module Model
