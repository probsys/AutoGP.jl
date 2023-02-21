# Tree rejuvenation via Subtree-Replace move.

@gen function subtree_replace_proposal(trace::Gen.Trace, biased::Bool)
    # Extract arguments of trace.
    (ts, config) = Gen.get_args(trace)

    # Choose a random node in the current trace to replace.
    root = trace[]
    (subtree_node, subtree_idx, subtree_depth) =
            {:pick_node} ~ pick_random_node(root, 1, 1, biased, false, false, config.max_branch)

    # Determine whether a ChangePoint is permitted in the proposed subtree.
    changepoints = begin
        if !config.changepoints
            false
        elseif subtree_idx == 1
            true
        else
         # false
           parent_idx = Gen.get_parent(subtree_idx, config.max_branch)
           parent_type = trace[:tree => (parent_idx, :node_type)]
           @assert parent_type in [config.Plus, config.Times, config.ChangePoint]
           parent_type == config.ChangePoint
        end
    end

    # Simulate the proposed subtree.
    config = GPConfig(config; changepoints=changepoints)
    subtree_proposed = {:subtree} ~ covariance_prior(subtree_idx, config)

    # Return subtree index, depth, and proposed.
    return (subtree_idx, subtree_depth, subtree_proposed)
end

"""
Involutive mapping from (model trace, proposal trace)
to (new model trace, reverse proposal trace).
"""
# @transform subtree_replace_involution (model_in, aux_in) to (model_out, aux_out) begin
#     # Copy the path to selected node.
#     @copy(aux_in[:pick_node], aux_out[:pick_node])
#     # Copy the proposed subtree into the output model trace.
#     @copy(aux_in[:subtree], model_out[:tree])
#     # Copy the entire previous tree into the output proposal trace.
#     # Technically speaking we should only copy the discarded subtree,
#     # but since the MH involution uses generate
#     # the extra constraints in :tree are ignored.
#     @copy(model_in[:tree], aux_out[:subtree])
# end

"""Involution for the detach_attach IMCMC kernel."""
function subtree_replace_involution(
        model_trace::Gen.Trace,
        proposal_choices_in::Gen.ChoiceMap,
        proposal_retval,
        proposal_args::Tuple)

    (subtree_idx, subtree_depth, subtree_proposed) = proposal_retval

    # Obtain the new model trace.
    model_choices_diff = choicemap()
    proposal_choices_subtree = get_submap(proposal_choices_in, :subtree)
    set_submap!(model_choices_diff, :tree, proposal_choices_subtree)
    (model_trace_out, weight, retdiff, discard) = update(model_trace, model_choices_diff)

    # Write the reverse proposal trace.
    proposal_choices_out = choicemap()
    proposal_choices_pick_node = get_submap(proposal_choices_in, :pick_node)
    set_submap!(proposal_choices_out, :pick_node, proposal_choices_pick_node)
    set_submap!(proposal_choices_out, :subtree, get_submap(discard, :tree))

    return (model_trace_out, proposal_choices_out, weight)

end
