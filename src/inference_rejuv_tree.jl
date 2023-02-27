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

# Included by ./Inference.jl

include("inference_rejuv_tree_sr.jl")
include("inference_rejuv_tree_da.jl")

using Match

"""Proposal for overall IMCMC kernel."""
@gen function tree_rejuvenation_proposal(trace::Gen.Trace, biased::Bool)
    config = Gen.get_args(trace)[2]
    allow_detach_attach = config.max_depth != 1
    p_detach_attach = allow_detach_attach ? .5 : 0.
    move_type ~ bernoulli(p_detach_attach)
    return @match move_type begin
        0   => ({*} ~ subtree_replace_proposal(trace, biased))
        1   => ({*} ~ detach_attach_proposal(trace, biased, false))
        _   => error("Unknown move type: $(move_type)")
    end
end

"""Involution for overall IMCMC kernel."""
function tree_rejuvenation_involution(
        model_trace::Gen.Trace,
        proposal_choices_in::Gen.ChoiceMap,
        proposal_retval,
        proposal_args::Tuple)
    move_type = proposal_choices_in[:move_type]
    involution_fn = @match move_type begin
        0 => subtree_replace_involution
        1 => detach_attach_involution
        _ => error("Unknown move type $(move_type)")
    end
    (model_trace_out, proposal_choices_out, weight) =
        involution_fn(
            model_trace,
            proposal_choices_in,
            proposal_retval,
            proposal_args)
    proposal_choices_out[:move_type] = move_type
    return model_trace_out, proposal_choices_out, weight
end
