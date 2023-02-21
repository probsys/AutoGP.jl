module Inference
  using ..GP
  using ..Model
  using ..TimeIt
  using ..Transforms

  using Gen

  include("inference_utils.jl")
  include("inference_rejuv_tree.jl")
  include("inference_smc_anneal_data.jl")

end # module Inference
