"""
Main module.

# Exports

$(EXPORTS)
"""
module AutoGP

using DocStringExtensions

# Helper modules.
include("Rescale.jl")
include("Schedule.jl")
include("TimeIt.jl")
include("Transforms.jl")

# Main modules.
include("GP.jl")
include("Model.jl")
include("Inference.jl")
include("Greedy.jl")

# Top-level API
include("api.jl")

# Callbacks.
include("Callbacks.jl")

end # module AutoGP
