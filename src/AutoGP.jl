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
