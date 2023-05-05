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

using Documenter
using AutoGP

# Build the tutorials.
# dir = dirname(@__FILE__)
# tutorials = joinpath(dirname(@__FILE__), "src", "tutorials")
# run(Cmd(`./build.sh`, dir=tutorials))

# Make the documents.
makedocs(
    sitename="AutoGP",
    format=Documenter.HTML(prettyurls=false, ansicolor=true),
    modules=[AutoGP],
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/overview.md",
            "tutorials/iclaims.md",
            "tutorials/callbacks.md",
            "tutorials/greedy_mcmc.md",
            "tutorials/test2.md",
        ],
        "Automatic Modeling API" => "api.md",
        "Gaussian Process Library" => "gp.md",
        "Utilities" => "utils.md",
        "Test" => "test.md"
    ],
)

deploydocs(repo="github.com/fsaad/AutoGP.jl.git")
