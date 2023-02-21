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
        ],
        "Automatic Modeling API" => "api.md",
        "Gaussian Process Library" => "gp.md",
        "Utilities" => "utils.md",
    ],
)
