# Tutorials

```@contents
Pages = [
    "tutorials/overview.md",
    "tutorials/iclaims.md",
    "tutorials/callbacks.md",
    "tutorials/greedy_mcmc.md",
]
```

To run the tutorials interactively, first install
[IJulia](https://julialang.github.io/IJulia.jl/stable/manual/installation/)
and [PyPlot](https://github.com/JuliaPy/PyPlot.jl).
Then run the following commands from the terminal, making sure to replace
`</path/to/>AutoGP.jl` with the actual path to the directory:

    $ export JULIA_NUM_THREADS=$(nproc)
    $ export JULIA_PROJECT=</path/to/>AutoGP.jl
    $ cd ${JULIA_PROJECT}/docs/src/tutorials
    $ julia -e 'using IJulia; notebook(dir=".")'

The notebook server will be available in the browser at https://localhost:8888.
