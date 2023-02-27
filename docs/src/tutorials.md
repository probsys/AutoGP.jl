# Tutorials

```@contents
Pages = [
    "tutorials/overview.md",
    "tutorials/iclaims.md",
    "tutorials/callbacks.md",
    "tutorials/greedy_mcmc.md",
]
```

The above tutorials can be run interactively in the browser using

    $ export JULIA_NUM_THREADS=$(nproc)
    $ export JULIA_PROJECT=/path/to/AutoGP.jl
    $ cd ${JULIA_PROJECT}/docs/src/tutorials
    $ julia -e 'using IJulia; notebook(dir=".")'

The notebook will be available at localhost, typically port 8888.
