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

"""Utilities for creating callbacks to inspect SMC inference."""
module Callbacks

import ..AutoGP
import ..GP
import ..Inference
import ..Model
import ..Transforms

import Distributions
import Gen

using Printf: @sprintf

function validate_callback(fn::Function; kwargs...)
    fn_name = Symbol(fn)
    # Validate 1 method.
    if length(methods(fn)) != 1
        error("Callback function $(fn_name) must have exactly 1 method")
    end
    m = methods(fn)[1]
    # Validate no named arguments.
    argnames = Base.method_argnames(m)
    @assert argnames[1] == Symbol("#self#")
    if length(argnames) != 1
        error("Callback $(fn_name) must have no argnames, received $(argnames[2:end])")
    end
    # Validate keyword arguments in signature of fn.
    fn_kwargs = Base.kwarg_decl(m)
    fn_kwargs_symbol = [s for s in fn_kwargs if endswith(String(s), "...")]
    @assert length(fn_kwargs_symbol) <= 1
    if length(fn_kwargs_symbol) == 0
        error("Callback $(fn_name) requires a varags specifier")
    end
    # Validate keyword arguments in call to make_smc_callback.
    unknown_kwargs = [k for k in keys(kwargs) if !(k in fn_kwargs)]
    if length(unknown_kwargs) > 0
        error("Unknown kwargs $(unknown_kwargs) in make_smc_callback for callback $(fn_name)")
    end
    # TODO: Check for missing kwargs. Not implemented because it is not clear
    # how to access the list of required and optional keyword arguments of fn.
end

"""
    make_smc_callback(fn::Function, model::AutoGP.GPModel; kwargs...)

Convert `fn` into a callback for [`AutoGP.fit_smc!`](@ref)`(model, ...)`.

The function `fn` must have a signature of the form `fn(; [<opt>,] kw...)`,
where `kw...` is a varargs specifier and `<opt>` denotes a (possibly empty)
collection of required and optional keyword arguments.

For example  `fn(; a, b=2, kw...)` is valid because `fn` takes no named
arguments and the varargs specifier `kw...` is present. If `fn` includes
named keyword arguments (i.e. `a` and `b`), then all required keyword
arguments (i.e., `a`) must be provided in the call to `make_smc_callback`
and optional keyword arguments (i.e., `b`) may be provided, as shown in the
examples below.

```julia
model = AutoGP.GPModel(...)
fn = (; a, b=2, kw...) -> ...
make_smc_callback(fn, model; a=2)       # valid, `a` specified, `b` optional.
make_smc_callback(fn, model; a=1, b=2)  # valid, `a` and `b` are specified.
make_smc_callback(fn, model)            # invalid, `a` required but not specified.
make_smc_callback(fn, model; b=2)       # invalid, `a` required but not specified.
```

The callback return by `make_smc_callback` is guaranteed to receive in its
varargs (which we called `kw`) an "SMC" state which. The following
variables can be accessed in the body of `fn` by indexing `kw`:

- `kw[:step]::Integer`: Current SMC step.
- `kw[:model]::AutoGP.GPModel`: Inferred model at current SMC step.
- `kw[:ds_next]::AutoGP.IndexType`: Future `ds` (time points) to be observed in later SMC rounds.
- `kw[:y_next]::Vector{<:Real}`:  Future `y` (observations) to be observed in later SMC rounds.
- `kw[:rejuvenated]::Bool`: Did rejuvenation occur at current step?
- `kw[:resampled]:Bool`: Did resampling occur at current step?
- `kw[:elapsed]::Float64`: Wall-clock inference runtime elapsed.
- `kw[:verbose]::Bool`: Verbose setting.
- `kw[:schedule]::Vector{Integer}`: The SMC data annealing schedule.
- `kw[:permutation]::Vector{Integer}`: Indexes used to shuffle (`model.ds`, `model.y`).
"""
function make_smc_callback(fn::Function, model::AutoGP.GPModel; kwargs...)

    validate_callback(fn; kwargs...)

    function g(; kwargs_smc...)
        state       = kwargs_smc[:state]
        ts          = kwargs_smc[:ts]
        xs          = kwargs_smc[:xs]
        permutation = kwargs_smc[:permutation]
        schedule    = kwargs_smc[:schedule]
        step        = kwargs_smc[:step]
        elapsed     = kwargs_smc[:elapsed]
        rejuvenated = kwargs_smc[:rejuvenated]
        resampled   = kwargs_smc[:resampled]
        verbose     = kwargs_smc[:verbose]

        # Permute observations.
        ds_permuted = model.ds[permutation]
        y_permuted = model.y[permutation]

        # Observed data.
        ds_obs = ds_permuted[1:step]
        y_obs = y_permuted[1:step]

        # Remaining data.
        ds_next = ds_permuted[step+1:end]
        y_next = y_permuted[step+1:end]

        # Current model.
        current_model = AutoGP.GPModel(
            state,
            model.config,
            ds_obs,
            y_obs,
            model.ds_transform,
            model.y_transform)

        return fn(;
            # Callback-specific arguments.
            kwargs...,
            # SMC stats.
            model=current_model,
            ds_next=ds_next,
            y_next=y_next,
            step=step,
            permutation=permutation,
            schedule=schedule,
            rejuvenated=rejuvenated,
            resampled=resampled,
            elapsed=elapsed,
            verbose=verbose)
    end

    return g
end

end # Module Callbacks
