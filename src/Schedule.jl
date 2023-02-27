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

"""Utilities for creating SMC annealing schedules, used for [`AutoGP.fit_smc!`](@ref)."""
module Schedule

import ..AutoGP

"""
    linear_schedule(n::Integer, percent::Float64)
Adds roughly `n⋅percent` new observations at each step.
"""
function linear_schedule(n::Integer, percent::Float64)
    @assert 0 < n
    @assert 0 < percent < 1
    step = Integer(round(percent * n))
    checkpoints = collect(range(start=step, stop=n, step=step))
    remaining = n - checkpoints[end]
    @assert 0 <= remaining < step
    if remaining == 0
        return checkpoints
    elseif remaining < step / 2
        checkpoints[end] = n
        return checkpoints
    else
        return vcat(checkpoints, n)
    end
end

"""
    logarithmic_schedule(n::Integer, base::Integer, start::Integer)
The first step adds `start` observations (must be positive).
At step `i`, `start⋅baseⁱ` new observations are added.
"""
function logarithmic_schedule(n::Integer, base::Real, start::Integer)
    @assert 0 < n
    @assert 1 <= base
    @assert 0 < start <= n
    checkpoints = []
    block = 0
    total = 0
    i = 0
    while true
        block = start * base^i
        if n < total + block
            break
        end
        total += round(block)
        i = i +1
        push!(checkpoints, total)
    end
    remaining = n - checkpoints[end]
    @assert 0 <= remaining
    if remaining == 0
        return Vector{Integer}(checkpoints)
    else
        return Vector{Integer}(vcat(checkpoints, n))
    end
end

"""
    logarithmic_schedule(n::Integer, base::Real)
The total number of observations at step `i` is `baseⁱ`
"""
function logarithmic_schedule(n::Integer, base::Real)
    @assert 0 < n
    @assert 1 < base
    n < base && return [n]
    checkpoints = Integer[round(base^i) for i=1:floor(log(base, n))]
    remaining = n - checkpoints[end]
    push!(checkpoints, n)
    return checkpoints
end

export linear_schedule
export logarithmic_schedule

end # module Schedule
