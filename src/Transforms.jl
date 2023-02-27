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

"""Utilities for creating transformations."""
module Transforms

using Statistics: mean

abstract type Transform end
function apply(::Transform, x) end
function unapply(::Transform, x) end

function apply(transforms::Vector{T}, x) where T <: Transform
    return foldl((u, transform) -> apply(transform, u), transforms; init=x)
end

function unapply(transforms::Vector{T}, x) where T <: Transform
    return foldr((transform, u) -> unapply(transform, u), transforms; init=x)
end

# Linear Transformation.
struct LinearTransform <: Transform
    slope::Float64
    intercept::Float64
end
apply(t::LinearTransform, x) = @. t.slope * x + t.intercept
unapply(t::LinearTransform, x) = @. (x - t.intercept) / t.slope

unapply_mean(t::LinearTransform, mean) = unapply(t, mean)
unapply_var(t::LinearTransform, var) = @. (1/t.slope^2) * var

function unapply_mean_var(t::LinearTransform, mean, var)
    m = unapply_mean(t, mean)
    v = unapply_var(t, var)
    return (m, v)
end

"""
    LinearTransform(data::Vector{<:Real}, lo, hi)
Transform such that `minimum(data) = lo` and `maximum(data)=hi`.
"""
function LinearTransform(data::Vector{<:Real}, lo, hi)
    tnan = filter(!isnan, data)
    1 < length(tnan) || error("Cannot scale with <2 values.")
    tmin = minimum(tnan)
    tmax = maximum(tnan)
    a = hi - lo
    b = tmax - tmin
    slope = a / b
    intercept = -slope * tmin + lo
    LinearTransform(slope, intercept)
end

"""
    LinearTransform(data::Vector{<:Real}, width)
Transform such that `mean(data) = 0` and `data` is within `[-width, width]`.
"""
function LinearTransform(data::Vector{<:Real}, width)
    tnan = filter(!isnan, data)
    1 < length(tnan) || error("Cannot scale with <2 values.")
    xavg = mean(tnan)
    xmin = minimum(tnan)
    xmax = maximum(tnan)
    a = xmax - xmin
    slope = width / a
    intercept = -(width * xavg) / a
    return LinearTransform(slope, intercept)
end

# Log Transformation.
struct LogTransform <: Transform end
apply(t::LogTransform, x) = @. log(x)
unapply(t::LogTransform, x) = @. exp(x)
function unapply_mean_var(t::LogTransform, mean, var)
    m = @. exp(mean + var/2)
    v = @. (exp(var)-1)*exp(2*mean + var)
    return (m, v)
end

end # module Transforms
