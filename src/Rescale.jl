module Rescale

import Statistics: mean

"""Scaling data"""
abstract type Scaler end
function transform end
function untransform end

function fit_transform(c::Type{T}, t) where T <: Scaler
    scaler = c(t)
    u = transform(scaler, t)
    return (scaler, u)
end


"""Linear scaling, usually for time."""
struct LinearScaler <: Scaler
    lo::Float64
    hi::Float64
    tmin::Float64
    tmax::Float64
    function LinearScaler(t, lo, hi)
        tnan = filter(!isnan, t)
        1 < length(tnan) || error("Cannot scale with <2 values.")
        tmin = minimum(tnan)
        tmax = maximum(tnan)
        return new(lo, hi, tmin, tmax)
    end
    LinearScaler(t) = LinearScaler(t, 0., 1.)
end

function transform(scaler::LinearScaler, t)
    u = (t .- scaler.tmin) ./ (scaler.tmax .- scaler.tmin)
    s = (scaler.hi - scaler.lo) .* u .+ scaler.lo
    return s
end

function untransform(scaler::LinearScaler, s)
    u = (s .- scaler.lo) ./ (scaler.hi - scaler.lo)
    t = u .* (scaler.tmax - scaler.tmin) .+ scaler.tmin
    return t
end

"""Mean scaling, usually for observations."""
struct MeanScaler <: Scaler
    width::Float64
    xavg::Float64
    xmin::Float64
    xmax::Float64
    function MeanScaler(t, width)
        tnan = filter(!isnan, t)
        1 < length(tnan) || error("Cannot scale with <2 values.")
        xavg = mean(tnan)
        xmin = minimum(tnan)
        xmax = maximum(tnan)
        return new(width, xavg, xmin, xmax)
    end
    MeanScaler(t) = MeanScaler(t, 1.)
end

function transform(scaler::MeanScaler, t)
    u = t .- scaler.xavg
    v = u ./ (scaler.xmax - scaler.xmin)
    s = scaler.width .* v
    return s
end

function untransform(scaler::MeanScaler, s)
    v = s ./ scaler.width
    u = (scaler.xmax - scaler.xmin) .* v
    t = u .+ scaler.xavg
    return t
end

end # module Rescale
