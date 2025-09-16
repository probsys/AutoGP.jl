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

"""Module for Gaussian process modeling library."""
module GP

import ..Transforms: LinearTransform

import Statistics
import Distributions
import LinearAlgebra

using DocStringExtensions
using Parameters: @with_kw
using Printf: @sprintf

import Gen

"""
    abstract type Node end
Abstract class for [covariance kernels](@ref gp_cov_kernel).
"""
abstract type Node end

"""
    abstract type LeafNode <: Node end
Abstract class for [primitive covariance kernels](@ref gp_cov_kernel_prim).
"""
abstract type LeafNode <: Node end

"""
    abstract type BinaryOpNode <: Node end
Abstract class for [composite covariance kernels](@ref (@ref gp_cov_kernel_comp).
"""
abstract type BinaryOpNode <: Node end

"""
    eval_cov(node::Node, t1::Real, t2::Real)
    eval_cov(node::Node, ts::Vector{Float64})

Evaluate the covariance function `node` at the given time indexes.
The first form returns a `Real` number and the second form returns a
covariance `Matrix`.
"""
function eval_cov end

"""
    reparameterize(node::Node, t::LinearTransform)

Reparameterize the covariance kernel according to the given
[`LinearTransform`](@ref) applied to the input (known as an "input warping").
For a kernel ``k(\\cdot,\\cdot; \\theta)`` and a linear
transform ``f(t) = at+b`` over the time domain, this function
returns a kernel with new parameters ``\\theta'`` such that
``k(at+b, au+b; \\theta) = k(t, u; \\theta')``.
"""
function reparameterize end

"""
    rescale(node::Node, t::LinearTransform)

Rescale the covariance kernel according to the given
[`LinearTransform`](@ref) applied to the output.
In particular, for a GP ``X \\sim \\mathrm{GP}(0, k(\\cdot,\\cdot; \\theta))`` and a
transformation ``Y = aX + b``, this function
returns a kernel with new parameters ``\\theta'``
such that ``Y \\sim \\mathrm{GP}(b, k(\\cdot,\\cdot; \\theta'))``.
"""
function rescale end

"""
    Base.size(node::Node)
    Base.size(node::LeafNode) = 1
    Base.size(a::LeafNode, b::Node) = size(a) + size(b)
Return the total number of subexpressions in a [`Node`](@ref), as defined above.
"""
Base.size(::Node) = error("Not implemented")
Base.size(::LeafNode) = 1
Base.size(node::BinaryOpNode) = node.size
Base.isapprox(a::Node, b::Node) =
    (typeof(a) == typeof(b)) &&
        all(isapprox(
            getfield(a, f),
            getfield(b, f))
        for f in fieldnames(typeof(a)))

depth(::LeafNode) = 1
depth(node::BinaryOpNode) = node.depth

"""
    unroll(node::Node)

Unroll a covariance kernel into a flat Vector of all intermediate kernels.
"""
function unroll end
unroll(node::LeafNode) = [node]
unroll(node::BinaryOpNode) = vcat(unroll(node.left), unroll(node.right), node)


@doc raw"""

    WhiteNoise(value)

White noise covariance kernel.

```math
k(t, t') = \mathbf{I}[t = t'] \theta
```

The random variables ``X[t]`` and ``X[t']`` are perfectly correlated
whenever ``t = t'`` and independent otherwise. This kernel cannot be used
to represent the joint distribution of multiple i.i.d. measurements of ``X[t]``,
instead see [`compute_cov_matrix_vectorized`](@ref).
"""
struct WhiteNoise <: LeafNode
    value::Real
end

eval_cov(node::WhiteNoise, t1, t2) = (t1==t2) * node.value

function eval_cov(node::WhiteNoise, ts::Vector{Float64})
    n = length(ts)
    return (ts .== ts') * node.value
end

reparameterize(node::WhiteNoise, t::LinearTransform) = node;
rescale(node::WhiteNoise, t::LinearTransform) = WhiteNoise(t.slope^2 * node.value);

@doc raw"""
    Constant(value)

Constant covariance kernel.

```math
k(t,t') = \theta
```

Draws from this kernel are horizontal lines, where ``\theta`` determines
the variance of the constant value around the mean (typically zero).
"""
struct Constant <: LeafNode
    value::Real
end

eval_cov(node::Constant, t1, t2) = node.value

function eval_cov(node::Constant, ts::Vector{Float64})
    n = length(ts)
    return node.value .* LinearAlgebra.ones(n, n)
end

reparameterize(node::Constant, t::LinearTransform) = node;
rescale(node::Constant, t::LinearTransform) = Constant(t.slope^2 * node.value);

@doc raw"""
    Linear(intercept[, bias=1, amplitude=1])

Linear covariance kernel.

```math
k(t, t') = \theta_2 + \theta_3 (t - \theta_1)(t'-\theta_1)
```

Draws from this kernel are sloped lines in the 2D plane.
The time intercept is ``\theta_1``.
The variance around the time intercept is ``\theta_2``.
The scale factor, which dictates the slope, is ``\theta_3``.
"""
struct Linear <: LeafNode
    intercept::Real
    bias::Real
    amplitude::Real
    function Linear(intercept::Real, bias::Real=1, amplitude::Real=1)
        return new(intercept, bias, amplitude)
    end
end

function eval_cov(node::Linear, t1, t2)
    c = (t1 - node.intercept) * (t2 - node.intercept)
    return node.bias + node.amplitude * c
end

function eval_cov(node::Linear, ts::Vector{Float64})
    ts_minus_intercept = ts .- node.intercept
    C = ts_minus_intercept * ts_minus_intercept'
    return node.bias .+ node.amplitude * C
end

function reparameterize(node::Linear, t::LinearTransform)
    intercept = (node.intercept - t.intercept) / t.slope
    amplitude = t.slope^2 * node.amplitude
    return Linear(intercept, node.bias, amplitude)
end

function rescale(node::Linear, t::LinearTransform)
    bias = t.slope^2 * node.bias
    amplitude = t.slope^2 * node.amplitude
    return Linear(node.intercept, bias, amplitude);
end

@doc raw"""
    SquaredExponential(lengthscale[, amplitude=1])

Squared Exponential covariance kernel.

```math
k(t,t') = \theta_2 \exp\left(-1/2|t-t'|/\theta_2)^2 \right)
```

Draws from this kernel are smooth functions.
"""
struct SquaredExponential <: LeafNode
    lengthscale::Real
    amplitude::Real
    function SquaredExponential(lengthscale::Real, amplitude::Real=1)
        return new(lengthscale, amplitude)
    end
end

function eval_cov(node::SquaredExponential, t1, t2)
    c = exp(-.5 * (t1 - t2) * (t1 - t2) / node.lengthscale^2)
    return node.amplitude * c
end

function eval_cov(node::SquaredExponential, ts::Vector{Float64})
    dx = ts .- ts'
    C = exp.(-.5 .* dx .* dx ./ node.lengthscale^2)
    return node.amplitude * C
end

function reparameterize(node::SquaredExponential, t::LinearTransform)
    lengthscale = node.lengthscale / abs(t.slope)
    return SquaredExponential(lengthscale, node.amplitude)
end

function rescale(node::SquaredExponential, t::LinearTransform)
    amplitude = t.slope^2 * node.amplitude
    return SquaredExponential(node.lengthscale, amplitude)
end

@doc raw"""
    GammaExponential(lengthscale, gamma[, amplitude=1])

Gamma Exponential covariance kernel.

```math
k(t,t') = \theta_3 \exp(-(|t-t'|/\theta_1)^{\theta_2})
```

Requires `0 < gamma <= 2`.
Recovers the [`SquaredExponential`](@ref) kernel when `gamma = 2`.
"""
struct GammaExponential <: LeafNode
    lengthscale::Real
    gamma::Real
    amplitude::Real
    function GammaExponential(lengthscale::Real, gamma::Real, amplitude::Real=1)
        @assert (0 < gamma <= 2)
        return new(lengthscale, gamma, amplitude)
    end
end

function eval_cov(node::GammaExponential, t1, t2)
    dt = abs(t1 - t2)
    c = exp(- (dt / node.lengthscale) ^ node.gamma)
    return node.amplitude * c
end

function eval_cov(node::GammaExponential, ts::Vector{Float64})
    dt = abs.(ts .- ts')
    C = exp.(- (dt ./ node.lengthscale) .^ node.gamma)
    return node.amplitude * C
end

function reparameterize(node::GammaExponential, t::LinearTransform)
    lengthscale = node.lengthscale / (abs(t.slope))
    return GammaExponential(lengthscale, node.gamma, node.amplitude)
end

function rescale(node::GammaExponential, t::LinearTransform)
    amplitude = t.slope^2 * node.amplitude
    return GammaExponential(node.lengthscale, node.gamma, amplitude)
end

@doc raw"""
    Periodic(lengthscale, period[, amplitude=1])

Periodic covariance kernel.

```math
k(t,t') = \exp\left( (-2/\theta_1^2) \sin^2((\pi/\theta_2) |t-t'|) \right)
```

The lengthscale determines how smooth the periodic function is within each period.
Heuristically, the periodic kernel can be understood as:
1. Sampling ``[X(t), t \in [0,p]] \sim \mathrm{GP}(0, \mathrm{SE}(\theta_1))``.
2. Repeating this fragment for all intervals ``[jp, (j+1)p], j \in \mathbb{Z}``.
"""
struct Periodic <: LeafNode
    lengthscale::Real
    period::Real
    amplitude::Real
    function Periodic(lengthscale::Real, period::Real, amplitude::Real=1)
        return new(lengthscale, period, amplitude)
    end
end

function eval_cov(node::Periodic, t1, t2)
    freq = pi / node.period
    dx = abs(t1 - t2)
    c = exp((-2/node.lengthscale^2) * (sin(freq * dx))^2)
    return node.amplitude * c
end

function eval_cov(node::Periodic, ts::Vector{Float64})
    freq = pi / node.period
    dx = abs.(ts .- ts')
    C = exp.((-2/node.lengthscale^2) .* (sin.(freq .* dx)).^2)
    return node.amplitude * C
end

function reparameterize(node::Periodic, t::LinearTransform)
    period = node.period / abs(t.slope)
    return Periodic(node.lengthscale, period, node.amplitude)
end

function rescale(node::Periodic, t::LinearTransform)
    amplitude = t.slope^2 * node.amplitude
    return Periodic(node.lengthscale, node.period, amplitude)
end

@doc raw"""
    Plus(left::Node, right::Node)
    Base.:+(left::Node, right::Node)

Covariance kernel obtained by summing two covariance kernels pointwise.

```math
k(t,t') = k_{\rm left}(t,t') + k_{\rm right}(t,t')
```
"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
    depth::Int
end

function Plus(left, right)
    s = 1 + size(left) + size(right)
    d = 1 + maximum((depth(left), depth(right)))
    return Plus(left, right, s, d)
end

function eval_cov(node::Plus, t1, t2)
    return eval_cov(node.left, t1, t2) + eval_cov(node.right, t1, t2)
end

function eval_cov(node::Plus, ts::Vector{Float64})
    return eval_cov(node.left, ts) .+ eval_cov(node.right, ts)
end

Base.:+(a::Node, b::Node) = Plus(a, b)
Base.:*(a::Node, b::Node) = Times(a, b)

function reparameterize(node::Plus, t::LinearTransform)
    left = reparameterize(node.left, t)
    right = reparameterize(node.right, t)
    return left + right
end

function rescale(node::Plus, t::LinearTransform)
    left = rescale(node.left, t)
    right = rescale(node.right, t)
    return left + right
end

@doc raw"""
    Times(left::Node, right::Node)
    Base.:*(left::Node, right::Node)

Covariance kernel obtained by multiplying two covariance kernels pointwise.

```math
k(t,t') = k_{\rm left}(t,t') \times k_{\rm right}(t,t')
```
"""
struct Times <: BinaryOpNode
    left::Node
    right::Node
    size::Int
    depth::Int
end

function Times(left, right)
    s = 1 + size(left) + size(right)
    d = 1 + maximum((depth(left), depth(right)))
    return Times(left, right, s, d)
end

function eval_cov(node::Times, t1, t2)
    return eval_cov(node.left, t1, t2) * eval_cov(node.right, t1, t2)
end

function eval_cov(node::Times, ts::Vector{Float64})
    return eval_cov(node.left, ts) .* eval_cov(node.right, ts)
end

function reparameterize(node::Times, t::LinearTransform)
    left = reparameterize(node.left, t)
    right = reparameterize(node.right, t)
    return left * right
end

function rescale(node::Times, t::LinearTransform)
    # Only rescale one of the two kernels.
    left = rescale(node.left, t)
    right = node.right
    return left * right
end

@doc raw"""
    ChangePoint(left::Node, right::Node, location::Real, scale::Real)

Covariance kernel obtained by switching between two kernels at `location`.

```math
\begin{aligned}
k(t,t') &= [\sigma_1 \cdot k_{\rm left}(t, t') \cdot \sigma_2] + [(1 - \sigma_1) \cdot k_{\rm right}(t, t') \cdot (1-\sigma_2)] \\
\mathrm{where}\,
\sigma_1 &= (1 + \tanh((t - \theta_1) / \theta_2))/2, \\
\sigma_2 &= (1 + \tanh((t' - \theta_1) / \theta_2))/2.
\end{aligned}
```

The `location` parameter ``\theta_1`` denotes the time point at which
the change occurs.  The `scale` parameter ``\theta_2`` is a nonnegative
number that controls the rate of change; its behavior can be understood
by analyzing the two extreme values:
- If `location=0` then ``k_{\rm left}`` is active and ``k_{\rm right}`` is inactive
  for all times less than `location`; ``k_{\rm right}`` is active and ``k_{\rm left}`` is
  inactive for all times greater than `location`; and ``X[t] \perp X[t']`` for
  all ``t`` and ``t'`` on opposite sides of `location`.

- If `location=Inf` then ``k_{\rm left}`` and ``k_{\rm right}`` have
  equal effect for all time points,
  and ``k(t,t') = 1/2 (k_{\rm left}(t,'t) + k_{\rm right}(t,t'))``,
  which is equivalent to a [`Plus`](@ref) kernel scaled by a factor of ``1/2``.
"""
struct ChangePoint <: BinaryOpNode
    left::Node
    right::Node
    location::Real
    scale::Real
    size::Int
    depth::Int
end

function ChangePoint(left, right, location, scale)
    s = 1 + size(left) + size(right)
    d = 1 + maximum((depth(left), depth(right)))
    return ChangePoint(left, right, location, scale, s, d)
end

function sigma_cp(x::Float64, location, scale)
    return .5 * (1 + tanh((location - x) / scale))
end

function eval_cov(node::ChangePoint, t1, t2)
    sigma_t1 = sigma_cp(t1, node.location, node.scale)
    sigma_t2 = sigma_cp(t2, node.location, node.scale)
    k_left = sigma_t1 * eval_cov(node.left, t1, t2) * sigma_t2
    k_right = (1 - sigma_t1) * eval_cov(node.right, t1, t2) * (1 - sigma_t2)
    return k_left + k_right
end

function eval_cov(node::ChangePoint, ts::Vector{Float64})
    change_x = sigma_cp.(ts, node.location, node.scale)
    sig_1 = change_x * change_x'
    sig_2 = (1 .- change_x) * (1 .- change_x')
    k_1 = eval_cov(node.left, ts)
    k_2 = eval_cov(node.right, ts)
    # TODO: Returning Symmetric makes code non-differentiable.
    # return LinearAlgebra.Symmetric(sig_1 .* k_1 + sig_2 .* k_2)
    K = sig_1 .* k_1 + sig_2 .* k_2
    return Matrix(LinearAlgebra.Symmetric(K))
end

function reparameterize(node::ChangePoint, t::LinearTransform)
    left = reparameterize(node.left, t)
    right = reparameterize(node.right, t)
    location = (node.location - t.intercept) / t.slope
    scale = node.scale / t.slope
    return ChangePoint(left, right, location, scale)
end

function rescale(node::ChangePoint, t::LinearTransform)
    left = rescale(node.left, t)
    right = rescale(node.right, t)
    return ChangePoint(left, right, node.location, node.scale)
end

"""
    compute_cov_matrix_vectorized(node::Node, noise, ts)
Compute covariance matrix by evaluating `node` on all pair of `ts`.
The `noise` is added to the diagonal of the covariance matrix, which
means that if `ts[i] == ts[j]`, then `X[ts[i]]` and `Xs[ts[j]]` are
i.i.d. samples of the true function at `ts[i]` plus mean zero
Gaussian noise.
"""
function compute_cov_matrix_vectorized(node::Node, noise, ts)
    return eval_cov(node, ts) + (noise * LinearAlgebra.I)
end

"""
    compute_cov_matrix(node::Node, noise, ts)
Non-vectorized implementation of [`compute_cov_matrix_vectorized`](@ref).
"""
function compute_cov_matrix(node::Node, noise, ts)
    n = length(ts)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i,j] = eval_cov(node, ts[i], ts[j])
        end
        cov_matrix[i,i] += noise
    end
    return cov_matrix
end

"""
    dist = Distributions.MvNormal(
            node::Node,
            noise::Float64,
            ts::Vector{Float64},
            xs::Vector{Float64},
            ts_pred::Vector{Float64};
            noise_pred::Union{Nothing,Float64}=nothing)

Return [`MvNormal`](https://juliastats.org/Distributions.jl/stable/multivariate/#Distributions.MvNormal)
posterior predictive distribution over `xs_pred` at time indexes `ts_pred`,
given noisy observations `[ts, xs]` and covariance function `node` with
given level of observation `noise`.

By default, the observation noise (`noise_pred`) of the new data is equal to
the `noise` of the observed data; use `noise_pred = 0.` to obtain the
predictive distribution over noiseless future values.

# See also
- To compute log probabilities, [`Distributions.logpdf`](https://juliastats.org/Distributions.jl/v0.24/multivariate/#Distributions.logpdf-Tuple{MultivariateDistribution{S}%20where%20S%3C:ValueSupport,%20AbstractArray})
- To generate samples, [`Base.rand`](https://juliastats.org/Distributions.jl/stable/multivariate/#Base.rand-Tuple{AbstractRNG,%20MultivariateDistribution})
- To compute quantiles, [`Distributions.quantile`](@ref)
"""
function Distributions.MvNormal(
        node::Node,
        noise::Float64,
        ts::Vector{Float64},
        xs::Vector{Float64},
        ts_pred::Vector{Float64};
        noise_pred::Union{Nothing,Float64}=nothing,
        mean::Function=t->0.)
    noise_pred = isnothing(noise_pred) ? noise : noise_pred
    n_prev = length(ts)
    n_new = length(ts_pred)
    means = map(mean, vcat(ts, ts_pred))
    cov_matrix = compute_cov_matrix_vectorized(node, 0., vcat(ts, ts_pred))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev] + (noise * LinearAlgebra.I)
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    # @assert cov_matrix_12 == cov_matrix_21'
    # CP kernel gives approximate symmetry due to tanh
    @assert isapprox(cov_matrix_12, cov_matrix_21')
    mu2 = means[n_prev+1:n_prev+n_new]
    mu1 = means[1:n_prev]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (xs - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = .5 * conditional_cov_matrix + .5 * conditional_cov_matrix'
    conditional_cov_matrix = conditional_cov_matrix + (noise_pred * LinearAlgebra.I)
    return Distributions.MvNormal(conditional_mu, conditional_cov_matrix)
end

"""
    Distributions.quantile(dist::Distributions.MvNormal, p)
Compute quantiles of marginal distributions of `dist`.

# Examples
```example
Distributions.quantile(Distributions.MvNormal([0,1,2,3], LinearAlgebra.I(4)), .5)
Distributions.quantile(Distributions.MvNormal([0,1,2,3], LinearAlgebra.I(4)), [[.1, .5, .9]])
```
"""
function Distributions.quantile(dist::Distributions.MvNormal, p)
    mu = Distributions.mean(dist)
    cov = Distributions.cov(dist)
    std = sqrt.(LinearAlgebra.diag(cov))
    x = Distributions.quantile.(Distributions.Normal.(mu, std), p)
    return hcat(x...)'
end

"""
    pretty(node::Node)
Return a pretty `String` representation of `node`.
"""
function pretty end
pretty(node::WhiteNoise) = @sprintf("WN(%1.2f)", node.value)
pretty(node::Constant) = @sprintf("CONST(%1.2f)", node.value)
pretty(node::Linear) = @sprintf("LIN(%1.2f; %1.2f, %1.2f)", node.intercept, node.bias, node.amplitude)
pretty(node::SquaredExponential) = @sprintf("SE(%1.2f; %1.2f)", node.lengthscale, node.amplitude)
pretty(node::GammaExponential) = @sprintf("GE(%1.2f, %1.2f; %1.2f)", node.lengthscale, node.gamma, node.amplitude)
pretty(node::Periodic) = @sprintf("PER(%1.2f, %1.2f; %1.2f)", node.lengthscale, node.period, node.amplitude)
pretty(node::Plus) = "($(pretty(node.left)) + $(pretty(node.right)))"
pretty(node::Times) = "($(pretty(node.left)) * $(pretty(node.right)))"
pretty(node::ChangePoint) = "CP($(pretty(node.left)), $(pretty(node.right)), $(node.location), $(node.scale))"

function _make_indent_strings(pre, vert_bars, first, last)
    first && return ("", "")
    VERT = '\u2502'
    PLUS = '\u251C'
    HORZ = '\u2500'
    LAST = '\u2514'
    # Prepare indentation.
    indent_vert = repeat([' '], pre+2)
    indent      = repeat([' '], pre+4)
    for i in vert_bars
        indent_vert[i]  = VERT
        indent[i]       = VERT
    end
    indent_vert[end-1:end] = [VERT, '\n']
    indent[end-3:end]      = [(last ? LAST : PLUS), HORZ, HORZ, ' ']
    # Make strings.
    indent_vert_str = join(indent_vert)
    indent_str = join(indent)
    return (indent_vert_str, indent_str)
end

function _show_pretty(io::IO, node::LeafNode, pre, vert_bars::Tuple; first=false, last=true)
    indent_vert_str, indent_str = _make_indent_strings(pre, vert_bars, first, last)
    # print(io, indent_vert_str)
    print(io, indent_str * "$(pretty(node))\n")
end

_pretty_BinaryOpNode(node::Plus) = '+'
_pretty_BinaryOpNode(node::Times) = '\u00D7'
_pretty_BinaryOpNode(node::ChangePoint) = ("CP$((node.location,node.scale))")
function _show_pretty(io::IO, node::BinaryOpNode, pre, vert_bars::Tuple; first=false, last=true)
    indent_vert_str, indent_str = _make_indent_strings(pre, vert_bars, first, last)
    # print(io, indent_vert_str)
    print(io, indent_str * "$(_pretty_BinaryOpNode(node))\n")
    vert_bars_left = (typeof(node.left) <: LeafNode) ? vert_bars : (vert_bars..., (first ? 1 : pre+5))
    _show_pretty(io, node.left, (first ? 0 : pre + 4), vert_bars_left; last=false)
    _show_pretty(io, node.right, (first ? 0 : pre + 4), (vert_bars...,); last=true)
end

function Base.show(io::IO, ::MIME"text/plain", node::Node)
    _show_pretty(io, node, 0, (); first=true)
end

# Normalize array to obtain probabilities.
normalize(x::Vector{Float64}) = x ./ sum(x)

"""
    config = GPConfig(kwargs...)

Configuration of prior distribution over Gaussian process kernels, i.e.,
an instance of [`Node`](@ref). The main `kwargs` (all optional) are:

- `node_dist_leaf::Vector{Real}`: Prior distribution over
  [`LeafNode`](@ref) kernels; default is uniform.

- `node_dist_nocp::Vector{Real}`: Prior distribution over
  [`BinaryOpNode`](@ref) kernels; only used if `changepoints=false`.

- `node_dist_cp::Vector{Real}`: Prior distribution over
  [`BinaryOpNode`](@ref) kernels; only used if `changepoints=true`.

- `max_depth::Integer`: Maximum depth of covariance node; default is `-1`
  for unbounded.

- `changepoints::Bool`: Whether to permit [`ChangePoint`](@ref)
  compositions; default is `true`.

- `noise::Union{Nothing,Float64}`: Whether to use a fixed observation
  noise; default is `nothing` to infer automatically.
"""
@with_kw struct GPConfig
    # Assignment of integer code to each node type.
    Constant::Integer           = 1
    Linear::Integer             = 2
    SquaredExponential::Integer = 3
    GammaExponential::Integer   = 4
    Periodic::Integer           = 5
    Plus::Integer               = 6
    Times::Integer              = 7
    ChangePoint::Integer        = 8
    # Mapping from integer code to node types.
    index_to_node::Dict{Integer,Type{<:Node}} = Dict(
        Constant           => GP.Constant,
        Linear             => GP.Linear,
        SquaredExponential => GP.SquaredExponential,
        GammaExponential   => GP.GammaExponential,
        Periodic           => GP.Periodic,
        Plus               => GP.Plus,
        Times              => GP.Times,
        ChangePoint        => GP.ChangePoint,
    )
    # Distribution over base and composite kernels.
    node_dist_leaf::Vector{Float64} = normalize([0., 1, 0, 1, 1,])
    node_dist_nocp::Vector{Float64} = normalize([0., 6, 0, 6, 6, 5, 5])
    node_dist_cp::Vector{Float64}   = normalize([0., 6, 0, 6, 6, 4, 4, 2])
    # Maximum number of children of composite node.
    max_branch::Integer = 2
    # Maximum depth of a covariance kernel tree.
    max_depth::Integer = -1
    # Enable ChangePoints.
    changepoints::Bool = true
    # Observation noise level.
    noise::Union{Nothing,Float64} = nothing
end

# Convert index of a node to its depth in tree.
idx_to_depth(idx::Int) = 1 + floor(log2(idx))

export Node
export LeafNode
export BinaryOpNode
export eval_cov
export compute_cov_matrix
export compute_cov_matrix_vectorized
export unroll

export WhiteNoise
export Constant
export Linear
export SquaredExponential
export GammaExponential
export Periodic

export Times
export Plus
export ChangePoint

export GPConfig

end # module GP
