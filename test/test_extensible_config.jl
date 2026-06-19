# Copyright 2025 CMU Probabilistic Computing Systems Lab
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

using Test
using Dates

using AutoGP
using AutoGP: GP
import AutoGP: Model
import AutoGP: Inference

using Gen

# ---------------------------------------------------------------------------
# Test-only primitive kernel fixture (NOT a shipped kernel). Behaves like a
# SquaredExponential so it is a well-behaved, positive-definite covariance.
# ---------------------------------------------------------------------------
struct DummyLeaf <: GP.LeafNode
    lengthscale::Real
    amplitude::Real
    DummyLeaf(lengthscale::Real, amplitude::Real=1) = new(lengthscale, amplitude)
end

GP.eval_cov(node::DummyLeaf, t1, t2) =
    node.amplitude * exp(-0.5 * (t1 - t2)^2 / node.lengthscale^2)

function GP.eval_cov(node::DummyLeaf, ts::Vector{Float64})
    dx = ts .- ts'
    return node.amplitude .* exp.(-0.5 .* dx .* dx ./ node.lengthscale^2)
end

GP.reparameterize(node::DummyLeaf, t::GP.LinearTransform) =
    DummyLeaf(node.lengthscale / abs(t.slope), node.amplitude)

GP.rescale(node::DummyLeaf, t::GP.LinearTransform) =
    DummyLeaf(node.lengthscale, t.slope^2 * node.amplitude)

# Extended config registering DummyLeaf at code 6 and shifting the operators
# to codes 7/8/9. Leaf mass is placed entirely on DummyLeaf so that fitted
# trees are guaranteed to contain it.
function extended_config(; changepoints=true)
    norm(v) = v ./ sum(v)
    return GP.GPConfig(
        index_to_node = Dict{Integer,Type{<:GP.Node}}(
            1 => GP.Constant,
            2 => GP.Linear,
            3 => GP.SquaredExponential,
            4 => GP.GammaExponential,
            5 => GP.Periodic,
            6 => DummyLeaf,
            7 => GP.Plus,
            8 => GP.Times,
            9 => GP.ChangePoint),
        Plus        = 7,
        Times       = 8,
        ChangePoint = 9,
        node_dist_leaf = norm([0., 0, 0, 0, 0, 1]),
        node_dist_nocp = norm([0., 0, 0, 0, 0, 8, 1, 1]),
        node_dist_cp   = norm([0., 0, 0, 0, 0, 8, 1, 1, 1]),
        changepoints   = changepoints,
    )
end

@testset "test_extensible_config" begin

    # -----------------------------------------------------------------------
    # Behaviour identity: the type-based leaf gate must route exactly the
    # same codes as the legacy hardcoded code list for the default config.
    # -----------------------------------------------------------------------
    @testset "leaf-gate routing unchanged for default config" begin
        config = GP.GPConfig()
        legacy_leaf_codes = [
            config.Constant,
            config.Linear,
            config.SquaredExponential,
            config.GammaExponential,
            config.Periodic]
        for (code, NodeType) in config.index_to_node
            @test (NodeType <: GP.LeafNode) == (code in legacy_leaf_codes)
        end
    end

    # node_to_integer (Edit 3) must return the same codes as before for every
    # built-in node type.
    @testset "node_to_integer round-trips built-in nodes" begin
        config = GP.GPConfig()
        @test Inference.node_to_integer(GP.Constant(0.5), config) == config.Constant
        @test Inference.node_to_integer(GP.Linear(0.1), config) == config.Linear
        @test Inference.node_to_integer(GP.SquaredExponential(0.5), config) == config.SquaredExponential
        @test Inference.node_to_integer(GP.GammaExponential(0.5, 1.0), config) == config.GammaExponential
        @test Inference.node_to_integer(GP.Periodic(0.5, 1.0), config) == config.Periodic
        @test Inference.node_to_integer(GP.Plus(GP.Constant(1.0), GP.Constant(1.0)), config) == config.Plus
        @test Inference.node_to_integer(GP.Times(GP.Constant(1.0), GP.Constant(1.0)), config) == config.Times
        @test Inference.node_to_integer(GP.ChangePoint(GP.Constant(1.0), GP.Constant(1.0), 0.5, 0.01), config) == config.ChangePoint
    end

    # The grammar still samples deterministically under a fixed seed.
    @testset "covariance_prior deterministic under fixed seed" begin
        config = GP.GPConfig()
        AutoGP.seed!(42)
        t1 = Gen.simulate(Model.covariance_prior, (1, config))
        AutoGP.seed!(42)
        t2 = Gen.simulate(Model.covariance_prior, (1, config))
        @test Gen.get_retval(t1) ≈ Gen.get_retval(t2)
        @test Gen.get_score(t1) == Gen.get_score(t2)
    end

    # -----------------------------------------------------------------------
    # Extensibility regression: a custom LeafNode can be proposed and used.
    # -----------------------------------------------------------------------
    @testset "fit_smc! proposes a custom LeafNode" begin
        config = extended_config()
        # Greedy's length-based leaf classification recognizes the new primitive.
        leaf_types = [config.index_to_node[c] for c in AutoGP.Greedy.get_leaf_node_types(config)]
        @test DummyLeaf in leaf_types

        ds = [Date(2025, 1, i) for i in 1:6]
        y = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2]
        AutoGP.seed!(7)
        model = AutoGP.GPModel(ds, y; n_particles=4, config=config)
        AutoGP.fit_smc!(model; n_mcmc=2, n_hmc=2, schedule=[3, 6])
        kernels = AutoGP.covariance_kernels(model)
        @test any(GP.has_leaf(k, DummyLeaf) for k in kernels)
    end

    @testset "decompose / predict_sum handle a custom LeafNode" begin
        config = extended_config(; changepoints=false)
        ds = [Date(2025, 1, i) for i in 1:6]
        y = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2]
        AutoGP.seed!(11)
        model = AutoGP.GPModel(ds, y; n_particles=2, config=config)
        AutoGP.fit_smc!(model; n_mcmc=2, n_hmc=2, schedule=[3, 6])

        # Both paths route through node_to_integer on a custom primitive (Edit 3).
        models = AutoGP.decompose(model)
        @test length(models) == AutoGP.num_particles(model)

        ds_future = [Date(2025, 1, i) for i in 7:9]
        predictions = AutoGP.predict_sum(model, ds_future, DummyLeaf)
        @test size(predictions, 1) > 0
    end

end
