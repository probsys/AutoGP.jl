# Copyright 2023 Google LLC
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
using Parameters
using Serialization

using AutoGP

function load_via_seralize(model::AutoGP.GPModel)
    dict = mktemp() do path, io
        serialize(path, Dict(model))
        return deserialize(path)
    end
    return AutoGP.GPModel(dict)
end

# function load_via_jld2(model::AutoGP.GPModel)
#     return mktempdir() do dir
#         path = joinpath(dir, "model.jld2")
#         save(path, "model", model)
#         load(path, "model")
#     end
# end

@testset "test_serialize" begin

    function check_model_same(model1::AutoGP.GPModel, model2::AutoGP.GPModel)
        @test model1.ds_transform  == model2.ds_transform
        @test model1.y_transform   == model2.y_transform
        @test model1.ds            == model2.ds
        @test model1.y             == model2.y
        @test type2dict(model1.config)        == type2dict(model2.config)
        kernels1 = AutoGP.covariance_kernels(model1)
        kernels2 = AutoGP.covariance_kernels(model2)
        @test all(kernels1 .≈ kernels2)
        noises1 = AutoGP.observation_noise_variances(model1)
        noises2 = AutoGP.observation_noise_variances(model2)
        @test all(isapprox.(noises1, noises2, rtol=1e-3)) # Why precision loss?
        weights1 = AutoGP.particle_weights(model1)
        weights2 = AutoGP.particle_weights(model2)
        @test all(isapprox.(weights1, weights2, atol=1e-4)) # Why precision loss?
    end

    # Initialize toy model.
    model1 = AutoGP.GPModel([Date("2025-01-01"), Date("2025-01-02")], [1.0, 2.0])
    AutoGP.fit_smc!(model1; n_mcmc=5, n_hmc=5, schedule=[2])

    # Write and load from disk.
    for model2 in [load_via_seralize(model1)]

        # Check initial models agree
        check_model_same(model1, model2)

        # Add data.
        AutoGP.add_data!(model1, [Date("2025-02-03")], [3.0]);
        AutoGP.add_data!(model2, [Date("2025-02-03")], [3.0]);
        check_model_same(model1, model2)

        # Remove data.
        AutoGP.remove_data!(model1, [Date("2025-01-01")])
        AutoGP.remove_data!(model2, [Date("2025-01-01")])
        check_model_same(model1, model2)

        # Infer with same seed.
        AutoGP.seed!(5)
        AutoGP.fit_smc!(model1; n_mcmc=5, n_hmc=5, schedule=[2])
        AutoGP.seed!(5)
        AutoGP.fit_smc!(model2; n_mcmc=5, n_hmc=5, schedule=[2])
        check_model_same(model1, model2)

    end

    # Ensure direct serialization fails.
    @test_throws ArgumentError serialize("model.autogp", model1)

end
