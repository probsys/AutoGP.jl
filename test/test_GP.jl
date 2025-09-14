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

using Revise
using Test
using AutoGP

using AutoGP: Transforms
using AutoGP: GP

@testset "reparameterize" begin

    # Create raw and transformed time points.
    ds_raw = collect(range(start=-10, stop=10, length=100))
    transformation = Transforms.LinearTransform(ds_raw, 0, 1)
    ds = Transforms.apply(transformation, ds_raw)

    # Kernels are assumed to be defined on the scale of ds.
    base_kernels = [
        GP.WhiteNoise(1),
        GP.Constant(0.5),
        GP.Linear(0.1, 1.3, 0.7),
        GP.SquaredExponential(0.47, 0.13),
        GP.GammaExponential(0.42, 0.58, 3.2),
        GP.Periodic(0.96, 0.21, 1.1),
    ]

    # Base kernels.
    for b in base_kernels
        b_raw = GP.reparameterize(b, transformation)
        M1 = GP.eval_cov(b, ds)
        M2 = GP.eval_cov(b_raw, ds_raw)
        @test all(M1 .≈ M2)
    end

    # Composite kernels.
    for b1 in base_kernels
        for b2 in base_kernels
            for op in [+, *, (x, y) -> GP.ChangePoint(x, y, 0.5, 0.95)]
                b = op(b1, b2)
                b_raw = GP.reparameterize(b, transformation)
                M1 = GP.eval_cov(b, ds)
                M2 = GP.eval_cov(b_raw, ds_raw)
                @test all(M1 .≈ M2)
            end
        end
    end

end