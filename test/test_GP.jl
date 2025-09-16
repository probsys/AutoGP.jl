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
using AutoGP

using AutoGP: Transforms
using AutoGP: GP

function get_reparam_base_kernels()
  return [
    GP.WhiteNoise(1),
    GP.Constant(0.5),
    GP.Linear(0.1, 1.3, 0.7),
    GP.SquaredExponential(0.47, 0.13),
    GP.GammaExponential(0.42, 0.58, 3.2),
    GP.Periodic(0.96, 0.21, 1.1),
  ]
end

@testset "reparameterize" begin

    # Create raw and transformed time points.
    ds_raw = collect(range(start=-10, stop=10, length=100))
    transformation = Transforms.LinearTransform(ds_raw, 0, 1)
    ds = Transforms.apply(transformation, ds_raw)
    base_kernels = get_reparam_base_kernels()

    # Base kernels.
    for b in base_kernels
      @testset "base $(b)" begin
        b_raw = GP.reparameterize(b, transformation)
        M1 = GP.eval_cov(b, ds)
        M2 = GP.eval_cov(b_raw, ds_raw)
        @test all(M1 .≈ M2)
      end
    end

    # Composite kernels.
    for b1 in base_kernels
      for b2 in base_kernels
        for op in [+, *, (x, y) -> GP.ChangePoint(x, y, 0.5, 0.95)]
          @testset "composite $(b1) $(b2) $(op)" begin
              b = op(b1, b2)
              b_raw = GP.reparameterize(b, transformation)
              M1 = GP.eval_cov(b, ds)
              M2 = GP.eval_cov(b_raw, ds_raw)
              @test all(M1 .≈ M2)
          end
        end
      end
    end

end # @testset "reparameterize"

@testset "rescale" begin

    # Create time points.
    ds = collect(range(start=-10, stop=10, length=50))

    # Create raw and transformed y values.
    ys_raw = collect(range(start=-10, stop=10, length=50))
    transformation = Transforms.LinearTransform(ys_raw, -1, 1)
    ys = Transforms.apply(transformation, ys_raw)
    base_kernels = get_reparam_base_kernels()

    # Base kernels.
    for b in base_kernels
      @testset "base $(b)" begin
        b_rescale = GP.rescale(b, Transforms.invert(transformation))
        M1 = GP.eval_cov(b_rescale, ds)
        M2 = Transforms.unapply_var(transformation, GP.eval_cov(b, ds))
        @test all(isapprox.(M1, M2, atol=1e-10))
      end
    end

    # Composite kernels.
    for b1 in base_kernels
      for b2 in base_kernels
        for op in [+, *, (x, y) -> GP.ChangePoint(x, y, 0.5, 0.95)]
          @testset "composite $(b1) $(b2) $(op)" begin
            b = op(b1, b2)
            b_rescale = GP.rescale(b, Transforms.invert(transformation))
            M1 = GP.eval_cov(b_rescale, ds)
            M2 = Transforms.unapply_var(transformation, GP.eval_cov(b, ds))
            @test all(isapprox.(M1, M2, atol=1e-8))
          end
        end
      end
    end

end # @testset "rescale"
