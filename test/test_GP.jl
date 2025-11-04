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
using Random

using AutoGP.GP.Distributions: cov
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

@testset "split_kernel_sop" begin
    l = GP.Linear(1)
    w = GP.WhiteNoise(1)
    p = GP.Periodic(1,1)
    g = GP.GammaExponential(1, 1)

    sentinel = GP.Constant(0)

    # Verify base kernels.
    for b in get_reparam_base_kernels()
      @test GP.split_kernel_sop(b, typeof(b)) == (b, sentinel)
      for j in get_reparam_base_kernels()
        if b != j
          @test GP.split_kernel_sop(b, typeof(j)) == (sentinel, b)
        end
      end
    end

    @test GP.split_kernel_sop(l*l + p*l + g*w, GP.Linear) == (l*l+p*l, g*w)
    @test GP.split_kernel_sop(l*(l+p+g), GP.Periodic) == (l*(l+p+g), sentinel)

    k = GP.ChangePoint(p*l+l, p*p+g, 1,1)
    @test GP.split_kernel_sop(k, GP.WhiteNoise) == (sentinel, k)
    @test GP.split_kernel_sop(k, GP.GammaExponential) == (
            GP.ChangePoint(sentinel, g, 1,1),
            GP.ChangePoint(p*l+l, p*p, 1, 1))

    k = GP.ChangePoint(l, p, 1, 1)
    @test GP.split_kernel_sop(k, GP.WhiteNoise) == (sentinel, k)
    @test GP.split_kernel_sop(k, GP.Linear) == (
            GP.ChangePoint(l, sentinel, 1,1),
            GP.ChangePoint(sentinel, p, 1, 1))

end # @testset "split_kernel_sop"

check_close(a,b) = all(isapprox.(a, b, atol=1e-5))

@testset "infer_gp_sum" begin
    T = 5
    ts_pred = Vector{Float64}(range(start=1, stop=13*T, step=1))
    ts_train = ts_pred[1:10*T]
    ts_test = ts_pred[10*T+1:end]
    noise = 0.01

    # Create test cases.
    kernels_base = get_reparam_base_kernels()
    kernels_single = [GP.Node[k] for k in kernels_base]
    permutations = Random.randperm.(length(kernels_single) .* ones(Int, 10))
    kernels_pairs = [
        GP.Node[
          reduce(*, kernels_base[p[1:2]]),
          reduce(*, kernels_base[p[1:2]]),
          kernels_base[p[end]]
          ]
        for p in permutations
      ]


    for ks in vcat(kernels_single, kernels_pairs)
      k = reduce(+, ks)

      # DIRECT GP.
      mvn1 = GP.Distributions.MvNormal(k, noise, Float64[], Float64[], ts_pred)
      xs = rand(mvn1)
      xs_train = xs[1:length(ts_train)]
      xs_test = t= xs[length(ts_train)+1:end]
      mvn1_cond = GP.Distributions.MvNormal(k, noise, ts_train, xs_train, ts_pred)

      C1 = cov(mvn1)
      C1_cond = cov(mvn1_cond)

      # CASE 1: GP SUM WITH SINGLE NODE.
      mvn2 = GP.infer_gp_sum(GP.Node[k], noise, Float64[], Float64[], ts_pred);
      mvn2_cond = GP.infer_gp_sum(GP.Node[k], noise, ts_train, xs_train, ts_pred);
      C2 = cov(mvn2.mvn)[mvn2.indexes.X, mvn2.indexes.X]
      C2_cond = cov(mvn2_cond.mvn)[mvn2_cond.indexes.X, mvn2_cond.indexes.X]

      # Confirm covariances match.
      @test check_close(C2, C1)
      @test check_close(C2_cond, C1_cond)

      # Confirm index dimensions match for mvn2.
      @test length(mvn2.indexes.F) == 1
      @test length(mvn2.indexes.X) == length(ts_pred)
      @test length(mvn2.indexes.F[1]) == length(ts_pred)

      # Confirm index dimensions match for mvn2_cond.
      @test length(mvn2_cond.indexes.F) == 1
      @test length(mvn2_cond.indexes.X) == length(ts_pred)
      @test length(mvn2_cond.indexes.F[1]) == length(ts_pred)

      # CASE 2: GP SUM WITH MULTIPLE NODE.
      mvn3 = GP.infer_gp_sum(ks, noise, Float64[], Float64[], ts_pred);
      mvn3_cond = GP.infer_gp_sum(ks, noise, ts_train, xs_train, ts_pred);
      C3 = cov(mvn3.mvn)[mvn3.indexes.X, mvn3.indexes.X]
      C3_cond = cov(mvn3_cond.mvn)[mvn3_cond.indexes.X, mvn3_cond.indexes.X]

      # Confirm index dimensions match for mvn3.
      @test length(mvn3.indexes.F) == length(ks)
      @test length(mvn3.indexes.X) == length(ts_pred)
      for i=1:length(ks)
        @test length(mvn3.indexes.F[i]) == length(ts_pred)
      end

      # Confirm index dimensions match for mvn3_cond.
      @test length(mvn3_cond.indexes.F) == length(ks)
      @test length(mvn3_cond.indexes.X) == length(ts_pred)
      for i=1:length(ks)
        @test length(mvn3_cond.indexes.F[i]) == length(ts_pred)
      end

      @test check_close(C3, C1)
      @test check_close(C3_cond, C1_cond)

    end

end
