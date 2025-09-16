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

using Random: randn

@testset "reparameterize" begin

  # Create raw data.
  num_train = 50
  y_train = randn(num_train)
  ds_train = collect(range(start=-10, stop=10, length=num_train))
  ds_pred = collect(range(start=10, stop=15, length=num_train))
  ds_query = vcat(ds_train, ds_pred);

  # Build the model.
  model = AutoGP.GPModel(ds_train, y_train; n_particles=4)

  # Obtain transformed data.
  ds_train_tr = AutoGP.Transforms.apply(model.ds_transform, ds_train)
  ds_pred_tr = AutoGP.Transforms.apply(model.ds_transform, ds_pred)
  ds_query_tr = vcat(ds_train_tr, ds_pred_tr)

  # Compute y average.
  y_train_avg = sum(y_train) / length(y_train)

  # Obtain variance in the raw space.
  noises = AutoGP.observation_noise_variances(model);
  noises_tr = AutoGP.observation_noise_variances(model, reparameterize=false);

  # Obtain kernels in raw and transformed space.
  kernels = AutoGP.covariance_kernels(model)
  kernels_tr = AutoGP.covariance_kernels(model; reparameterize=false);

  # Confirm the covariance kernels agree.
  for ((kr, nr), (kt,nt)) in zip(zip(kernels, noises), zip(kernels_tr, noises_tr))
    C1 = AutoGP.GP.compute_cov_matrix_vectorized(kr, nr, ds_query)
    C2 = Transforms.unapply_var(
      model.y_transform,
      AutoGP.GP.compute_cov_matrix_vectorized(kt, nt, ds_query_tr))
    @test all(isapprox.(C1, C2, atol=1e-8))
  end

  # Obtain MVN in the transformed space.
  mvn_tr = AutoGP.predict_mvn(model, ds_query)
  for (i, (kr, nr)) in enumerate(zip(kernels, noises))
    m = AutoGP.GP.Distributions.MvNormal(
      kr, nr, ds_train, y_train, ds_query;
      mean=t->y_train_avg)
    @test all(isapprox.(m.μ, mvn_tr.components[i].μ))
    @test all(isapprox.(m.Σ, mvn_tr.components[i].Σ))
  end

end
