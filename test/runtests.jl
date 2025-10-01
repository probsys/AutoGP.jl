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

@testset "AutoGP" begin
    @testset "test_GP.jl"        begin include("test_GP.jl") end
    @testset "test_api.jl"       begin include("test_api.jl") end
    @testset "test_serialize.jl" begin include("test_serialize.jl") end
end
