using Test: @test

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors:
  FusionTensor,
  matrix,
  n_codomain_legs,
  codomain_axes,
  domain_axes,
  n_domain_legs,
  matrix_size,
  row_axis,
  column_axis,
  sanity_check
using NDTensors.GradedAxes
using NDTensors.Sectors: U1

g1 = GradedAxes.gradedrange([U1(0), U1(1), U1(2)], [1, 2, 3])
g2 = GradedAxes.gradedrange([U1(0), U1(1), U1(3)], [2, 2, 1])
m = BlockSparseArray{Float64}(g1, g2)

ft1 = FusionTensor((g1, g2), 1, m)  # default constructor

# getters
@test matrix(ft1) === m
@test axes(ft1) == (g1, g2)
@test n_codomain_legs(ft1) == 1

# misc
@test codomain_axes(ft1) == (g1,)
@test domain_axes(ft1) == (g2,)
@test n_domain_legs(ft1) == 1
@test matrix_size(ft1) == (6, 5)
@test row_axis(ft1) == g1
@test column_axis(ft1) == g2
@test isnothing(sanity_check(ft1))

# Base methods
@test eltype(ft1) === Float64
@test length(ft1) == 30
@test ndims(ft1) == 2
@test size(ft1) == (6, 5)

# copy
ft2 = copy(ft1)
@test matrix(ft2) == m
@test axes(ft2) == (g1, g2)
@test n_codomain_legs(ft2) == 1

# deepcopy
ft3 = deepcopy(ft1)
@test matrix(ft3) == m
@test axes(ft3) == (g1, g2)
@test n_codomain_legs(ft3) == 1

# more than 2 axes
g3 = GradedAxes.gradedrange([U1(-1), U1(0), U1(1)], [1, 1, 1])
g4 = GradedAxes.fuse(g2, g3)
m2 = BlockSparseArray{Float64}(g1, g4)
ft4 = FusionTensor((g1,), (g2, g3), m2)  # constructor from split axes

@test matrix(ft4) === m2
@test axes(ft4) == (g1, g2, g3)
@test n_codomain_legs(ft2) == 1

@test codomain_axes(ft4) == (g1,)
@test domain_axes(ft4) == (g2, g3)
@test n_domain_legs(ft4) == 2
@test matrix_size(ft4) == (6, 15)
@test row_axis(ft4) == g1
@test column_axis(ft4) == g4
@test isnothing(sanity_check(ft4))

@test ndims(ft4) == 3
@test size(ft4) == (6, 5, 3)
