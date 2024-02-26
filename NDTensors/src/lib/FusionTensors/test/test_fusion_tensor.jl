using Test: @test

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors:
  FusionTensor,
  codomain_axes,
  column_axis,
  domain_axes,
  matrix,
  matrix_size,
  n_codomain_axes,
  n_domain_axes,
  row_axis,
  sanity_check
using NDTensors.GradedAxes
using NDTensors.Sectors: U1

g1 = GradedAxes.gradedrange([U1(0), U1(1), U1(2)], [1, 2, 3])
g2 = GradedAxes.gradedrange([U1(0), U1(1), U1(3)], [2, 2, 1])
m = BlockSparseArray{Float64}(g1, g2)

ft1 = FusionTensor((g1,), (g2,), m)  # constructor from split axes

# getters
@test matrix(ft1) === m
@test axes(ft1) == (g1, g2)
@test n_codomain_axes(ft1) == 1

# misc
@test codomain_axes(ft1) == (g1,)
@test domain_axes(ft1) == (g2,)
@test n_domain_axes(ft1) == 1
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
@test n_codomain_axes(ft2) == 1

# deepcopy
ft3 = deepcopy(ft1)
@test matrix(ft3) == m
@test axes(ft3) == (g1, g2)
@test n_codomain_axes(ft3) == 1

# more than 2 axes
g3 = GradedAxes.gradedrange([U1(-1), U1(0), U1(1)], [1, 2, 1])
g4 = GradedAxes.gradedrange([U1(-1), U1(0), U1(1)], [1, 1, 1])
gr = GradedAxes.fuse(g1, g2)
gc = GradedAxes.fuse(g3, g4)
m2 = BlockSparseArray{Float64}(gr, gc)
ft4 = FusionTensor((g1, g2), (g3, g4), m2)  # constructor from concatenated axes

@test matrix(ft4) === m2
@test axes(ft4) == (g1, g2, g3, g4)
@test n_codomain_axes(ft4) == 2

@test codomain_axes(ft4) == (g1, g2)
@test domain_axes(ft4) == (g3, g4)
@test n_domain_axes(ft4) == 2
@test matrix_size(ft4) == (30, 12)
@test row_axis(ft4) == gr
@test column_axis(ft4) == gc
@test isnothing(sanity_check(ft4))

@test ndims(ft4) == 4
@test size(ft4) == (6, 5, 4, 3)

# test cast from and to dense
arr = zeros((6, 5, 4, 3))
ft5 = FusionTensor((g1, g2), (g3, g4), arr)
@test axes(ft5) == (g1, g2, g3, g4)
@test n_codomain_axes(ft5) == 2
@test isnothing(sanity_check(ft5))

ft6 = FusionTensor{2}((g1, g2, g3, g4), arr)
@test axes(ft6) == (g1, g2, g3, g4)
@test n_codomain_axes(ft6) == 2
@test isnothing(sanity_check(ft6))

# test permutedims
ft7 = permutedims(ft4, (1, 2), (3, 4))   # trivial
@test ft7 === ft4  # same object

ft7 = permutedims(ft4, (4,), (1, 2, 3))
@test axes(ft7) == (g4, g1, g2, g3)
@test n_codomain_axes(ft7) == 1
@test isnothing(sanity_check(ft7))
