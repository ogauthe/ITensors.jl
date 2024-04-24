using Test: @test

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors:
  FusionTensor,
  codomain_axes,
  domain_axes,
  data_matrix,
  matching_axes,
  matching_dual,
  matrix_column_axis,
  matrix_row_axis,
  matrix_size,
  n_codomain_axes,
  n_domain_axes,
  sanity_check
using NDTensors.GradedAxes: fusion_product, gradedisequal, gradedrange
using NDTensors.Sectors: U1

g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
m = BlockSparseArray{Float64}(g1, g2)

ft1 = FusionTensor((g1,), (g2,), m)  # constructor from split axes

# getters
@test data_matrix(ft1) === m
@test matching_axes(codomain_axes(ft1), (g1,))
@test matching_axes(domain_axes(ft1), (g2,))

# misc
@test matching_axes(axes(ft1), (g1, g2))
@test n_codomain_axes(ft1) == 1
@test n_domain_axes(ft1) == 1
@test matrix_size(ft1) == (6, 5)
@test gradedisequal(matrix_row_axis(ft1), g1)
@test gradedisequal(matrix_column_axis(ft1), g2)
@test isnothing(sanity_check(ft1))

# Base methods
@test eltype(ft1) === Float64
@test length(ft1) == 30
@test ndims(ft1) == 2
@test size(ft1) == (6, 5)

# copy
ft2 = copy(ft1)
@test isnothing(sanity_check(ft2))
@test ft2 !== ft1
@test data_matrix(ft2) == data_matrix(ft1)
@test data_matrix(ft2) !== data_matrix(ft1)
@test matching_axes(codomain_axes(ft2), codomain_axes(ft1))
@test matching_axes(domain_axes(ft2), domain_axes(ft1))

# deepcopy
ft2 = deepcopy(ft1)
@test ft2 !== ft1
@test data_matrix(ft2) == data_matrix(ft1)
@test data_matrix(ft2) !== data_matrix(ft1)
@test matching_axes(codomain_axes(ft2), codomain_axes(ft1))
@test matching_axes(domain_axes(ft2), domain_axes(ft1))

# more than 2 axes
g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
gr = fusion_product(g1, g2)
gc = fusion_product(g3, g4)
m2 = BlockSparseArray{Float64}(gr, gc)
ft3 = FusionTensor((g1, g2), (g3, g4), m2)

@test data_matrix(ft3) === m2
@test matching_axes(codomain_axes(ft3), (g1, g2))
@test matching_axes(domain_axes(ft3), (g3, g4))

@test axes(ft3) == (g1, g2, g3, g4)
@test n_codomain_axes(ft3) == 2
@test n_domain_axes(ft3) == 2
@test matrix_size(ft3) == (30, 12)
@test gradedisequal(matrix_row_axis(ft3), gr)
@test gradedisequal(matrix_column_axis(ft3), gc)
@test isnothing(sanity_check(ft3))

@test ndims(ft3) == 4
@test size(ft3) == (6, 5, 4, 3)

# test Base operations
ft4 = +ft3
@test ft4 === ft3  # same object

ft4 = -ft3
@test isnothing(sanity_check(ft4))
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)

ft4 = ft3 + ft3
@test isnothing(sanity_check(ft4))
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)

ft4 = ft3 - ft3
@test isnothing(sanity_check(ft4))
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)

ft4 = 2 * ft3
@test isnothing(sanity_check(ft4))
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)
@test eltype(ft4) == Float64

ft4 = 2.0 * ft3
@test isnothing(sanity_check(ft4))
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)
@test eltype(ft4) == Float64

ft4 = ft3 / 2.0
@test codomain_axes(ft4) === codomain_axes(ft3)
@test domain_axes(ft4) === domain_axes(ft3)
@test isnothing(sanity_check(ft4))

ft5 = 2.0im * ft3
@test isnothing(sanity_check(ft5))
@test codomain_axes(ft5) === codomain_axes(ft3)
@test domain_axes(ft5) === domain_axes(ft3)
@test eltype(ft5) == ComplexF64

ft4 = conj(ft3)
@test ft4 === ft3  # same object

ft6 = conj(ft5)
@test ft6 !== ft5  # different object
@test isnothing(sanity_check(ft6))
@test codomain_axes(ft6) === codomain_axes(ft5)
@test domain_axes(ft6) === domain_axes(ft5)
@test eltype(ft6) == ComplexF64

# ft7 = adjoint(ft3) # currently unimplemented for BlockSparseArray
# @test isnothing(sanity_check(ft7))

# test cast from and to dense
arr = zeros((6, 5, 4, 3))
#TODO fill with data
ft8 = FusionTensor((g1, g2), (g3, g4), arr)
@test matching_axes(axes(ft8), (g1, g2, g3, g4))
@test n_codomain_axes(ft8) == 2
@test isnothing(sanity_check(ft8))

arr2 = Array(ft8)
@test arr2 ≈ arr
