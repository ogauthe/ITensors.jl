@eval module $(gensym())
using Test: @test, @test_broken, @testset

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
  ndims_codomain,
  ndims_domain,
  sanity_check
using NDTensors.GradedAxes: dual, fusion_product, gradedisequal, gradedrange
using NDTensors.Sectors: U1

@testset "Fusion matrix" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])

  # check dual convention when initializing data_matrix
  ft0 = FusionTensor(Float64, (dual(g1),), (g2,))
  @test gradedisequal(matrix_row_axis(ft0), dual(g1))
  @test gradedisequal(matrix_column_axis(ft0), g2)

  m = BlockSparseArray{Float64}(dual(g1), g2)
  ft1 = FusionTensor(m, (dual(g1),), (g2,))

  # getters
  @test data_matrix(ft1) === m
  @test matching_axes(codomain_axes(ft1), (dual(g1),))
  @test matching_axes(domain_axes(ft1), (g2,))

  # misc
  @test matching_axes(axes(ft1), (dual(g1), g2))
  @test ndims_codomain(ft1) == 1
  @test ndims_domain(ft1) == 1
  @test matrix_size(ft1) == (6, 5)
  @test gradedisequal(matrix_row_axis(ft1), dual((g1)))
  @test gradedisequal(matrix_column_axis(ft1), g2)
  @test isnothing(sanity_check(ft0))
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

  ft2 = deepcopy(ft1)
  @test ft2 !== ft1
  @test data_matrix(ft2) == data_matrix(ft1)
  @test data_matrix(ft2) !== data_matrix(ft1)
  @test matching_axes(codomain_axes(ft2), codomain_axes(ft1))
  @test matching_axes(domain_axes(ft2), domain_axes(ft1))

  # similar
  ft2 = similar(ft1)
  @test isnothing(sanity_check(ft2))
  @test eltype(ft2) == Float64
  @test matching_axes(codomain_axes(ft2), codomain_axes(ft1))
  @test matching_axes(domain_axes(ft2), domain_axes(ft1))

  ft3 = similar(ft1, ComplexF64)
  @test isnothing(sanity_check(ft3))
  @test eltype(ft3) == ComplexF64
  @test matching_axes(codomain_axes(ft3), codomain_axes(ft1))
  @test matching_axes(domain_axes(ft3), domain_axes(ft1))

  ft4 = similar(ft1, Float32)
  @test eltype(ft4) == Float64  # promoted

  ft5 = similar(ft1, ComplexF32, ((g1, g1), (g2,)))
  @test isnothing(sanity_check(ft5))
  @test eltype(ft5) == ComplexF64
  @test matching_axes(codomain_axes(ft5), (g1, g1))
  @test matching_axes(domain_axes(ft5), (g2,))
end

@testset "More than 2 axes" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
  gr = dual(fusion_product(g1, g2))
  gc = fusion_product(g3, g4)
  m2 = BlockSparseArray{Float64}(gr, gc)
  ft = FusionTensor(m2, (dual(g1), dual(g2)), (g3, g4))

  @test data_matrix(ft) === m2
  @test matching_axes(codomain_axes(ft), (dual(g1), dual(g2)))
  @test matching_axes(domain_axes(ft), (g3, g4))

  @test axes(ft) == (g1, g2, g3, g4)
  @test ndims_codomain(ft) == 2
  @test ndims_domain(ft) == 2
  @test matrix_size(ft) == (30, 12)
  @test gradedisequal(matrix_row_axis(ft), gr)
  @test gradedisequal(matrix_column_axis(ft), gc)
  @test isnothing(sanity_check(ft))

  @test ndims(ft) == 4
  @test size(ft) == (6, 5, 4, 3)
end

@testset "Base operations" begin
  g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
  ft3 = FusionTensor(Float64, (g1, g2), (g3, g4))
  @test isnothing(sanity_check(ft3))

  ft4 = +ft3
  @test ft4 === ft3  # same object

  ft4 = -ft3
  @test isnothing(sanity_check(ft4))
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)

  ft4 = ft3 + ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test_broken gradedisequal(matrix_row_axis(ft4), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft4), matrix_column_axis(ft3))
  @test_broken isnothing(sanity_check(ft4))

  ft4 = ft3 - ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test_broken gradedisequal(matrix_row_axis(ft4), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft4), matrix_column_axis(ft3))
  @test_broken isnothing(sanity_check(ft4))

  ft4 = 2 * ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test gradedisequal(matrix_row_axis(ft4), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft4), matrix_column_axis(ft3))
  @test isnothing(sanity_check(ft4))
  @test eltype(ft4) == Float64

  ft4 = 2.0 * ft3
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test gradedisequal(matrix_row_axis(ft4), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft4), matrix_column_axis(ft3))
  @test isnothing(sanity_check(ft4))
  @test eltype(ft4) == Float64

  ft4 = ft3 / 2.0
  @test codomain_axes(ft4) === codomain_axes(ft3)
  @test domain_axes(ft4) === domain_axes(ft3)
  @test gradedisequal(matrix_row_axis(ft4), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft4), matrix_column_axis(ft3))
  @test isnothing(sanity_check(ft4))
  @test eltype(ft4) == Float64

  ft5 = 2.0im * ft3
  @test codomain_axes(ft5) === codomain_axes(ft3)
  @test domain_axes(ft5) === domain_axes(ft3)
  @test gradedisequal(matrix_row_axis(ft5), matrix_row_axis(ft3))
  @test gradedisequal(matrix_column_axis(ft5), matrix_column_axis(ft3))
  @test isnothing(sanity_check(ft4))
  @test eltype(ft5) == ComplexF64

  ft4 = conj(ft3)
  @test ft4 === ft3  # same object

  ft6 = conj(ft5)
  @test ft6 !== ft5  # different object
  @test isnothing(sanity_check(ft6))
  @test codomain_axes(ft6) === codomain_axes(ft5)
  @test domain_axes(ft6) === domain_axes(ft5)
  @test gradedisequal(matrix_row_axis(ft6), matrix_row_axis(ft5))
  @test gradedisequal(matrix_column_axis(ft6), matrix_column_axis(ft5))
  @test eltype(ft6) == ComplexF64

  @test_broken isnothing(sanity_check(adjoint(ft3)))
end
end
