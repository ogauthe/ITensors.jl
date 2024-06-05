@eval module $(gensym())
using Test: @test, @testset

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, ndims_codomain, sanity_check, matching_axes
using NDTensors.GradedAxes
using NDTensors.Sectors: U1

@testset "permutedims" begin
  g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
  ft1 = FusionTensor(Float64, GradedAxes.dual.((g1, g2)), (g3, g4))
  @test isnothing(sanity_check(ft1))

  # test permutedims
  ft2 = permutedims(ft1, (1, 2), (3, 4))   # trivial
  @test ft2 === ft1  # same object

  ft2 = permutedims(ft1, ((1, 2), (3, 4)))   # trivial with 2-tuple of tuples
  @test ft2 === ft1  # same object

  ft3 = permutedims(ft1, (4,), (1, 2, 3))
  @test matching_axes(axes(ft3), (g4, GradedAxes.dual(g1), GradedAxes.dual(g2), g3))
  @test ndims_codomain(ft3) == 1
  @test isnothing(sanity_check(ft3))
end
end
