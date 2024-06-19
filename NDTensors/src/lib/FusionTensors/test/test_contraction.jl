@eval module $(gensym())
using LinearAlgebra: LinearAlgebra
using Test: @test, @testset

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, codomain_axes, domain_axes, check_sanity
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: U1
using NDTensors.TensorAlgebra: TensorAlgebra

@testset "contraction" begin
  g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor(Float64, (g1, g2), (g3, g4))
  @test isnothing(check_sanity(ft1))

  ft2 = FusionTensor(Float64, GradedAxes.dual.((g3, g4)), (g1,))
  @test isnothing(check_sanity(ft2))

  ft3 = ft1 * ft2  # tensor contraction
  @test isnothing(check_sanity(ft3))
  @test codomain_axes(ft3) === codomain_axes(ft1)
  @test domain_axes(ft3) === domain_axes(ft2)

  # test LinearAlgebra.mul! with in-place matrix product
  LinearAlgebra.mul!(ft3, ft1, ft2)
  @test isnothing(check_sanity(ft3))
  @test codomain_axes(ft3) === codomain_axes(ft1)
  @test domain_axes(ft3) === domain_axes(ft2)

  LinearAlgebra.mul!(ft3, ft1, ft2, 1.0, 1.0)
  @test isnothing(check_sanity(ft2))
  @test codomain_axes(ft3) === codomain_axes(ft1)
  @test domain_axes(ft3) === domain_axes(ft2)
end

@testset "TensorAlgebra interface" begin
  g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor(Float64, (g1, g2), (g3, g4))
  ft2 = FusionTensor(Float64, GradedAxes.dual.((g3, g4)), (g1,))

  ft4, legs = TensorAlgebra.contract(ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test legs == (1, 2, 5)
  @test isnothing(check_sanity(ft4))
  @test codomain_axes(ft4) === codomain_axes(ft1)
  @test domain_axes(ft4) === domain_axes(ft2)
end
end
