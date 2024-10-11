@eval module $(gensym())
using LinearAlgebra: LinearAlgebra
using Test: @test, @testset, @test_broken

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, domain_axes, codomain_axes, check_sanity
using NDTensors.GradedAxes: GradedAxes
using NDTensors.SymmetrySectors: U1
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
  @test domain_axes(ft3) === domain_axes(ft1)
  @test codomain_axes(ft3) === codomain_axes(ft2)

  # test LinearAlgebra.mul! with in-place matrix product
  LinearAlgebra.mul!(ft3, ft1, ft2)
  @test isnothing(check_sanity(ft3))
  @test domain_axes(ft3) === domain_axes(ft1)
  @test codomain_axes(ft3) === codomain_axes(ft2)

  LinearAlgebra.mul!(ft3, ft1, ft2, 1.0, 1.0)
  @test isnothing(check_sanity(ft2))
  @test domain_axes(ft3) === domain_axes(ft1)
  @test codomain_axes(ft3) === codomain_axes(ft2)
end

@testset "TensorAlgebra interface" begin
  g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  g2 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
  g3 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
  g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

  ft1 = FusionTensor(Float64, (g1, g2), (g3, g4))
  ft2 = FusionTensor(Float64, GradedAxes.dual.((g3, g4)), (GradedAxes.dual(g1),))
  ft3 = FusionTensor(Float64, GradedAxes.dual.((g3, g4)), GradedAxes.dual.((g1, g2)))

  ft4, legs = TensorAlgebra.contract(ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test legs == (1, 2, 5)
  @test isnothing(check_sanity(ft4))
  @test domain_axes(ft4) === domain_axes(ft1)
  @test codomain_axes(ft4) === codomain_axes(ft2)

  ft5 = TensorAlgebra.contract((1, 2, 5), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
  @test isnothing(check_sanity(ft5))

  # biperm is not allowed
  @test_broken TensorAlgebra.contract(((1, 2), (5,)), ft1, (1, 2, 3, 4), ft2, (3, 4, 5))

  # issue with 0 axis
  @test permutedims(ft1, (), (1, 2, 3, 4)) * permutedims(ft3, (3, 4, 1, 2), ()) isa
    FusionTensor{Float64,0}
  @test_broken TensorAlgebra.contract(ft1, (1, 2, 3, 4), ft3, (3, 4, 1, 2))
end
end
