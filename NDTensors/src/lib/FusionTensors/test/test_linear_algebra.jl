@eval module $(gensym())
using LinearAlgebra: norm, tr
using Test: @test, @testset

using BlockArrays: BlockArrays

using NDTensors.BlockSparseArrays: BlockSparseArrays
using NDTensors.FusionTensors: FusionTensor, check_sanity
using NDTensors.GradedAxes
using NDTensors.SymmetrySectors: U1, SU2, TrivialSector

@testset "LinearAlgebra interface" begin
  sds22 = [
    0.25 0.0 0.0 0.0
    0.0 -0.25 0.5 0.0
    0.0 0.5 -0.25 0.0
    0.0 0.0 0.0 0.25
  ]
  sdst = reshape(sds22, (2, 2, 2, 2))

  g0 = GradedAxes.gradedrange([TrivialSector() => 2])
  gu1 = GradedAxes.gradedrange([U1(1) => 1, U1(-1) => 1])
  gsu2 = GradedAxes.gradedrange([SU2(1 / 2) => 1])

  for g in [g0, gu1, gsu2]
    ft0 = FusionTensor(Float64, (g, g), (GradedAxes.dual(g), GradedAxes.dual(g)))
    @test isnothing(check_sanity(ft0))
    @test norm(ft0) == 0
    @test tr(ft0) == 0

    ft = FusionTensor(sdst, (g, g), (GradedAxes.dual(g), GradedAxes.dual(g)))
    @test isnothing(check_sanity(ft))
    @test norm(ft) ≈ √3 / 2
    @test isapprox(tr(ft), 0; atol=eps(Float64))
  end
end
end
