using LinearAlgebra
using Test: @test

using BlockArrays: Block

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, sanity_check
using NDTensors.GradedAxes
using NDTensors.Sectors: U1

g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
g2 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
g3 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

gr1 = GradedAxes.fusion_product(g1, g2)
gc1 = GradedAxes.fusion_product(g3, g4)
m1 = BlockSparseArray{Float64}(gr1, gc1)
m1[Block(1, 3)] = ones((2, 4))
ft1 = FusionTensor(m1, (g1, g2), (g3, g4))
@test isnothing(sanity_check(ft1))

@test norm(ft1) ≈ sqrt(8)
