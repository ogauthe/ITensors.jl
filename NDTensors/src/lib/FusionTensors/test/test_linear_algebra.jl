using LinearAlgebra
using Test: @test

using BlockArrays: Block

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, sanity_check
using NDTensors.GradedAxes: gradedrange, fuse
using NDTensors.Sectors: U1

g1 = gradedrange([U1(0), U1(1), U1(2)], [1, 2, 3])
g2 = gradedrange([U1(0), U1(1), U1(3)], [2, 2, 1])
g3 = gradedrange([U1(-1), U1(0), U1(1)], [1, 2, 1])
g4 = gradedrange([U1(-1), U1(0), U1(1)], [1, 1, 1])

gr1 = fuse(g1, g2)
gc1 = fuse(g3, g4)
m1 = BlockSparseArray{Float64}(gr1, gc1)
m1[Block(1, 3)] = ones((2, 4))
ft1 = FusionTensor((g1, g2), (g3, g4), m1)
@test isnothing(sanity_check(ft1))

@test norm(ft1) ≈ sqrt(8)
