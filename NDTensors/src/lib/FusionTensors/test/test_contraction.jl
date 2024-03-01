using LinearAlgebra
using Test: @test

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, codomain_axes, domain_axes, sanity_check
using NDTensors.GradedAxes
using NDTensors.Sectors: U1
using NDTensors.TensorAlgebra

g1 = GradedAxes.gradedrange([U1(0), U1(1), U1(2)], [1, 2, 3])
g2 = GradedAxes.gradedrange([U1(0), U1(1), U1(3)], [2, 2, 1])
g3 = GradedAxes.gradedrange([U1(-1), U1(0), U1(1)], [1, 2, 1])
g4 = GradedAxes.gradedrange([U1(-1), U1(0), U1(1)], [1, 1, 1])

gr1 = GradedAxes.fuse(g1, g2)
gc1 = GradedAxes.fuse(g3, g4)
m1 = BlockSparseArray{Float64}(gr1, gc1)
ft1 = FusionTensor((g1, g2), (g3, g4), m1)

gr2 = GradedAxes.dual(GradedAxes.fuse(g3, g4))
gc2 = GradedAxes.dual(g1)
m2 = BlockSparseArray{Float64}(gr2, gc2)
ft2 = FusionTensor(GradedAxes.dual.((g3, g4)), (g1,), m2)

ft3 = ft1 * ft2  # tensor contraction
@test isnothing(sanity_check(ft1))
@test codomain_axes(ft3) === codomain_axes(ft1)
@test domain_axes(ft3) === domain_axes(ft2)

# test LinearAlgebra.mul! with in-place matrix product
mul!(ft3, ft1, ft2)
@test isnothing(sanity_check(ft2))
@test codomain_axes(ft3) === codomain_axes(ft1)
@test domain_axes(ft3) === domain_axes(ft2)

mul!(ft3, ft1, ft2, 1.0, 1.0)
@test isnothing(sanity_check(ft2))
@test codomain_axes(ft3) === codomain_axes(ft1)
@test domain_axes(ft3) === domain_axes(ft2)

# test TensorAlgebra interface
x = TensorAlgebra.contract(ft1, (1, 2, 3, 4), ft2, (3, 4, 5))
@test length(x) == 2
@test x[2] == (1, 2, 5)
ft4 = x[1]
@test codomain_axes(ft4) === codomain_axes(ft1)
@test domain_axes(ft4) === domain_axes(ft2)
