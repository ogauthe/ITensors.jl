using Test: @test
using LinearAlgebra: I

using BlockArrays: BlockArrays

using NDTensors.FusionTensors: fusion_trees
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: Sectors, ⊗, SU, SU2, U1, Z, quantum_dimension, sector

trees, tree_irreps = fusion_trees((U1(0), U1(0)), (false, false))
@test tree_irreps == [U1(0)]
@test trees == [ones((1, 1, 1, 1))]

trees, tree_irreps = fusion_trees((U1(1), U1(1), U1(2)), (false, false, true))
@test tree_irreps == [U1(0)]
@test trees == [ones((1, 1, 1, 1, 1))]

s1 = sector(Z{2}(1), U1(0))
s2 = sector(Z{2}(1), U1(1))
trees, tree_irreps = fusion_trees((s1, s2), (false, false))
@test tree_irreps == [sector(Z{2}(0), U1(1))]
@test trees == [ones((1, 1, 1, 1))]

trees, tree_irreps = fusion_trees((SU2(0), SU2(0), SU2(0)), (false, false, false))
@test tree_irreps == [SU2(0)]
@test trees == [ones((1, 1, 1, 1, 1))]

trees, tree_irreps = fusion_trees((SU2(1 / 2),), (false,))
@test tree_irreps == [SU2(1 / 2)]
@test trees == [reshape(I(2), (2, 2, 1))]

trees, tree_irreps = fusion_trees((SU2(1 / 2),), (true,))
@test tree_irreps == [SU2(1 / 2)]
@test trees == [reshape([0, 1, -1, 0], (2, 2, 1))]

trees, tree_irreps = fusion_trees(
  (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)), (false, false, false)
)
@test tree_irreps == [SU2(1 / 2), SU2(3 / 2)]
@test size.(trees) == [(2, 2, 2, 2, 2), (2, 2, 2, 4, 1)]

N = 3
isdual = (false, false, false)
s4 = Sectors.sector(; A=Sectors.SU((1, 0)), B=Sectors.SU((1,)), C=Sectors.U1(1))
s5 = Sectors.sector(Sectors.SU((1, 0)), Sectors.SU((1,)), Sectors.U1(1))
irreps4 = ntuple(_ -> s4, N)
irreps5 = ntuple(_ -> s5, N)
trees4, tree_irreps4 = fusion_trees(irreps4, isdual);
rep4 = reduce(GradedAxes.fusion_product, irreps4)
@test GradedAxes.blocklabels(rep4) == tree_irreps4
@test quantum_dimension.(tree_irreps4) == size.(trees4, N + 1)
@test GradedAxes.unlabel.(BlockArrays.blocklengths(rep4)) == size.(trees4, N + 2)

trees5, tree_irreps5 = fusion_trees(irreps5, isdual);
rep5 = reduce(GradedAxes.fusion_product, irreps5)
@test GradedAxes.blocklabels(rep5) == tree_irreps5
@test quantum_dimension.(tree_irreps5) == size.(trees5, N + 1)
@test GradedAxes.unlabel.(BlockArrays.blocklengths(rep5)) == size.(trees5, N + 2)
