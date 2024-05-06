using Test: @test
using LinearAlgebra: I

using NDTensors.Sectors: Sectors, ⊗, sector, SU, SU2, U1, Z

trees, tree_irreps = fusion_trees((U1(0), U1(0)), (false, false))
@test tree_irreps == [U1(0)]
@test trees == [ones((1, 1, 1))]

trees, tree_irreps = fusion_trees((U1(1), U1(1), U1(2)), (false, false, true))
@test tree_irreps == [U1(0)]
@test trees == [ones((1, 1, 1, 1))]

s1 = sector(Z{2}(1), U1(0))
s2 = sector(Z{2}(1), U1(1))
trees, tree_irreps = fusion_trees((s1, s2), (false, false))
@test tree_irreps == [sector(Z{2}(0), U1(1))]
@test trees == [ones((1, 1, 1))]

trees, tree_irreps = fusion_trees((SU2(0), SU2(0), SU2(0)), (false, false, false))
@test tree_irreps == [SU2(0)]
@test trees == [ones((1, 1, 1, 1))]

trees, tree_irreps = fusion_trees((SU2(1 / 2),), (false,))
@test tree_irreps == [SU2(1 / 2)]
@test trees == [I(2)]

trees, tree_irreps = fusion_trees((SU2(1 / 2),), (true,))
@test tree_irreps == [SU2(1 / 2)]
@test trees == [[0 -1; 1 0]]

trees, tree_irreps = fusion_trees(
  (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)), (false, false, false)
)
@test tree_irreps == [SU2(1 / 2), SU2(1 / 2), SU2(3 / 2)]
