using Test: @test
using LinearAlgebra: I

using NDTensors.Sectors: Sectors, ⊗, SU, SU2, U1, Z, quantum_dimension, sector

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

N = 3
isdual = (false, false, false)
s4 = Sectors.sector(; A=Sectors.SU((1, 0)), B=Sectors.SU((1,)), C=Sectors.U1(1))
s5 = Sectors.sector(Sectors.SU((1, 0)), Sectors.SU((1,)), Sectors.U1(1))
irreps4 = ntuple(_ -> s4, N)
irreps5 = ntuple(_ -> s5, N)
trees4, tree_irreps = fusion_trees(irreps4, isdual)
@test quantum_dimension.(tree_irreps) == size.(trees4, N + 1)

trees5, tree_irreps = fusion_trees(irreps5, isdual)
@test quantum_dimension.(tree_irreps) == size.(trees5, N + 1)
@test trees4 == trees5
