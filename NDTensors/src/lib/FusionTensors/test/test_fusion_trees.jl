@eval module $(gensym())
using Test: @test, @testset
using LinearAlgebra: LinearAlgebra

using BlockArrays: BlockArrays

using NDTensors.FusionTensors: fusion_trees
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: Sectors, SU, SU2, U1, Z

@testset "Abelian fusion trees" begin
  trees2, tree_irreps2 = fusion_trees((U1(0), U1(0)), (false, false))
  @test tree_irreps2 == [U1(0)]
  @test trees2 == [ones((1, 1, 1, 1))]

  trees3, tree_irreps3 = fusion_trees((U1(1), U1(1), U1(2)), (false, false, true))
  @test tree_irreps3 == [U1(0)]
  @test trees3 == [ones((1, 1, 1, 1, 1))]

  s1 = Sectors.sector(Z{2}(1), U1(0))
  s2 = Sectors.sector(Z{2}(1), U1(1))
  trees_z2u1, tree_irreps_z2u1 = fusion_trees((s1, s2), (false, false))
  @test tree_irreps_z2u1 == [Sectors.sector(Z{2}(0), U1(1))]
  @test trees_z2u1 == [ones((1, 1, 1, 1))]

  s3 = Sectors.sector(; A=Z{2}(1), B=U1(0))
  s4 = Sectors.sector(; A=Z{2}(1), B=U1(1))
  trees_nt, tree_irreps_nt = fusion_trees((s3, s4), (false, false))
  @test tree_irreps_nt == [Sectors.sector(; A=Z{2}(0), B=U1(1))]
  @test trees_nt == [ones((1, 1, 1, 1))]
end

@testset "SU(2) fusion trees" begin
  trees0, tree_irreps0 = fusion_trees((SU2(0), SU2(0), SU2(0)), (false, false, false))
  @test tree_irreps0 == [SU2(0)]
  @test trees0 == [ones((1, 1, 1, 1, 1))]

  trees1, tree_irreps1 = fusion_trees((SU2(1 / 2),), (false,))
  @test tree_irreps1 == [SU2(1 / 2)]
  @test trees1 == [reshape(LinearAlgebra.I(2), (2, 2, 1))]

  trees_1b, tree_irreps_1b = fusion_trees((SU2(1 / 2),), (true,))
  @test tree_irreps_1b == [SU2(1 / 2)]
  @test trees_1b == [reshape([0, 1, -1, 0], (2, 2, 1))]

  trees3h, tree_irreps3h = fusion_trees(
    (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)), (false, false, false)
  )
  @test tree_irreps3h == [SU2(1 / 2), SU2(3 / 2)]
  @test size.(trees3h) == [(2, 2, 2, 2, 2), (2, 2, 2, 4, 1)]
end

@testset "CategoryProduct fusion trees" begin
  hole = Sectors.sector(; N=U1(1), S=SU2(0))
  s12 = Sectors.sector(; N=U1(0), S=SU2(1 / 2))
  trees, tree_irreps = fusion_trees((hole, hole, hole, s12), (false, false, false, false))
  @test tree_irreps == [Sectors.sector(; N=U1(3), S=SU2(1 / 2))]
  @test trees == [reshape(LinearAlgebra.I(2), (1, 1, 1, 2, 2, 1))]

  isdual = (false, false, false)
  s3 = Sectors.sector(Sectors.SU((1, 0)), Sectors.SU((1,)), Sectors.U1(1))
  irreps = (s3, s3, s3)
  trees, tree_irreps = fusion_trees(irreps, isdual)
  rep = reduce(GradedAxes.fusion_product, irreps)
  @test GradedAxes.blocklabels(rep) == tree_irreps
  @test Sectors.quantum_dimension.(tree_irreps) == size.(trees, 4)
  @test GradedAxes.unlabel.(BlockArrays.blocklengths(rep)) == size.(trees, 5)

  s_nt = Sectors.sector(; A=Sectors.SU((1, 0)), B=Sectors.SU((1,)), C=Sectors.U1(1))
  irreps_nt = (s_nt, s_nt, s_nt)
  trees_nt, tree_irreps_nt = fusion_trees(irreps_nt, isdual)
  rep_nt = reduce(GradedAxes.fusion_product, irreps_nt)
  @test GradedAxes.blocklabels(rep_nt) == tree_irreps_nt
  @test Sectors.quantum_dimension.(tree_irreps_nt) == size.(trees_nt, 4)
  @test GradedAxes.unlabel.(BlockArrays.blocklengths(rep_nt)) == size.(trees_nt, 5)
end
end
