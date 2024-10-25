@eval module $(gensym())
using Test: @test, @testset
using LinearAlgebra: LinearAlgebra

using BlockArrays: BlockArrays

using NDTensors.FusionTensors: fusion_trees
using NDTensors.GradedAxes: GradedAxes
using NDTensors.SymmetrySectors: SymmetrySectors, SU, SU2, TrivialSector, U1, Z

@testset "Trivial fusion trees" begin
  tree_irreps_pairs1 = fusion_trees((), ())
  @test first.(tree_irreps_pairs1) == [TrivialSector()]
  @test last.(tree_irreps_pairs1) == [ones((1, 1))]

  tree_irreps_pairs2 = fusion_trees((TrivialSector(), TrivialSector()), (false, false))
  @test first.(tree_irreps_pairs2) == [TrivialSector()]
  @test last.(tree_irreps_pairs2) == [ones((1, 1, 1, 1))]

  tree_irreps_pairs3 = fusion_trees(
    (TrivialSector(), TrivialSector(), TrivialSector()), (false, true, true)
  )
  @test first.(tree_irreps_pairs3) == [TrivialSector()]
  @test last.(tree_irreps_pairs3) == [ones((1, 1, 1, 1, 1))]
end

@testset "Abelian fusion trees" begin
  tree_irreps_pairs1 = fusion_trees((U1(0), U1(0)), (false, false))
  @test first.(tree_irreps_pairs1) == [U1(0)]
  @test last.(tree_irreps_pairs1) == [ones((1, 1, 1, 1))]

  tree_irreps_pairs2 = fusion_trees((U1(1), U1(1), U1(-2)), (false, false, false))
  @test first.(tree_irreps_pairs2) == [U1(0)]
  @test last.(tree_irreps_pairs2) == [ones((1, 1, 1, 1, 1))]

  s1 = SymmetrySectors.SectorProduct(Z{2}(1), U1(0))
  s2 = SymmetrySectors.SectorProduct(Z{2}(1), U1(1))
  tree_irreps_z2u1 = fusion_trees((s1, s2), (false, false))
  @test first.(tree_irreps_z2u1) == [SymmetrySectors.SectorProduct(Z{2}(0), U1(1))]
  @test last.(tree_irreps_z2u1) == [ones((1, 1, 1, 1))]

  s3 = SymmetrySectors.SectorProduct(; A=Z{2}(1), B=U1(0))
  s4 = SymmetrySectors.SectorProduct(; A=Z{2}(1), B=U1(1))
  tree_irreps_nt = fusion_trees((s3, s4), (false, false))
  @test first.(tree_irreps_nt) == [SymmetrySectors.SectorProduct(; A=Z{2}(0), B=U1(1))]
  @test last.(tree_irreps_nt) == [ones((1, 1, 1, 1))]
end

@testset "SU(2) fusion trees" begin
  tree_irreps_pairs1 = fusion_trees((SU2(0), SU2(0), SU2(0)), (false, false, false))
  @test first.(tree_irreps_pairs1) == [SU2(0)]
  @test last.(tree_irreps_pairs1) == [ones((1, 1, 1, 1, 1))]

  tree_irreps_pairs2 = fusion_trees((SU2(1 / 2),), (false,))
  @test first.(tree_irreps_pairs2) == [SU2(1 / 2)]
  @test last.(tree_irreps_pairs2) == [reshape(LinearAlgebra.I(2), (2, 2, 1))]

  tree_irreps_pairs3 = fusion_trees((SU2(1 / 2),), (true,))
  @test first.(tree_irreps_pairs3) == [SU2(1 / 2)]
  @test last.(tree_irreps_pairs3) ≈ [reshape([0, 1, -1, 0], (2, 2, 1))]

  tree_irreps_pairs4 = fusion_trees(
    (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)), (false, false, false)
  )
  @test first.(tree_irreps_pairs4) == [SU2(1 / 2), SU2(3 / 2)]
  @test size.(last.(tree_irreps_pairs4)) == [(2, 2, 2, 2, 2), (2, 2, 2, 4, 1)]
end

@testset "SectorProduct fusion trees" begin
  hole = SymmetrySectors.SectorProduct(; N=U1(1), S=SU2(0))
  s12 = SymmetrySectors.SectorProduct(; N=U1(0), S=SU2(1 / 2))
  tree_irreps_pairs1 = fusion_trees((hole, hole, hole, s12), (false, false, false, false))
  @test first.(tree_irreps_pairs1) ==
    [SymmetrySectors.SectorProduct(; N=U1(3), S=SU2(1 / 2))]
  @test last.(tree_irreps_pairs1) == [reshape(LinearAlgebra.I(2), (1, 1, 1, 2, 2, 1))]

  s3 = SymmetrySectors.SectorProduct(
    SymmetrySectors.SU((1, 0)), SymmetrySectors.SU((1,)), SymmetrySectors.U1(1)
  )
  irreps = (s3, s3, s3)
  arrows = (false, false, false)
  tree_irreps_pairs2 = fusion_trees(irreps, arrows)
  tree_irreps, trees = first.(tree_irreps_pairs2), last.(tree_irreps_pairs2)
  rep = GradedAxes.fusion_product(irreps...)
  @test GradedAxes.blocklabels(rep) == tree_irreps
  @test SymmetrySectors.quantum_dimension.(tree_irreps) == size.(trees, 4)
  @test GradedAxes.unlabel.(BlockArrays.blocklengths(rep)) == size.(trees, 5)

  s_nt = SymmetrySectors.SectorProduct(;
    A=SymmetrySectors.SU((1, 0)), B=SymmetrySectors.SU((1,)), C=SymmetrySectors.U1(1)
  )
  irreps_nt = (s_nt, s_nt, s_nt)
  tree_irreps_nt_pairs = fusion_trees(irreps_nt, arrows)
  tree_irreps_nt, trees_nt = first.(tree_irreps_nt_pairs), last.(tree_irreps_nt_pairs)
  rep_nt = reduce(GradedAxes.fusion_product, irreps_nt)
  @test GradedAxes.blocklabels(rep_nt) == tree_irreps_nt
  @test SymmetrySectors.quantum_dimension.(tree_irreps_nt) == size.(trees_nt, 4)
  @test GradedAxes.unlabel.(BlockArrays.blocklengths(rep_nt)) == size.(trees_nt, 5)
end
end
