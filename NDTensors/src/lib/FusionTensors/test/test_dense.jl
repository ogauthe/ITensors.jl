@eval module $(gensym())
using LinearAlgebra: LinearAlgebra
using Test: @test, @testset

using BlockArrays: BlockArrays

using NDTensors.FusionTensors: FusionTensor, data_matrix
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: SU2, U1, sector

@testset "Abelian FusionTensor" begin
  # trivial matrix
  g = GradedAxes.gradedrange([U1(0) => 1])
  gb = GradedAxes.dual(g)
  m = ones((1, 1))
  ft = FusionTensor(m, (gb,), (g,))
  @test size(data_matrix(ft)) == (1, 1)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[1, 1] ≈ 1.0
  @test Array(ft) ≈ m

  # non self-conjugate
  g = GradedAxes.gradedrange([U1(1) => 2])
  gb = GradedAxes.dual(g)
  m = ones((2, 2))
  ft = FusionTensor(m, (gb,), (g,))
  @test size(data_matrix(ft)) == (2, 2)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[BlockArrays.Block(1, 1)] ≈ m
  @test Array(ft) ≈ m

  # several axes, one block
  g1 = GradedAxes.gradedrange([U1(1) => 2])
  g2 = GradedAxes.gradedrange([U1(2) => 3])
  g3 = GradedAxes.gradedrange([U1(3) => 4])
  g4 = GradedAxes.gradedrange([U1(0) => 2])
  codomain_legs = GradedAxes.dual.((g1, g2))
  domain_legs = (g3, g4)
  m = convert.(Float64, reshape(collect(1:48), (2, 3, 4, 2)))
  ft = FusionTensor(m, codomain_legs, domain_legs)
  @test size(data_matrix(ft)) == (6, 8)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[BlockArrays.Block(1, 1)] ≈ reshape(m, (6, 8))
  @test Array(ft) ≈ m

  # several axes, several blocks
  g1 = GradedAxes.gradedrange([U1(1) => 2, U1(2) => 2])
  g2 = GradedAxes.gradedrange([U1(2) => 3, U1(3) => 2])
  g3 = GradedAxes.gradedrange([U1(3) => 4, U1(4) => 1])
  g4 = GradedAxes.gradedrange([U1(0) => 2, U1(2) => 1])
  codomain_legs = GradedAxes.dual.((g1, g2))
  domain_legs = (g3, g4)
  dense = zeros((4, 5, 5, 3))
  dense[1:2, 1:3, 1:4, 1:2] .= 1.0
  dense[3:4, 1:3, 5:5, 1:2] .= 2.0
  dense[1:2, 4:5, 5:5, 1:2] .= 3.0
  dense[3:4, 4:5, 1:4, 3:3] .= 4.0
  ft = FusionTensor(dense, codomain_legs, domain_legs)
  @test size(data_matrix(ft)) == (20, 15)
  @test BlockArrays.blocksize(data_matrix(ft)) == (3, 4)
  @test LinearAlgebra.norm(ft) ≈ LinearAlgebra.norm(dense)
  @test Array(ft) ≈ dense
end

@testset "SU(2) FusionTensor" begin
  # trivial  matrix
  g = GradedAxes.gradedrange([SU2(0) => 1])
  gb = GradedAxes.dual(g)
  m = ones((1, 1))
  ft = FusionTensor(m, (gb,), (g,))
  @test size(data_matrix(ft)) == (1, 1)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[1, 1] ≈ 1.0
  @test Array(ft) ≈ m

  g2 = GradedAxes.gradedrange([SU2(1 / 2) => 1])
  g2b = GradedAxes.dual(g2)

  # spin 1/2 Id
  ft = FusionTensor(LinearAlgebra.I((2)), (g2b,), (g2,))
  @test LinearAlgebra.norm(ft) ≈ √2
  @test Array(ft) ≈ LinearAlgebra.I((2))

  # S⋅S
  sds22 = reshape(
    [
      [0.25, 0.0, 0.0, 0.0]
      [0.0, -0.25, 0.5, 0.0]
      [0.0, 0.5, -0.25, 0.0]
      [0.0, 0.0, 0.0, 0.25]
    ],
    (2, 2, 2, 2),
  )
  dense, codomain_legs, domain_legs = sds22, (g2b, g2b), (g2, g2)
  ft = FusionTensor(dense, codomain_legs, domain_legs)
  @test LinearAlgebra.norm(ft) ≈ √3 / 2
  @test Array(ft) ≈ sds22

  # dual over one spin. This changes the dense coefficients but not the FusionTensor ones
  sds22b = reshape(
    [
      [-0.25, 0.0, 0.0, -0.5]
      [0.0, 0.25, 0.0, 0.0]
      [0.0, 0.0, 0.25, 0.0]
      [-0.5, 0.0, 0.0, -0.25]
    ],
    (2, 2, 2, 2),
  )
  sds22b_domain_legs = (g2, g2b)
  dense, codomain_legs, domain_legs = sds22b, (g2, g2b), (g2b, g2)
  ftb = FusionTensor(dense, codomain_legs, domain_legs)
  @test LinearAlgebra.norm(ftb) ≈ √3 / 2
  @test Array(ftb) ≈ sds22b

  # no domain axis
  dense, codomain_legs, domain_legs = sds22, (g2b, g2b, g2, g2), ()
  ft = FusionTensor(dense, codomain_legs, domain_legs)
  @test Array(ft) ≈ sds22

  # no codomain axis
  dense, codomain_legs, domain_legs = sds22, (), (g2b, g2b, g2, g2)
  ft = FusionTensor(dense, codomain_legs, domain_legs)
  @test Array(ft) ≈ sds22

  # large identity
  g = reduce(GradedAxes.fusion_product, (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)))
  N = 3
  domain_legs = ntuple(_ -> g, N)
  codomain_legs = GradedAxes.dual.(domain_legs)
  d = 8
  dense = reshape(LinearAlgebra.I(d^N), ntuple(_ -> d, 2 * N))
  ft = FusionTensor(dense, codomain_legs, domain_legs)
  @test Array(ft) ≈ dense
end

@testset "U(1)×SU(2) FusionTensor" begin
  for d in 1:6  # any spin dimension
    s = SU2((d - 1)//2)  # d = 2s+1
    D = d + 1
    tRVB = zeros((d, D, D, D, D))  # tensor RVB SU(2) for spin s
    for i in 1:d
      tRVB[i, i + 1, 1, 1, 1] = 1.0
      tRVB[i, 1, i + 1, 1, 1] = 1.0
      tRVB[i, 1, 1, i + 1, 1] = 1.0
      tRVB[i, 1, 1, 1, i + 1] = 1.0
    end

    gd = GradedAxes.gradedrange([sector(s, U1(3)) => 1])
    codomain_legs = (GradedAxes.dual(gd),)
    gD = GradedAxes.gradedrange([sector(SU2(0), U1(1)) => 1, sector(s, U1(0)) => 1])
    domain_legs = (gD, gD, gD, gD)
    ft = FusionTensor(tRVB, codomain_legs, domain_legs)
    @test Array(ft) ≈ tRVB

    # same with NamedTuples
    gd_nt = GradedAxes.gradedrange([sector(; S=s, N=U1(3)) => 1])
    codomain_legs_nt = (GradedAxes.dual(gd_nt),)
    gD_nt = GradedAxes.gradedrange([
      sector(; S=SU2(0), N=U1(1)) => 1, sector(; S=s, N=U1(0)) => 1
    ])
    domain_legs_nt = (gD_nt, gD_nt, gD_nt, gD_nt)
    ft_nt = FusionTensor(tRVB, codomain_legs_nt, domain_legs_nt)
    @test Array(ft_nt) ≈ tRVB
  end
end
end
