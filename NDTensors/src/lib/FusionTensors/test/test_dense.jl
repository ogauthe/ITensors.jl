using LinearAlgebra: LinearAlgebra, norm
using Test: @test, @testset

using BlockArrays: BlockArrays

using NDTensors.BlockSparseArrays: BlockSparseArrays, BlockSparseArray
using NDTensors.FusionTensors:
  FusionTensor,
  FusionTensors,
  ndims_codomain,
  ndims_domain,
  codomain_axes,
  domain_axes,
  data_matrix,
  sanity_check,
  fusion_trees,
  initialize_data_matrix,
  initialize_trivial_axis
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: Sectors, SU2, U1, sector, quantum_dimension
using NDTensors.TensorAlgebra: TensorAlgebra, BlockedPermutation, blockedperm, blocklengths

@testset "Abelian FusionTensor" begin
  # trivial  matrix
  g = GradedAxes.gradedrange([U1(0) => 1])
  gb = GradedAxes.dual(g)
  m = ones((1, 1))
  ft = FusionTensor((gb,), (g,), m)
  @test size(data_matrix(ft)) == (1, 1)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[1, 1] ≈ 1.0
  @test Array(ft) ≈ m

  # non self-conjugate
  g = GradedAxes.gradedrange([U1(1) => 2])
  gb = GradedAxes.dual(g)
  m = ones((2, 2))
  ft = FusionTensor((gb,), (g,), m)
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
  ft = FusionTensor(codomain_legs, domain_legs, m)
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
  ft = FusionTensor(codomain_legs, domain_legs, dense)
  @test size(data_matrix(ft)) == (20, 15)
  @test BlockArrays.blocksize(data_matrix(ft)) == (3, 4)
  @test Array(ft) ≈ dense
end

@testset "SU(2) FusionTensor" begin
  # trivial  matrix
  g = GradedAxes.gradedrange([SU2(0) => 1])
  gb = GradedAxes.dual(g)
  m = ones((1, 1))
  ft = FusionTensor((gb,), (g,), m)
  @test size(data_matrix(ft)) == (1, 1)
  @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
  @test data_matrix(ft)[1, 1] ≈ 1.0
  @test Array(ft) ≈ m

  g2 = GradedAxes.gradedrange([SU2(1 / 2) => 1])
  g2b = GradedAxes.dual(g2)

  # spin 1/2 Id
  ft = FusionTensor((g2b,), (g2,), LinearAlgebra.I((2)))
  @test norm(ft) ≈ √2
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
  codomain_legs, domain_legs, dense = (g2b, g2b), (g2, g2), sds22
  ft = FusionTensor(codomain_legs, domain_legs, dense)
  @test norm(ft) ≈ √3 / 2
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
  codomain_legs, domain_legs, dense = (g2, g2b), (g2b, g2), sds22b
  ftb = FusionTensor(codomain_legs, domain_legs, dense)
  @test norm(ftb) ≈ √3 / 2
  @test Array(ftb) ≈ sds22b

  # no domain axis
  codomain_legs, domain_legs, dense = (g2b, g2b, g2, g2), (), sds22
  ft = FusionTensor(codomain_legs, domain_legs, dense)
  @test Array(ft) ≈ sds22

  # no codomain axis
  codomain_legs, domain_legs, dense = (), (g2b, g2b, g2, g2), sds22
  ft = FusionTensor(codomain_legs, domain_legs, dense)
  @test Array(ft) ≈ sds22

  # large identity
  g = reduce(GradedAxes.fusion_product, (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)))
  N = 3
  domain_legs = ntuple(_ -> g, N)
  codomain_legs = GradedAxes.dual.(domain_legs)
  d = quantum_dimension(g)
  dense = reshape(LinearAlgebra.I(d^N), ntuple(_ -> d, 2 * N))
  ft = FusionTensor(codomain_legs, domain_legs, dense)
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
    ft = FusionTensor(codomain_legs, domain_legs, tRVB)
    @test Array(ft) ≈ tRVB
  end
end
