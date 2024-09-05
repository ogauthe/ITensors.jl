@eval module $(gensym())
using LinearAlgebra: LinearAlgebra
using Test: @test, @testset, @test_broken

using BlockArrays: BlockArrays

using NDTensors.FusionTensors: FusionTensor, check_sanity, data_matrix
using NDTensors.GradedAxes: GradedAxes
using NDTensors.Sectors: O2, SU2, U1, sector

@testset "Empty FusionTensor" begin
  @testset "trivial matrix" begin
    g = GradedAxes.gradedrange([sector() => 1])
    gb = GradedAxes.dual(g)
    m = ones((1, 1))
    ft = FusionTensor(m, (gb,), (g,))
    @test size(data_matrix(ft)) == (1, 1)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[1, 1] ≈ 1.0
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ m
    @test Array(adjoint(ft)) ≈ m
  end

  @testset "several axes, one block" begin
    g1 = GradedAxes.gradedrange([sector() => 2])
    g2 = GradedAxes.gradedrange([sector() => 3])
    g3 = GradedAxes.gradedrange([sector() => 4])
    g4 = GradedAxes.gradedrange([sector() => 2])
    domain_legs = GradedAxes.dual.((g1, g2))
    codomain_legs = (g3, g4)
    t = convert.(Float64, reshape(collect(1:48), (2, 3, 4, 2)))
    ft = FusionTensor(t, domain_legs, codomain_legs)
    @test size(data_matrix(ft)) == (6, 8)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[BlockArrays.Block(1, 1)] ≈ reshape(t, (6, 8))
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ t
    @test Array(adjoint(ft)) ≈ permutedims(t, (3, 4, 1, 2))
  end
end

@testset "Abelian FusionTensor" begin
  @testset "trivial matrix" begin
    g = GradedAxes.gradedrange([U1(0) => 1])
    gb = GradedAxes.dual(g)
    m = ones((1, 1))
    ft = FusionTensor(m, (gb,), (g,))
    @test size(data_matrix(ft)) == (1, 1)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[1, 1] ≈ 1.0
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ m
    @test Array(adjoint(ft)) ≈ m
  end

  @testset "non self-conjugate matrix" begin
    g = GradedAxes.gradedrange([U1(1) => 2])
    gb = GradedAxes.dual(g)
    m = ones((2, 2))
    ft = FusionTensor(m, (gb,), (g,))
    @test size(data_matrix(ft)) == (2, 2)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[BlockArrays.Block(1, 1)] ≈ m
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ m
    @test Array(adjoint(ft)) ≈ m
  end

  @testset "several axes, one block" begin
    g1 = GradedAxes.gradedrange([U1(1) => 2])
    g2 = GradedAxes.gradedrange([U1(2) => 3])
    g3 = GradedAxes.gradedrange([U1(3) => 4])
    g4 = GradedAxes.gradedrange([U1(0) => 2])
    domain_legs = GradedAxes.dual.((g1, g2))
    codomain_legs = (g3, g4)
    t = convert.(Float64, reshape(collect(1:48), (2, 3, 4, 2)))
    ft = FusionTensor(t, domain_legs, codomain_legs)
    @test size(data_matrix(ft)) == (6, 8)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[BlockArrays.Block(1, 1)] ≈ reshape(t, (6, 8))
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ t
    @test Array(adjoint(ft)) ≈ permutedims(t, (3, 4, 1, 2))
  end

  @testset "several axes, several blocks" begin
    g1 = GradedAxes.gradedrange([U1(1) => 2, U1(2) => 2])
    g2 = GradedAxes.gradedrange([U1(2) => 3, U1(3) => 2])
    g3 = GradedAxes.gradedrange([U1(3) => 4, U1(4) => 1])
    g4 = GradedAxes.gradedrange([U1(0) => 2, U1(2) => 1])
    domain_legs = GradedAxes.dual.((g1, g2))
    codomain_legs = (g3, g4)
    dense = zeros((4, 5, 5, 3))
    dense[1:2, 1:3, 1:4, 1:2] .= 1.0
    dense[3:4, 1:3, 5:5, 1:2] .= 2.0
    dense[1:2, 4:5, 5:5, 1:2] .= 3.0
    dense[3:4, 4:5, 1:4, 3:3] .= 4.0
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test size(data_matrix(ft)) == (20, 15)
    @test BlockArrays.blocksize(data_matrix(ft)) == (3, 4)
    @test LinearAlgebra.norm(ft) ≈ LinearAlgebra.norm(dense)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ dense
    @test Array(adjoint(ft)) ≈ permutedims(dense, (3, 4, 1, 2))
  end

  @testset "mixing dual and nondual" begin
    g1 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])
    g2 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
    g3 = GradedAxes.gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
    g4 = GradedAxes.gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
    domain_legs = (g1,)
    codomain_legs = (GradedAxes.dual(g2), GradedAxes.dual(g3), g4)
    dense = zeros((3, 6, 5, 4))
    dense[2:2, 1:1, 1:2, 2:3] .= 1.0
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test size(data_matrix(ft)) == (3, 120)
    @test BlockArrays.blocksize(data_matrix(ft)) == (3, 8)
    @test LinearAlgebra.norm(ft) ≈ LinearAlgebra.norm(dense)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ dense
    @test Array(adjoint(ft)) ≈ permutedims(dense, (2, 3, 4, 1))
  end

  @testset "Less than 2 axes" begin
    g1 = GradedAxes.gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
    v1 = zeros((6,))
    v1[1] = 1.0

    ft1 = FusionTensor(v1, (g1,), ())
    @test isnothing(check_sanity(ft1))
    @test Array(ft1) ≈ v1

    ft2 = FusionTensor(v1, (), (g1,))
    @test isnothing(check_sanity(ft2))
    @test Array(ft2) ≈ v1

    zerodim = ones(())
    @test_broken FusionTensor(zerodim, (), ()) isa FusionTensor  # https://github.com/JuliaArrays/BlockArrays.jl/issues/409
  end
end

@testset "O(2) FusionTensor" begin
  @testset "trivial matrix" begin
    g = GradedAxes.gradedrange([O2(0) => 1])
    gb = GradedAxes.dual(g)
    m = ones((1, 1))
    ft = FusionTensor(m, (gb,), (g,))
    @test size(data_matrix(ft)) == (1, 1)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[1, 1] ≈ 1.0
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ m
    @test Array(adjoint(ft)) ≈ m
  end

  @testset "spin 1/2 S.S" begin
    g2 = GradedAxes.gradedrange([O2(1//2) => 1])
    g2b = GradedAxes.dual(g2)

    # identity
    id2 = LinearAlgebra.I((2))
    ft = FusionTensor(id2, (g2b,), (g2,))
    @test LinearAlgebra.norm(ft) ≈ √2
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ id2
    @test Array(adjoint(ft)) ≈ id2

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
    dense, domain_legs, codomain_legs = sds22, (g2b, g2b), (g2, g2)
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test LinearAlgebra.norm(ft) ≈ √3 / 2
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22

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
    sds22b_codomain_legs = (g2, g2b)
    dense, domain_legs, codomain_legs = sds22b, (g2, g2b), (g2b, g2)
    ftb = FusionTensor(dense, domain_legs, codomain_legs)
    @test LinearAlgebra.norm(ftb) ≈ √3 / 2
    @test isnothing(check_sanity(ft))
    @test Array(ftb) ≈ sds22b
    @test Array(adjoint(ftb)) ≈ sds22b

    # no codomain axis
    dense, domain_legs, codomain_legs = sds22, (g2b, g2b, g2, g2), ()
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22

    # no domain axis
    dense, domain_legs, codomain_legs = sds22, (), (g2b, g2b, g2, g2)
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22
  end
end

@testset "SU(2) FusionTensor" begin
  @testset "trivial matrix" begin
    g = GradedAxes.gradedrange([SU2(0) => 1])
    gb = GradedAxes.dual(g)
    m = ones((1, 1))
    ft = FusionTensor(m, (gb,), (g,))
    @test size(data_matrix(ft)) == (1, 1)
    @test BlockArrays.blocksize(data_matrix(ft)) == (1, 1)
    @test data_matrix(ft)[1, 1] ≈ 1.0
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ m
    @test Array(adjoint(ft)) ≈ m
  end

  @testset "spin 1/2 S.S" begin
    g2 = GradedAxes.gradedrange([SU2(1 / 2) => 1])
    g2b = GradedAxes.dual(g2)

    # identity
    id2 = LinearAlgebra.I((2))
    ft = FusionTensor(id2, (g2b,), (g2,))
    @test LinearAlgebra.norm(ft) ≈ √2
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ id2
    @test Array(adjoint(ft)) ≈ id2

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
    dense, domain_legs, codomain_legs = sds22, (g2b, g2b), (g2, g2)
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test LinearAlgebra.norm(ft) ≈ √3 / 2
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22

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
    sds22b_codomain_legs = (g2, g2b)
    dense, domain_legs, codomain_legs = sds22b, (g2, g2b), (g2b, g2)
    ftb = FusionTensor(dense, domain_legs, codomain_legs)
    @test LinearAlgebra.norm(ftb) ≈ √3 / 2
    @test isnothing(check_sanity(ft))
    @test Array(ftb) ≈ sds22b
    @test Array(adjoint(ftb)) ≈ sds22b

    # no codomain axis
    dense, domain_legs, codomain_legs = sds22, (g2b, g2b, g2, g2), ()
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22

    # no domain axis
    dense, domain_legs, codomain_legs = sds22, (), (g2b, g2b, g2, g2)
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ sds22
    @test Array(adjoint(ft)) ≈ sds22
  end

  @testset "large identity" begin
    g = reduce(GradedAxes.fusion_product, (SU2(1 / 2), SU2(1 / 2), SU2(1 / 2)))
    N = 3
    codomain_legs = ntuple(_ -> g, N)
    domain_legs = GradedAxes.dual.(codomain_legs)
    d = 8
    dense = reshape(LinearAlgebra.I(d^N), ntuple(_ -> d, 2 * N))
    ft = FusionTensor(dense, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ dense
    @test Array(adjoint(ft)) ≈ dense
  end
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
    domain_legs = (GradedAxes.dual(gd),)
    gD = GradedAxes.gradedrange([sector(SU2(0), U1(1)) => 1, sector(s, U1(0)) => 1])
    codomain_legs = (gD, gD, gD, gD)
    ft = FusionTensor(tRVB, domain_legs, codomain_legs)
    @test isnothing(check_sanity(ft))
    @test Array(ft) ≈ tRVB

    # same with NamedTuples
    gd_nt = GradedAxes.gradedrange([sector(; S=s, N=U1(3)) => 1])
    domain_legs_nt = (GradedAxes.dual(gd_nt),)
    gD_nt = GradedAxes.gradedrange([
      sector(; S=SU2(0), N=U1(1)) => 1, sector(; S=s, N=U1(0)) => 1
    ])
    codomain_legs_nt = (gD_nt, gD_nt, gD_nt, gD_nt)
    ft_nt = FusionTensor(tRVB, domain_legs_nt, codomain_legs_nt)
    @test isnothing(check_sanity(ft_nt))
    @test Array(ft_nt) ≈ tRVB
  end
end
end
