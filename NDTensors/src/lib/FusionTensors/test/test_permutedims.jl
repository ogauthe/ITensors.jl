@eval module $(gensym())
using Test: @test, @testset, @test_broken

using NDTensors.FusionTensors:
  FusionTensor,
  check_sanity,
  data_matrix,
  matching_axes,
  matrix_column_axis,
  matrix_row_axis,
  naive_permutedims,
  ndims_domain,
  ndims_codomain
using NDTensors.GradedAxes: dual, gradedrange, space_isequal
using NDTensors.SymmetrySectors: O2, U1, SectorProduct, SU2
using NDTensors.TensorAlgebra: blockedperm

@testset "Abelian permutedims" begin
  @testset "dummy" begin
    g1 = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
    g2 = gradedrange([U1(0) => 2, U1(1) => 2, U1(3) => 1])
    g3 = gradedrange([U1(-1) => 1, U1(0) => 2, U1(1) => 1])
    g4 = gradedrange([U1(-1) => 1, U1(0) => 1, U1(1) => 1])

    for elt in (Float64, ComplexF64)
      ft1 = FusionTensor(elt, dual.((g1, g2)), (g3, g4))
      @test isnothing(check_sanity(ft1))

      # test permutedims interface
      ft2 = permutedims(ft1, (1, 2), (3, 4))   # trivial with 2 tuples
      @test ft2 === ft1  # same object

      ft2 = permutedims(ft1, ((1, 2), (3, 4)))   # trivial with tuple of 2 tuples
      @test ft2 === ft1  # same object

      biperm = blockedperm((1, 2), (3, 4))
      ft2 = permutedims(ft1, biperm)   # trivial with biperm
      @test ft2 === ft1  # same object

      ft3 = permutedims(ft1, (4,), (1, 2, 3))
      @test ft3 !== ft1
      @test ft3 isa FusionTensor{elt,4}
      @test matching_axes(axes(ft3), (g4, dual(g1), dual(g2), g3))
      @test ndims_domain(ft3) == 1
      @test ndims_codomain(ft3) == 3
      @test ndims(ft3) == 4
      @test isnothing(check_sanity(ft3))

      ft4 = permutedims(ft3, (2, 3), (4, 1))
      @test matching_axes(axes(ft1), axes(ft4))
      @test space_isequal(matrix_column_axis(ft1), matrix_column_axis(ft4))
      @test space_isequal(matrix_row_axis(ft1), matrix_row_axis(ft4))
      @test ft4 ≈ ft1
    end
  end

  @testset "Many axes" begin
    g1 = gradedrange([U1(1) => 2, U1(2) => 2])
    g2 = gradedrange([U1(2) => 3, U1(3) => 2])
    g3 = gradedrange([U1(3) => 4, U1(4) => 1])
    g4 = gradedrange([U1(0) => 2, U1(2) => 1])
    domain_legs = (g1, g2)
    codomain_legs = dual.((g3, g4))
    arr = zeros(ComplexF64, (4, 5, 5, 3))
    arr[1:2, 1:3, 1:4, 1:2] .= 1.0im
    arr[3:4, 1:3, 5:5, 1:2] .= 2.0
    arr[1:2, 4:5, 5:5, 1:2] .= 3.0
    arr[3:4, 4:5, 1:4, 3:3] .= 4.0
    ft = FusionTensor(arr, domain_legs, codomain_legs)
    biperm = blockedperm((3,), (2, 4, 1))

    ftp = permutedims(ft, biperm)
    @test ftp ≈ naive_permutedims(ft, biperm)
    ftpp = permutedims(ftp, (4, 2), (1, 3))
    @test ftpp ≈ ft

    ft2 = adjoint(ft)
    ftp2 = permutedims(ft2, biperm)
    @test ftp2 ≈ naive_permutedims(ft2, biperm)
    ftpp2 = permutedims(ftp2, (4, 2), (1, 3))
    @test ftpp2 ≈ ft2
    @test adjoint(ftpp2) ≈ ft
  end

  @testset "Less than two axes" begin
    if VERSION >= v"1.11"
      ft0 = FusionTensor(ones(()), (), ())
      ft0p = permutedims(ft0, (), ())
      @test ft0p isa FusionTensor{Float64,0}
      @test data_matrix(ft0p) ≈ data_matrix(ft0)
      @test ft0p ≈ ft0

      @test permutedims(ft0, ((), ())) isa FusionTensor{Float64,0}
      @test permutedims(ft0, blockedperm((), ())) isa FusionTensor{Float64,0}
    end

    g = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
    v = zeros((6,))
    v[1] = 1.0
    biperm = blockedperm((), (1,))
    ft1 = FusionTensor(v, (g,), ())
    @test_broken permutedims(ft1, biperm) isa FusionTensor
    #ft2 = permutedims(ft1, biperm) isa FusionTensor
    #@test isnothing(check_sanity(ft2))
    #@test ft2 ≈ naive_permutedims(ft1, biperm)
    #ft3 = permutedims(ft2, (1,), ())
    #@test ft1 ≈ ft3
  end
end

@testset "Non-abelian permutedims" begin
  sds22 = reshape(
    [
      [0.25, 0.0, 0.0, 0.0]
      [0.0, -0.25, 0.5, 0.0]
      [0.0, 0.5, -0.25, 0.0]
      [0.0, 0.0, 0.0, 0.25]
    ],
    (2, 2, 2, 2),
  )

  sds22b = reshape(
    [
      [-0.25, 0.0, 0.0, -0.5]
      [0.0, 0.25, 0.0, 0.0]
      [0.0, 0.0, 0.25, 0.0]
      [-0.5, 0.0, 0.0, -0.25]
    ],
    (2, 2, 2, 2),
  )

  for g2 in (
    gradedrange([O2(1//2) => 1]),
    dual(gradedrange([O2(1//2) => 1])),
    gradedrange([SU2(1//2) => 1]),
    dual(gradedrange([SU2(1//2) => 1])),
    gradedrange([SectorProduct(SU2(1//2), U1(0)) => 1]),
    gradedrange([SectorProduct(SU2(1//2), SU2(0)) => 1]),
  )
    g2b = dual(g2)
    for biperm in [
      blockedperm((2, 1), (3, 4)), blockedperm((3, 1), (2, 4)), blockedperm((3, 1, 4), (2,))
    ]
      ft = FusionTensor(sds22, (g2, g2), (g2b, g2b))
      @test permutedims(ft, biperm) ≈ naive_permutedims(ft, biperm)
      @test permutedims(adjoint(ft), biperm) ≈ naive_permutedims(adjoint(ft), biperm)

      ft = FusionTensor(sds22b, (g2, g2b), (g2, g2b))
      @test permutedims(ft, biperm) ≈ naive_permutedims(ft, biperm)
      @test permutedims(adjoint(ft), biperm) ≈ naive_permutedims(adjoint(ft), biperm)
    end
    for biperm in [blockedperm((1, 2, 3, 4), ()), blockedperm((), (3, 1, 2, 4))]
      ft = FusionTensor(sds22, (g2, g2), (g2b, g2b))
      @test_broken permutedims(ft, biperm) ≈ naive_permutedims(ft, biperm)
    end
  end
end
end
