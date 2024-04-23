@eval module $(gensym())
using NDTensors.GradedAxes: dual
using NDTensors.Sectors:
  Fib, Ising, SU, SU2, U1, Z, adjoint, quantum_dimension, fundamental, istrivial, trivial
using Test: @inferred, @test, @testset
@testset "Test Category Types" begin
  @testset "U(1)" begin
    q1 = U1(1)
    q2 = U1(2)
    q3 = U1(3)

    @test quantum_dimension(q1) == 1
    @test quantum_dimension(q2) == 1
    @test (@inferred quantum_dimension(q1)) == 1

    @test trivial(U1) == U1(0)
    @test istrivial(U1(0))

    @test dual(U1(2)) == U1(-2)
    @test isless(U1(1), U1(2))
    @test !isless(U1(2), U1(1))
  end

  @testset "Z₂" begin
    z0 = Z{2}(0)
    z1 = Z{2}(1)

    @test trivial(Z{2}) == Z{2}(0)
    @test istrivial(Z{2}(0))

    @test quantum_dimension(z0) == 1
    @test quantum_dimension(z1) == 1
    @test (@inferred quantum_dimension(z0)) == 1

    @test dual(z0) == z0
    @test dual(z1) == z1

    @test dual(Z{2}(1)) == Z{2}(1)
    @test isless(Z{2}(0), Z{2}(1))
    @test !isless(Z{2}(1), Z{2}(0))
  end

  @testset "SU2" begin
    j1 = SU2(0)
    j2 = SU2(1//2)
    j3 = SU2(1)
    j4 = SU2(3//2)

    # alternative tuple constructor
    @test j1 == SU{2}((0,))
    @test j2 == SU{2}((1,))
    @test j3 == SU{2}((2,))
    @test j4 == SU{2}((3,))

    @test trivial(SU2) == SU2(0)
    @test istrivial(SU2(0))

    @test fundamental(SU2) == SU2(1//2)
    @test adjoint(SU2) == SU2(1)

    @test quantum_dimension(j1) == 1
    @test quantum_dimension(j2) == 2
    @test quantum_dimension(j3) == 3
    @test quantum_dimension(j4) == 4
    @test (@inferred quantum_dimension(j1)) == 1

    @test dual(j1) == j1
    @test dual(j2) == j2
    @test dual(j3) == j3
    @test dual(j4) == j4
  end

  @testset "SU(N)" begin
    f3 = SU{3}((1, 0))
    f4 = SU{4}((1, 0, 0))
    ad3 = SU{3}((2, 1))
    ad4 = SU{4}((2, 1, 1))

    @test trivial(SU{3}) == SU{3}((0, 0))
    @test istrivial(SU{3}((0, 0)))
    @test trivial(SU{4}) == SU{4}((0, 0, 0))
    @test istrivial(SU{4}((0, 0, 0)))

    @test fundamental(SU{3}) == f3
    @test adjoint(SU{3}) == ad3
    @test fundamental(SU{4}) == f4
    @test adjoint(SU{4}) == ad4

    @test dual(f3) == SU{3}((1, 1))
    @test dual(f4) == SU{4}((1, 1, 1))
    @test dual(ad3) == ad3
    @test dual(ad4) == ad4

    @test quantum_dimension(f3) == 3
    @test quantum_dimension(f4) == 4
    @test quantum_dimension(ad3) == 8
    @test quantum_dimension(ad4) == 15
    @test quantum_dimension(SU{3}((4, 2))) == 27
    @test quantum_dimension(SU{3}((3, 3))) == 10
    @test quantum_dimension(SU{3}((3, 0))) == 10
    @test quantum_dimension(SU{3}((0, 0))) == 1
    @test (@inferred quantum_dimension(f3)) == 3
  end

  @testset "Fibonacci" begin
    ı = Fib("1")
    τ = Fib("τ")

    @test trivial(Fib) == ı
    @test istrivial(ı)

    @test dual(ı) == ı
    @test dual(τ) == τ

    @test (@inferred quantum_dimension(ı)) == 1.0
    @test (@inferred quantum_dimension(τ)) == ((1 + √5) / 2)
  end

  @testset "Ising" begin
    ı = Ising("1")
    σ = Ising("σ")
    ψ = Ising("ψ")

    @test trivial(Ising) == ı
    @test istrivial(ı)

    @test dual(ı) == ı
    @test dual(σ) == σ
    @test dual(ψ) == ψ

    @test (@inferred quantum_dimension(ı)) == 1.0
    @test (@inferred quantum_dimension(σ)) == √2
    @test (@inferred quantum_dimension(ψ)) == 1.0
  end
end
end
