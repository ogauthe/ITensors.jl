@eval module $(gensym())
using BlockArrays: Block, blockaxes, blockfirsts, blocklasts, blocklength, blocks, findblock
using NDTensors.GradedAxes:
  GradedAxes,
  UnitRangeDual,
  blockmergesortperm,
  blocksortperm,
  dual,
  gradedrange,
  isdual,
  nondual
using NDTensors.LabelledNumbers: LabelledInteger, label, labelled
using Test: @test, @test_broken, @testset
struct U1
  n::Int
end
GradedAxes.dual(c::U1) = U1(-c.n)
Base.isless(c1::U1, c2::U1) = c1.n < c2.n
@testset "dual" begin
  a = gradedrange([U1(0) => 2, U1(1) => 3])
  ad = dual(a)
  @test eltype(ad) == LabelledInteger{Int,U1}
  @test dual(ad) == a
  @test nondual(ad) == a
  @test nondual(a) == a
  @test isdual(ad)
  @test !isdual(a)
  @test blockfirsts(ad) == [labelled(1, U1(0)), labelled(3, U1(-1))]
  @test blocklasts(ad) == [labelled(2, U1(0)), labelled(5, U1(-1))]
  @test findblock(ad, 4) == Block(2)
  @test only(blockaxes(ad)) == Block(1):Block(2)
  @test blocks(ad) == [labelled(1:2, U1(0)), labelled(3:5, U1(-1))]
  @test ad[4] == 4
  @test label(ad[4]) == U1(-1)
  @test ad[2:4] == 2:4
  @test ad[2:4] isa UnitRangeDual
  @test label(ad[2:4][Block(2)]) == U1(-1)
  @test ad[[2, 4]] == [2, 4]
  @test label(ad[[2, 4]][2]) == U1(-1)
  @test ad[Block(2)] == 3:5
  @test label(ad[Block(2)]) == U1(-1)
  @test ad[Block(1):Block(2)][Block(2)] == 3:5
  @test label(ad[Block(1):Block(2)][Block(2)]) == U1(-1)
  @test ad[[Block(2), Block(1)]][Block(1)] == 3:5
  @test label(ad[[Block(2), Block(1)]][Block(1)]) == U1(-1)
  @test ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)] == 3:4
  @test label(ad[[Block(2)[1:2], Block(1)[1:2]]][Block(1)]) == U1(-1)
  @test blocksortperm(a) == [Block(1), Block(2)]
  @test blocksortperm(ad) == [Block(2), Block(1)]
  @test blocklength(blockmergesortperm(a)) == 2
  @test blocklength(blockmergesortperm(ad)) == 2
  @test blockmergesortperm(a) == [Block(1), Block(2)]
  @test blockmergesortperm(ad) == [Block(2), Block(1)]
end
end
