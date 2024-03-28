@eval module $(gensym())
using NDTensors.LabelledNumbers: LabelledInteger, islabelled, label, labelled, unlabel
using Test: @test, @testset
@testset "LabelledNumbers" begin
  @testset "Labelled number ($n)" for n in (2, 2.0)
    x = labelled(2, "x")
    @test typeof(x) == LabelledInteger{Int,String}
    @test islabelled(x)
    @test x == 2
    @test label(x) == "x"
    @test unlabel(x) == 2
    @test !islabelled(unlabel(x))

    @test x * 2 == 4
    @test !islabelled(x * 2)
    @test 2 * x == 4
    @test !islabelled(2 * x)
    @test x * x == 4
    @test !islabelled(x * x)

    @test x + 3 == 5
    @test !islabelled(x + 3)
    @test 3 + x == 5
    @test !islabelled(3 + x)
    @test x + x == 4
    @test !islabelled(x + x)

    @test x - 3 == -1
    @test !islabelled(x - 3)
    @test 3 - x == 1
    @test !islabelled(3 - x)
    @test x - x == 0
    @test !islabelled(x - x)

    @test x / 2 == 1
    @test !islabelled(x / 2)
    @test x ÷ 2 == 1
    @test !islabelled(x ÷ 2)
    @test -x == -2
    @test hash(x) == hash(2)
    @test zero(x) == false
    @test label(zero(x)) == "x"
    @test one(x) == true
    @test !islabelled(one(x))
    @test oneunit(x) == true
    @test label(oneunit(x)) == "x"
    @test islabelled(oneunit(x))
    @test one(typeof(x)) == true
    @test !islabelled(one(typeof(x)))
  end
  @testset "Labelled array ($a)" for a in (collect(2:5), 2:5)
    x = labelled(a, "x")
    @test eltype(x) == LabelledInteger{Int,String}
    @test x == 2:5
    @test label(x) == "x"
    @test unlabel(x) == 2:5
    @test first(iterate(x, 3)) == 4
    @test label(first(iterate(x, 3))) == "x"
    @test collect(x) == 2:5
    @test label.(collect(x)) == fill("x", 4)
    @test x[2] == 3
    @test label(x[2]) == "x"
    @test x[2:3] == 3:4
    @test label(x[2:3]) == "x"
    @test x[[2, 4]] == [3, 5]
    @test label(x[[2, 4]]) == "x"

    if x isa AbstractUnitRange
      @test step(x) == true
      @test islabelled(step(x))
      @test label(step(x)) == "x"
    end
  end
end
end
