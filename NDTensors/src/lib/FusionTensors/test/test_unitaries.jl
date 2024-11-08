@eval module $(gensym())
using Test: @test, @testset
using LinearAlgebra: LinearAlgebra

using BlockArrays: AbstractBlockMatrix, blocksize

using NDTensors.FusionTensors:
  FusionTensor,
  domain_axes,
  codomain_axes,
  data_matrix,
  matching_axes,
  matching_dual,
  matrix_column_axis,
  matrix_row_axis,
  matrix_size,
  ndims_domain,
  ndims_codomain,
  check_sanity,
  check_data_matrix_axes,
  find_shared_indices,
  FusedAxes,
  allowed_outer_blocks_sectors,
  get_tree!,
  FusionTensors,
  overlap_fusion_trees,
  compute_unitaries_clebsch_gordan
using NDTensors.GradedAxes: gradedrange, dual, GradedAxes
using NDTensors.SymmetrySectors: SymmetrySectors, SU, SU2, TrivialSector, U1, Z
using NDTensors.TensorAlgebra: blockedperm, TensorAlgebra

@testset "Trivial unitaries" begin
  function check_trivial_unitary(u)
    @test u isa AbstractBlockMatrix
    @test blocksize(u) == (1, 1)
    @test size(u) == (1, 1)
    @test u[1, 1] == 1.0
  end

  g = gradedrange([TrivialSector() => 1])
  old_domain_legs = (g, g)
  old_codomain_legs = (g,)
  biperm = blockedperm((2,), (3, 1))
  u_d = compute_unitaries_clebsch_gordan(old_domain_legs, old_codomain_legs, biperm)
  @test collect(keys(u_d)) == [(1, 1, 1)]
  check_trivial_unitary(u_d[1, 1, 1])

  g = gradedrange([U1(0) => 1, U1(1) => 2, U1(2) => 3])
  biperm = blockedperm((2,), (3, 1, 4))
  u_d = compute_unitaries_clebsch_gordan((g, g), (dual(g), dual(g)), biperm)
  @test length(u_d) == 19
  check_trivial_unitary.(values(u_d))
end

@testset "SU(2) unitaries" begin
  g12 = gradedrange([SU2(1 / 2) => 1])
  g1 = gradedrange([SU2(1) => 1])
  biperm = blockedperm((2,), (3, 1))
  u_d = compute_unitaries_clebsch_gordan((g12, g12), (g1,), biperm)
  @test collect(keys(u_d)) == [(1, 1, 1)]
  @test u_d[1, 1, 1] ≈ -√(3 / 2) * ones((1, 1))
end

@testset "SectorProduct unitaries" begin end
end
