module FusionTensors

using LinearAlgebra: LinearAlgebra, Adjoint, norm, tr

using BlockArrays:
  AbstractBlockArray,
  AbstractBlockMatrix,
  Block,
  BlockArray,
  BlockedArray,
  BlockMatrix,
  blockedrange,
  blocklength,
  blocklengths,
  blocks

using NDTensors.BlockSparseArrays:
  BlockSparseArrays, BlockSparseArray, BlockSparseMatrix, stored_indices
using NDTensors.GradedAxes:
  GradedAxes,
  AbstractGradedUnitRange,
  blocklabels,
  dual,
  fusion_product,
  gradedrange,
  isdual,
  space_isequal
using NDTensors.SymmetrySectors:
  AbstractSector, TrivialSector, block_dimensions, istrivial, quantum_dimension, trivial
using NDTensors.TensorAlgebra:
  TensorAlgebra, BlockedPermutation, blockedperm, blockpermute, contract, contract!

include("fusiontrees/clebsch_gordan.jl")
include("fusiontrees/fusion_tree.jl")

include("fusiontensor/fusedaxes.jl")
include("fusiontensor/fusiontensor.jl")
include("fusiontensor/base_interface.jl")
include("fusiontensor/array_cast.jl")
include("fusiontensor/linear_algebra_interface.jl")
include("fusiontensor/tensor_algebra_interface.jl")
include("permutedims/unitaries.jl")
include("permutedims/permutedims.jl")
end
