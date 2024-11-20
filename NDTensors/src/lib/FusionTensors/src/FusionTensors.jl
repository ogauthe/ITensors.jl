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
using LRUCache: LRU

using NDTensors.BlockSparseArrays:
  AbstractBlockSparseMatrix,
  BlockSparseArrays,
  BlockSparseArray,
  BlockSparseMatrix,
  stored_indices,
  view!
using NDTensors.GradedAxes:
  GradedAxes,
  AbstractGradedUnitRange,
  blocklabels,
  blockmergesort,
  dual,
  fusion_product,
  gradedrange,
  isdual,
  labelled_blocks,
  space_isequal,
  unlabel_blocks
using NDTensors.SymmetrySectors:
  AbstractSector, TrivialSector, block_dimensions, istrivial, quantum_dimension, trivial
using NDTensors.TensorAlgebra:
  TensorAlgebra, BlockedPermutation, blockedperm, blockpermute, contract, contract!
using NDTensors.LabelledNumbers: label_type # TODO avoid calling LabelledNumbers

include("fusion_trees/fusiontree.jl")
include("fusion_trees/clebsch_gordan_tensors.jl")
include("fusion_trees/fusion_tree_tensors.jl")

include("fusiontensor/fusedaxes.jl")
include("fusiontensor/fusiontensor.jl")
include("fusiontensor/base_interface.jl")
include("fusiontensor/array_cast.jl")
include("fusiontensor/linear_algebra_interface.jl")
include("fusiontensor/tensor_algebra_interface.jl")
include("permutedims/unitaries.jl")
include("permutedims/permutedims.jl")
end
