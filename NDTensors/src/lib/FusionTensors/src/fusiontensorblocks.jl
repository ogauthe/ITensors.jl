# This file defines block interface for FusionTensor

using BlockArrays
using NDTensors.FusionTensors: FusionTensor
using ITensors: @debug_check

# TODO iterator over *initialized* blocks
function BlockArrays.blocks(ft::FusionTensor)
  return BlockArrays.blocks(matrix(ft))
end

# TBD return blocklength(t.matrix_blocks), authorized blocks or initialized blocks?
# has to match BlockSparseArray convention
BlockArrays.blocklength(ft::FusionTensor) = blocklength(matrix(ft))
BlockArrays.blocklength(ft::FusionTensor) = nstored(blocks(matrix(ft)))  # TODO update w. BlockSparseArray
