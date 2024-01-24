# This file defines block interface for FusionTensor

using BlockArrays
using NDTensors.FusionTensors: FusionTensor
using ITensors: @debug_check

# TODO iterator over *initialized* blocks
function BlockArrays.blocks(t::FusionTensor)
  return BlockArrays.blocks(matrix_blocks)
end

# TBD return blocklength(t.matrix_blocks), authorized blocks or initialized blocks?
# has to match BlockSparseArray convention
BlockArrays.blocklength(t::FusionTensor) = blocklength(t.matrix_blocks)
BlockArrays.blocklength(t::FusionTensor) = nstored(blocks(t.matrix_blocks))  # TODO update w. BlockSparseArray
