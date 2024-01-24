# This file defines linalg for FusionTensor

using LinearAlgebra
using NDTensors.FusionTensors: FusionTensor

# simpler to define as Frobenius norm(block) than Tr(t^dagger * t)
function LinearAlgebra.norm(t::FusionTensor)
  n2 = 0.0
  for i in 1:(t.nblocks)
    n2 += dimension(sector(t.matrix_row_axis, Block(i))) * norm(t.blocks[i])^2
  end
  return sqrt(n2)
end

function LinearAlgebra.qr(t::FusionTensor)
  q, r = block_qr(t.matrix_blocks)
  mid_axis = TODO
  qtens = FusionTensor(t.codomain_axes, (mid_axis,), q)
  rtens = FusionTensor((mid_axis,), t.domain_axes, r)
  return qtens, rtens
end

function LinearAlgebra.svd(t::FusionTensor)
  u, s, v = block_svd(t.matrix_blocks)
  mid_axis = TODO
  utens = FusionTensor(t.codomain_axes, (mid_axis,), u)
  stens = FusionTensor((mid_axis,), (mid_axis,), s)  # TODO struct DiagonalBlockMatrix
  vtens = FusionTensor((mid_axis,), t.domain_axes, v)
  return utens, stens, vtens
end
