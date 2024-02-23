# This file defines linalg for FusionTensor

using LinearAlgebra
using NDTensors.FusionTensors: FusionTensor

# simpler to define as Frobenius norm(block) than Tr(t^dagger * t)
function LinearAlgebra.norm(ft::FusionTensor)
  n2 = 0.0
  for i in blocks(ft)
    n2 += dimension(sector(row_axis(ft)[Block(i)])) * norm(matrix(ft)[Block(i)])^2
  end
  return sqrt(n2)
end

function LinearAlgebra.qr(ft::FusionTensor)
  q, r = block_qr(matrix(ft))
  mid_axis = nothing  # TODO
  qtens = FusionTensor(codomain_axes(ft), (mid_axis,), q)
  rtens = FusionTensor((mid_axis,), domain_axes(ft), r)
  return qtens, rtens
end

function LinearAlgebra.svd(ft::FusionTensor)
  u, s, v = block_svd(matrix(ft))
  mid_axis = nothing  # TODO
  utens = FusionTensor(codomain_axes(ft), (mid_axis,), u)
  stens = FusionTensor((mid_axis,), (mid_axis,), s)  # TODO struct DiagonalBlockMatrix
  vtens = FusionTensor((mid_axis,), domain_axes(ft), v)
  return utens, stens, vtens
end
