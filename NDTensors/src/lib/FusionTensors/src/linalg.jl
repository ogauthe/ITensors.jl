# This file defines linalg for FusionTensor

using LinearAlgebra
using NDTensors.FusionTensors: FusionTensor

# simpler to define as Frobenius norm(block) than Tr(t^dagger * t)
function LinearAlgebra.norm(ft::FusionTensor)
  n2 = 0.0
  for i in blocks(ft)  # TODO
    n2 += dimension(sector(row_axis(ft)[Block(i)])) * norm(matrix(ft)[Block(i)])^2
  end
  return sqrt(n2)
end

function LinearAlgebra.qr(ft::FusionTensor)
  qmat, rmat = block_qr(matrix(ft))
  qtens = FusionTensor(codomain_axes(ft), (axes(qmat)[1],), qmat)
  rtens = FusionTensor((axes(rmat)[0],), domain_axes(ft), rmat)
  return qtens, rtens
end

function LinearAlgebra.svd(ft::FusionTensor)
  umat, s, vmat = block_svd(matrix(ft))
  utens = FusionTensor(codomain_axes(ft), (axes(umat)[1],), umat)
  stens = FusionTensor((axes(umat)[1],), (axes(vmat)[0],), s)
  vtens = FusionTensor((axes(vmat)[0],), domain_axes(ft), vmat)
  return utens, stens, vtens
end
