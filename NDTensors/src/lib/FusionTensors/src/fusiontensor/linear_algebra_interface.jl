# This file defines linalg for FusionTensor

using LinearAlgebra

using BlockArrays: blocks

using NDTensors.BlockSparseArrays: stored_indices
using NDTensors.GradedAxes: sectors
using NDTensors.Sectors: dimension

# simpler to define as Frobenius norm(block) than Tr(t^dagger * t)
function LinearAlgebra.norm(ft::FusionTensor)
  n2 = 0.0
  m = data_matrix(ft)
  row_sectors = sectors(axes(m)[1])
  col_sectors = sectors(axes(m)[2])
  for idx in stored_indices(blocks(m))  # TODO update interface?
    nb = norm(m[Block(Tuple(idx))])
    # do not assume row_sector == col_sector (may be false for equivariant tensor)
    dr = dimension(row_sectors[idx[1]])
    dc = dimension(col_sectors[idx[2]])
    n2 += sqrt(dr) * sqrt(dc) * nb^2
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
