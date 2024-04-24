# This file defines linalg for FusionTensor

using LinearAlgebra

using BlockArrays

using NDTensors.BlockSparseArrays: stored_indices #, block_qr, block_svd
using NDTensors.Sectors: quantum_dimension

# allow to contract with different eltype and let BlockSparseArray ensure compatibility
# impose matching type and number of axes at compile time
# impose matching axes at run time
function LinearAlgebra.mul!(
  C::FusionTensor{T1,N,NCoAxes,NDoAxes},
  A::FusionTensor{T2,M,NCoAxes,NContractedAxes},
  B::FusionTensor{T3,K,NContractedAxes,NDoAxes},
  α::Number,
  β::Number,
) where {T1,T2,T3,N,M,K,NCoAxes,NDoAxes,NContractedAxes}
  if domain_axes(A) != dual.(codomain_axes(B))
    throw(DomainError("Incompatible tensor axes for A and B"))
  end
  if codomain_axes(C) != codomain_axes(A)
    throw(DomainError("Incompatible tensor axes for C and A"))
  end
  if domain_axes(C) != domain_axes(B)
    throw(DomainError("Incompatible tensor axes for C and B"))
  end
  mul!(data_matrix(C), data_matrix(A), data_matrix(B), α, β)
  return C
end

# the compiler automatically defines LinearAlgebra.mul!(C,A,B)

# simpler to define as Frobenius norm(block) than Tr(t^dagger * t)
function LinearAlgebra.norm(ft::FusionTensor)
  n2 = 0.0
  m = data_matrix(ft)
  row_sectors = BlockArrays.blocklengths(axes(m)[1])
  col_sectors = BlockArrays.blocklengths(axes(m)[2])
  for idx in stored_indices(BlockArrays.blocks(m))  # TODO update interface?
    nb = norm(m[Block(Tuple(idx))])
    # do not assume row_sector == col_sector (may be false for equivariant tensor)
    dr = quantum_dimension(row_sectors[idx[1]])
    dc = quantum_dimension(col_sectors[idx[2]])
    n2 += sqrt(dr * dc) * nb^2
  end
  return sqrt(n2)
end

function LinearAlgebra.qr(ft::FusionTensor)
  qmat, rmat = block_qr(data_matrix(ft))
  qtens = FusionTensor(codomain_axes(ft), (axes(qmat)[1],), qmat)
  rtens = FusionTensor((axes(rmat)[0],), domain_axes(ft), rmat)
  return qtens, rtens
end

function LinearAlgebra.svd(ft::FusionTensor)
  umat, s, vmat = block_svd(data_matrix(ft))
  utens = FusionTensor(codomain_axes(ft), (axes(umat)[1],), umat)
  stens = FusionTensor((axes(umat)[1],), (axes(vmat)[0],), s)
  vtens = FusionTensor((axes(vmat)[0],), domain_axes(ft), vmat)
  return utens, stens, vtens
end
