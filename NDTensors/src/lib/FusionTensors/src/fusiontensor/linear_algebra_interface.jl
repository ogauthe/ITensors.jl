# This file defines linalg for FusionTensor

# allow to contract with different eltype and let BlockSparseArray ensure compatibility
# impose matching type and number of axes at compile time
# impose matching axes at run time
# TODO remove this once TensorAlgebra.contract can be used?
function LinearAlgebra.mul!(
  C::FusionTensor, A::FusionTensor, B::FusionTensor, α::Number, β::Number
)

  # compile time checks
  if ndims_codomain(A) != ndims_domain(B)
    throw(DomainError("Incompatible tensor structures for A and B"))
  end
  if ndims_domain(A) != ndims_domain(C)
    throw(DomainError("Incompatible tensor structures for A and C"))
  end
  if ndims_codomain(B) != ndims_codomain(C)
    throw(DomainError("Incompatible tensor structures for B and C"))
  end

  # input validation
  if !matching_dual(codomain_axes(A), domain_axes(B))
    throw(DomainError("Incompatible tensor axes for A and B"))
  end
  if !matching_axes(domain_axes(C), domain_axes(A))
    throw(DomainError("Incompatible tensor axes for C and A"))
  end
  if !matching_axes(codomain_axes(C), codomain_axes(B))
    throw(DomainError("Incompatible tensor axes for C and B"))
  end
  LinearAlgebra.mul!(data_matrix(C), data_matrix(A), data_matrix(B), α, β)
  return C
end

function LinearAlgebra.norm(ft::FusionTensor)
  m = data_matrix(ft)
  row_sectors = blocklabels(matrix_row_axis(ft))
  n2 = mapreduce(
    idx -> quantum_dimension(row_sectors[idx[1]]) * norm(m[Block(Tuple(idx))])^2,
    +,
    stored_indices(blocks(m));
    init=0.0,
  )
  return sqrt(n2)
end

function LinearAlgebra.tr(ft::FusionTensor)
  m = data_matrix(ft)
  row_sectors = blocklabels(matrix_row_axis(ft))
  return mapreduce(
    idx -> quantum_dimension(row_sectors[idx[1]]) * tr(m[Block(Tuple(idx))]),
    +,
    stored_indices(blocks(m));
    init=eltype(ft)(0),
  )
end

function LinearAlgebra.qr(ft::FusionTensor)
  qmat, rmat = BlockSparseArrays.block_qr(data_matrix(ft))
  qtens = FusionTensor(qmat, domain_axes(ft), (axes(qmat)[1],))
  rtens = FusionTensor(rmat, (axes(rmat)[0],), codomain_axes(ft))
  return qtens, rtens
end

function LinearAlgebra.svd(ft::FusionTensor)
  umat, s, vmat = BlockSparseArrays.block_svd(data_matrix(ft))
  utens = FusionTensor(umat, domain_axes(ft), (axes(umat)[1],))
  stens = FusionTensor(s, (axes(umat)[1],), (axes(vmat)[0],))
  vtens = FusionTensor(vmat, (axes(vmat)[0],), codomain_axes(ft))
  return utens, stens, vtens
end
