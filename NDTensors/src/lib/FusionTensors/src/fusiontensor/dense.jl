# This file defines interface to cast from and to dense array

# constructor from dense array
function FusionTensor(codomain_legs::Tuple, domain_legs::Tuple, dense::DenseArray)

  # compile time check
  if length(codomain_legs) + length(domain_legs) != ndims(dense)
    throw(DomainError("legs dimensions are incompatible with dense array"))
  end

  # input validation
  if Sectors.quantum_dimension.((codomain_legs..., domain_legs...)) != size(dense)
    throw(DomainError("legs dimensions are incompatible with dense array"))
  end

  # initialize data_matrix
  data_matrix = initialize_data_matrix(codomain_legs, domain_legs, eltype(dense))

  # fill data_matrix
  # dummy: TODO

  return FusionTensor(codomain_legs, domain_legs, data_matrix)
end

# constructor from dense array with norm check
function FusionTensor(
  codomain_legs::Tuple, domain_legs::Tuple, dense::DenseArray, tol_check::Real
)
  ft = FusionTensor(codomain_legs, domain_legs, dense)

  # check that norm is the same in input and output
  dense_norm = LinearAlgebra.norm(dense)
  if abs(LinearAlgebra.norm(ft) - dense_norm) > tol_check * dense_norm
    throw(DomainError("Dense tensor norm is not preserved in FusionTensor cast"))
  end
  return ft
end

# cast to julia dense array with tensor size
function Base.Array(ft::FusionTensor)

  # initialize dense matrix
  dense_mat = zeros(matrix_size(ft))

  # fill dense
  # dummy: TODO

  dense = reshape(dense_mat, size(ft))
  return dense
end
