# This file defines interface to cast from and to dense array

using LinearAlgebra

# constructor from dense array with split axes
function FusionTensor(
  codomain_legs::NTuple{NCoAxes}, domain_legs::NTuple{NDoAxes}, dense::DA
) where {NCoAxes,NDoAxes,T<:Number,N,DA<:DenseArray{T,N}}
  legs = (codomain_legs..., domain_legs...)

  # impose passage through concatenated axes to enforce ndims(ft) == ndims(dense)
  # at compile time
  return FusionTensor{T,N,NCoAxes}(legs, dense)
end

# constructor from dense array with split axes with norm check
function FusionTensor(
  codomain_legs::NTuple{NCoAxes}, domain_legs::NTuple{NDoAxes}, dense::DA, tol_check::Real
) where {NCoAxes,NDoAxes,T<:Number,N,DA<:DenseArray{T,N}}
  legs = (codomain_legs..., domain_legs...)

  # impose passage through concatenated axes to enforce ndims(ft) == ndims(dense)
  # at compile time
  return FusionTensor{T,N,NCoAxes}(legs, dense, tol_check)
end

# constructor from dense array with concatenate axes
function FusionTensor{T,N,NCoAxes}(
  legs::Axes, dense::DA
) where {NCoAxes,N,Axes<:NTuple{N},T<:Number,DA<:DenseArray{T,N}}

  # input validation
  # ndims(dense) = length(legs) is enforced at compile time
  if dimension.(legs) != size(dense)
    throw(DomainError("legs dimensions are incompatible with dense array"))
  end

  codomain_legs = legs[begin:NCoAxes]
  domain_legs = legs[(NCoAxes + 1):end]

  # initialize data_matrix
  mat_row_axis = reduce(fuse, codomain_legs)
  mat_col_axis = reduce(fuse, domain_legs)
  data_matrix = BlockSparseArray{T}(mat_row_axis, mat_col_axis)

  # fill data_matrix
  # dummy: TODO

  return out = FusionTensor(codomain_legs, domain_legs, data_matrix)
end

# constructor from dense array with concatenate axes with norm check
function FusionTensor{T,N,NCoAxes}(
  legs::Axes, dense::DA, tol_check::Real
) where {NCoAxes,N,Axes<:NTuple{N},T<:Number,DA<:DenseArray{T,N}}
  out = FusionTensor{T,N,NCoAxes}(legs, dense)

  # check that norm is the same in input and output
  dense_norm = norm(dense)
  if abs(norm(out) - dense_norm) > tol_check * dense_norm
    throw(DomainError("Dense tensor norm is not preserved in FusionTensor cast"))
  end
  return out
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
