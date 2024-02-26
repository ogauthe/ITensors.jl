# This file defines interface to cast from and to dense array

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, matrix_size
using NDTensors.GradedAxes: fuse

# constructor from dense array with concatenate axes
function FusionTensor{M}(
  legs::Axes, arr::DA, tol_check::R=0.0
) where {M,N,Axes<:NTuple{N},T<:Number,DA<:DenseArray{T,N},R<:Real}
  codomain_legs = legs[begin:M]
  domain_legs = legs[(M + 1):end]
  return FusionTensor(codomain_legs, domain_legs, arr, tol_check)
end

# constructor from dense array with split axes
function FusionTensor(
  codomain_legs, domain_legs, arr::DA, tol_check::Real=0.0
) where {T<:Number,DA<:DenseArray{T}}

  # input validation
  # ndims(arr) = length(axes_in) is enforced at compile time
  if (length.(codomain_legs)..., length.(domain_legs)...) != size(arr)
    throw(DomainError("Axes are incompatible with dense array"))
  end

  # initialize matrix
  matrix_row_axis = reduce(fuse, codomain_legs)
  matrix_col_axis = reduce(fuse, domain_legs)
  matrix = BlockSparseArray{T}(matrix_row_axis, matrix_col_axis)

  # fill matrix
  # dummy: TODO

  out = FusionTensor(codomain_legs, domain_legs, matrix)

  # check that norm is the same in input and output
  if tol_check > 0
    dense_norm = norm(dense)
    if abs(norm(out) - dense_norm(dense)) > tol_check * dense_norm
      throw(DomainError("Dense tensor norm is not conserved"))
    end
  end
  return out
end

# cast to julia dense array with tensor size
function Base.Array(ft::FusionTensor)

  # initialize dense
  dense = zeros(matrix_size(ft))

  # fill dense
  # dummy: TODO

  arr = reshape(dense, size(ft))
  return arr
end
