# This file defines interface to cast from and to dense array

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, matrix_size
using NDTensors.GradedAxes: fuse

# constructor from dense array with split axes
function FusionTensor(
  codomain_legs::NTuple{NCoAxes}, domain_legs::NTuple{NDoAxes}, arr::DA, tol_check::R=0.0
) where {NCoAxes,NDoAxes,T<:Number,N,DA<:DenseArray{T,N},R<:Real}
  legs = (codomain_legs..., domain_legs...)

  # impose passage through concatenated axes to enforce ndims(ft) == ndims(arr)
  # at compile time
  return FusionTensor{T,N,NCoAxes}(legs, arr, tol_check)
end

# constructor from dense array with concatenate axes
function FusionTensor{T,N,NCoAxes}(
  legs::Axes, arr::DA, tol_check::R=0.0
) where {NCoAxes,N,Axes<:NTuple{N},T<:Number,DA<:DenseArray{T,N},R<:Real}
  codomain_legs = legs[begin:NCoAxes]
  domain_legs = legs[(NCoAxes + 1):end]

  # input validation
  # ndims(arr) = length(legs) is enforced at compile time
  if (length.(codomain_legs)..., length.(domain_legs)...) != size(arr)
    throw(DomainError("Axes are incompatible with dense array"))
  end

  # initialize data_matrix
  mat_row_axis = reduce(fuse, codomain_legs)
  mat_col_axis = reduce(fuse, domain_legs)
  data_matrix = BlockSparseArray{T}(mat_row_axis, mat_col_axis)

  # fill data_matrix
  # dummy: TODO

  out = FusionTensor(codomain_legs, domain_legs, data_matrix)

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
