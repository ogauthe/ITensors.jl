# This file defines interface to cast from and to dense array

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, matrix_size
using NDTensors.GradedAxes: fuse

# constructor from dense array
function FusionTensor(
  axes_in::NTuple{N}, n_codomain_legs::Int, arr::DenseArray{T,N}, tol_check::Real=0.0
) where {T<:Number,N}

  # input validation
  # ndims(arr) = length(axes_in) is enforced at compile time
  if length.(axes_in) != size(arr)
    throw(DomainError("Axis is incompatible with dense array"))
  end

  # initialize matrix
  matrix_row_axis = reduce(fuse, axes_in[begin:n_codomain_legs])
  matrix_col_axis = reduce(fuse, axes_in[(n_codomain_legs + 1):end])
  matrix = BlockSparseArray{T}(matrix_row_axis, matrix_col_axis)

  # fill matrix
  # dummy: TODO

  out = FusionTensor(axes_in, n_codomain_legs, matrix)
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
