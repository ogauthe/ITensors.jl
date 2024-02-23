# This file defines interface to cast from and to dense array

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.FusionTensors: FusionTensor, matrix_size
using NDTensors.GradedAxes: fuse

# constructor from dense array
function FusionTensor{T,N,G}(
  axes::Axes <: NTuple{N,G}, n_codomain_axes::Int, arr::A, tol_check::Real=0.0
) where {A<:DenseArray{T,N}}

  # input validation
  # ndims(arr) = length(axes) is enforced at compile time
  if length.(axes) != size(arr)
    throw(DomainError("Axis is incompatible with dense array"))
  end

  # initialize matrix
  row_axis = fuse(codomain_axes)
  col_axis = fuse(domain_axes)
  matrix = BlockSparseArray{T,2}(row_axis, col_axis)

  # fill matrix
  # dummy: TODO

  out = FusionTensor(axes, n_codomain_axes, matrix)

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
  mat = zeros(matrix_size(ft))

  # dummy: TODO

  arr = reshape(mat, size(ft))
  return arr
end
