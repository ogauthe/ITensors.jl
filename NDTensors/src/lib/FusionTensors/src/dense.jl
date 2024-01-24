# This file defines interface to cast from and to dense array

using ITensors: @debug_check
using NDTensors.GradedAxes: fuse
using NDTensors.FusionTensor: FusionTensor, matrix_size

# constructor from dense array
function FusionTensor(
  codomain_axes, domain_axes, dense::Array{T}, tol_check::Float=0.0
) where {T<:number}
  axes = [codomain_axes, domain_axes]
  if length(axes) != ndims(dense)
    throw(DomainError("incompatible number of axes"))
  end
  ndim = ndims(dense)
  for i in 1:ndim
    if length(axes[i]) != size(dense, i)
      throw(DomainError("Axis is incompatible with dense array"))
    end
  end

  matrix_row_axis = fuse(codomain_axes)
  matrix_col_axis = fuse(domain_axes)
  shared = match_blocks(matrix_row_axis, matrix_col_axis)
  matrix_row_axis = matrix_row_axis[shared]
  matrix_col_axis = matrix_col_axis[shared]
  blocks = []
  for i in shared
    size = (matrix_row_axis[i], matrix_col_axis[i])
    push!(blocks, zeros{T}(size))
  end

  n_ele = .*(blocklength.(axes))
  for i_ele in 1:n_ele
    for b in blocks
      b[slices] = m(i_ele)
    end
  end

  out = NonAbelianTensor(
    codomain_axes, domain_axes, matrix_row_axis, matrix_column_axis, matrix_blocks
  )

  # check that norm is the same in input and output
  if tol_check > 0
    dense_norm = norm(dense)
    if abs(norm(out) - dense_norm(dense)) > tol_check * dense_norm
      throw(DomainError("Dense tensor norm is not conserved"))
    end
  end
  return out
end

# cast to julia dense matrix with matrix size
function Base.Matrix(t::FusionTensor)  # TBD overload Base.Matrix or tomatrix?
  mat = zeros(matrix_size(t))
  @debug_check abs(norm(mat) - norm(t)) < 1e-13 * norm(t)
  return mat
end

# cast to julia dense array with tensor size
function Base.Array(t::FusionTensor)  # TBD overload Base.Matrix or toarray?
  mat = tomatrix(t)
  dense = reshape(mat, size(t))
  return dense
end
