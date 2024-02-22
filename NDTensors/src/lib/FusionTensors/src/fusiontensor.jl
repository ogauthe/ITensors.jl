# This file defines struct FusionTensor and constructors

using NDTensors.Sectors: AbstractCategory
using NDTensors.GradedAxes
using BlockArrays
using NDTensors.BlockSparseArrays: BlockSparseArray
using ITensors: @debug_check

struct FusionTensor{
  T<:Number,N,Axes<:Tuple{N,AbstractUnitRange{Int}},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  axes::Axes
  n_row_legs::Int  # TBD more type stable with only N fixed or with NRL and NCL?
  matrix::Arr
end

"""
function check_consistency(ft::FusionTensor)
  if length(t.codomain_axes) != t.n_codomain_legs
    return false
  end
  if length(t.domain_axes) != t.n_domain_legs
    return false
  end
  if t.n_codomain_legs < 1
    return false
  end
  if t.n_domain_legs < 1
    return false
  end
  if t.ndims != t.n_codomain_legs + t.n_domain_legs
    return false
  end
  if length(t.matrix) != t.nblocks
    return false
  end
  if blocklength(matrix_row_axis) != t.nblocks
    return false
  end
  if blocklength(matrix_column_axis) != t.nblocks
    return false
  end
  for i in 1:(t.nblocks)
    if size(matrix[i], 1) != blocklengths(t.matrix_row_axis)[i]
      return false
    end
    if size(matrix[i], 2) != blocklengths(t.matrix_column_axis)[i]
      return false
    end
  end

  fused_codomain = fuse(t.codomain_axes)  # this is slow
  fused_domain = fuse(t.domain_axes)  # this is slow
  # TODO define "included in"
  if !(t.matrix_row_axis < fused_codomain)
    return false
  end
  if !(t.matrix_column_axis < fused_domain)
    return false
  end
  return true
end
"""

# getters
matrix(ft::FusionTensor) = ft.matrix
axes(ft::FusionTensor) = ft.axes
n_row_legs(ft::FusionTensor) = ft.n_row_legs

# misc
domain_axes(ft::FusionTensor) = axes(ft)[begin:n_row_legs(ft)]
codomain_axes(ft::FusionTensor) = axes(ft)[n_row_legs(t):end]
n_column_legs(ft::FusionTensor) = ndims(ft) - n_row_legs(ft)
matrix_size(ft::FusionTensor) = size(matrix(ft))
row_axis(ft::FusionTensor) = axes(matrix(ft))[1]
column_axis(ft::FusionTensor) = axes(matrix(ft))[2]

# constructors
function FusionTensor(codomain_axes, domain_axes, matrix)
  axes = (codomain_axes..., domain_axes...)
  n_row_legs = length(axes)
  return FusionTensor(axes, n_row_legs, matrix)
end

# swap row and column axes, transpose matrix blocks, dual any axis. No basis change.
function dagger(ft::FusionTensor)  # TBD change name? TBD move to permutedims?
  return FusionTensor(
    dual.(domain_axes(ft)), dual.(codomain_axes(ft)), transpose(matrix(ft))
  )
end
