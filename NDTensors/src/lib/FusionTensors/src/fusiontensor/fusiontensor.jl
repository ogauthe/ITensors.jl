# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: GradedUnitRange, fuse

struct FusionTensor{
  M,K,T<:Number,N,G<:GradedUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  _axes::Axes
  _data_matrix::Arr

  function FusionTensor{M}(
    legs::Axes, mat::Arr
  ) where {M,T<:Number,N,G<:GradedUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}}
    return new{M,N - M,T,N,G,Axes,Arr}(legs, mat)
  end
end

# alternative constructor from split codomain and domain
function FusionTensor(
  codomain_legs::NTuple{M}, domain_legs::NTuple{K}, mat::Arr
) where {M,K,T<:Number,Arr<:BlockSparseArray{T,2}}
  axes_in = (codomain_legs..., domain_legs...)
  return FusionTensor{M}(axes_in, mat)
end

# empty matrix, split codomain and domain
function FusionTensor{T}
  (codomain_legs::NTuple{M}, domain_legs::NTuple{K}) where {T<:Number,M,K}
  row_axis = reduce(fuse, codomain_legs)
  col_axis = reduce(fuse, domain_legs)
  mat = BlockSparseArray{T}(row_axis, col_axis)
  return FusionTensor(codomain_legs, domain_legs, mat)
end

# empty matrix, concatenate axes
function FusionTensor{M,T}
  (legs::Axes) where {T<:Number,M,Axes}
  codomain_legs = legs[begin:M]
  domain_legs = legs[(M + 1):end]
  return FusionTensor(codomain_legs, domain_legs)
end

# getters
data_matrix(ft::FusionTensor) = ft._data_matrix
Base.axes(ft::FusionTensor) = ft._axes

# misc
n_codomain_axes(::FusionTensor{M}) where {M} = M
n_domain_axes(::FusionTensor{M,K}) where {M,K} = K
codomain_axes(ft::FusionTensor) = axes(ft)[begin:n_codomain_axes(ft)]
domain_axes(ft::FusionTensor) = axes(ft)[(n_codomain_axes(ft) + 1):end]
matrix_size(ft::FusionTensor) = size(data_matrix(ft))
matrix_row_axis(ft::FusionTensor) = axes(data_matrix(ft))[1]
matrix_column_axis(ft::FusionTensor) = axes(data_matrix(ft))[2]

# sanity check
function sanity_check(ft::FusionTensor)
  # TODO replace @assert with @check when JuliaLang PR 41342 is merged
  nca = n_codomain_axes(ft)
  @assert nca == length(codomain_axes(ft)) "n_codomain_axes does not match codomain_axes"
  @assert 0 < nca < ndims(ft) "invalid n_codomain_axes"

  nda = n_domain_axes(ft)
  @assert nda == length(domain_axes(ft)) "n_domain_axes does not match domain_axes"
  @assert 0 < nda < ndims(ft) "invalid n_domain_axes"

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  @assert size(m, 1) == prod(length.(codomain_axes(ft))) "invalid data_matrix row number"
  @assert size(m, 2) == prod(length.(domain_axes(ft))) "invalid data_matrix column number"

  return nothing
end
