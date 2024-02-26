# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: GradedUnitRange

struct FusionTensor{
  NCA,NDA,T<:Number,N,G<:GradedUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  _axes::Axes
  _matrix::Arr

  function FusionTensor{NCA}(
    legs::Axes, mat::Arr
  ) where {NCA,T<:Number,N,G<:GradedUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}}
    return new{NCA,N - NCA,T,N,G,Axes,Arr}(legs, mat)
  end
end

# alternative constructor from split codomain and domain
function FusionTensor(
  codomain_legs::NTuple{NCA}, domain_legs::NTuple{NDA}, arr::Arr
) where {NCA,NDA,T<:Number,Arr<:BlockSparseArray{T,2}}
  axes_in = (codomain_legs..., domain_legs...)
  return FusionTensor{NCA}(axes_in, arr)
end

# getters
matrix(ft::FusionTensor) = ft._matrix
Base.axes(ft::FusionTensor) = ft._axes

# misc
n_codomain_axes(::FusionTensor{NCA}) where {NCA} = NCA
n_domain_axes(::FusionTensor{NCA,NDA}) where {NCA,NDA} = NDA
codomain_axes(ft::FusionTensor) = axes(ft)[begin:n_codomain_axes(ft)]
domain_axes(ft::FusionTensor) = axes(ft)[(n_codomain_axes(ft) + 1):end]
matrix_size(ft::FusionTensor) = size(matrix(ft))
row_axis(ft::FusionTensor) = axes(matrix(ft))[1]
column_axis(ft::FusionTensor) = axes(matrix(ft))[2]

# sanity check
function sanity_check(ft::FusionTensor)
  nca = length(codomain_axes(ft))
  if !(0 < nca < ndims(ft))
    throw(DomainError(nca, "invalid codomain axes length"))
  end
  nda = length(domain_axes(ft))
  if !(0 < nda < ndims(ft))
    throw(DomainError(nda, "invalid domain axes length"))
  end
  if nca + nda != ndims(ft)
    throw(DomainError(nca + nda, "invalid ndims"))
  end
  m = matrix(ft)
  if ndims(m) != 2
    throw(DomainError(ndims(m), "invalid matrix ndims"))
  end
  if size(m, 1) != prod(length.(codomain_axes(ft)))
    throw(DomainError(size(m, 1), "invalid matrix row number"))
  end
  if size(m, 2) != prod(length.(domain_axes(ft)))
    throw(DomainError(size(m, 2), "invalid matrix column number"))
  end
  return nothing
end
