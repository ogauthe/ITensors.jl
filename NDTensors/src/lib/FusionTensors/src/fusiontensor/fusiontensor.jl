# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: GradedUnitRange, fuse

struct FusionTensor{
  T<:Number,
  N,
  NCoAxes,
  NDoAxes,
  G<:GradedUnitRange,
  CoDomainAxes<:NTuple{NCoAxes,G},
  DomainAxes<:NTuple{NDoAxes,G},
  Mat<:BlockSparseArray{T,2},
} <: AbstractArray{T,N}
  _codomain_axes::CoDomainAxes
  _domain_axes::DomainAxes
  _data_matrix::Mat

  # inner constructor to impose NCoAxes + NDoAxes = N
  function FusionTensor(
    codomain_legs::CoDomainAxes, domain_legs::DomainAxes, mat::Mat
  ) where {
    NCoAxes,
    NDoAxes,
    G,
    T<:Number,
    CoDomainAxes<:NTuple{NCoAxes,G},
    DomainAxes<:NTuple{NDoAxes,G},
    Mat<:BlockSparseArray{T,2},
  }
    return new{T,NCoAxes + NDoAxes,NCoAxes,NDoAxes,G,CoDomainAxes,DomainAxes,Mat}(
      codomain_legs, domain_legs, mat
    )
  end
end

# alternative constructor from concatenated axes
function FusionTensor{T,N,NCoAxes}(legs::NTuple{N}, arr) where {T,N,NCoAxes}
  codomain_legs = legs[begin:NCoAxes]  # or ntuplie(i->legs[i], NCoAxes)?
  domain_legs = legs[(NCoAxes + 1):end]
  return FusionTensor(codomain_legs, domain_legs, arr)
end

# empty matrix, split axes
#function FusionTensor{T}
# (codomain_legs::NTuple, domain_legs::NTuple) where {T}
# row_axis = reduce(fuse, codomain_legs)
#  col_axis = reduce(fuse, domain_legs)
#  mat = BlockSparseArray{T}(row_axis, col_axis)
#  return FusionTensor(codomain_legs, domain_legs, mat)
#end

# empty matrix with concatenated axes
function FusionTensor{T,N,NCoAxes}(legs::NTuple{N}) where {T,N,NCoAxes}
  codomain_legs = legs[begin:NCoAxes]
  domain_legs = legs[(NCoAxes + 1):end]
  return FusionTensor(codomain_legs, domain_legs)
end

# getters
data_matrix(ft::FusionTensor) = ft._data_matrix
codomain_axes(ft::FusionTensor) = ft._codomain_axes
domain_axes(ft::FusionTensor) = ft._domain_axes

# misc
n_codomain_axes(::FusionTensor{T,N,NCoAxes}) where {T,N,NCoAxes} = NCoAxes
n_domain_axes(::FusionTensor{T,N,NCoAxes,NDoAxes}) where {T,N,NCoAxes,NDoAxes} = NDoAxes

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
