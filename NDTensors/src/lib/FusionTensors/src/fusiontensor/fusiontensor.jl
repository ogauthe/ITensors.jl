# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray
using NDTensors.GradedAxes: GradedUnitRange, UnitRangeDual, fusion_product

# TBD remve NCoAxes and NDoAxes as explicit parameters?
struct FusionTensor{T,N,NCoAxes,NDoAxes,CoDomainAxes,DomainAxes,Mat} <: AbstractArray{T,N}
  _codomain_axes::CoDomainAxes
  _domain_axes::DomainAxes
  _data_matrix::Mat

  # inner constructor to impose constraints on types
  function FusionTensor(
    codomain_legs::CoDomainAxes, domain_legs::DomainAxes, mat::Mat
  ) where {
    T<:Number,
    NCoAxes,
    NDoAxes,
    CoDomainAxes<:Tuple{Vararg{Union{GradedUnitRange,UnitRangeDual},NCoAxes}},
    DomainAxes<:Tuple{Vararg{Union{GradedUnitRange,UnitRangeDual},NDoAxes}},
    Mat<:BlockSparseArray{T,2},
  }
    return new{T,NCoAxes + NDoAxes,NCoAxes,NDoAxes,CoDomainAxes,DomainAxes,Mat}(
      codomain_legs, domain_legs, mat
    )
  end
end

# alternative constructor from concatenated axes
function FusionTensor{T,N,NCoAxes}(
  legs::L, arr::AbstractArray
) where {T,N,NCoAxes,L<:Tuple{Vararg{Union{GradedUnitRange,UnitRangeDual},N}}}
  codomain_legs = legs[begin:NCoAxes]
  domain_legs = legs[(NCoAxes + 1):end]
  return FusionTensor(codomain_legs, domain_legs, arr)
end

# empty matrix, split axes
function FusionTensor{T}(codomain_legs::Tuple, domain_legs::Tuple) where {T}
  row_axis = reduce(fusion_product, codomain_legs)
  col_axis = reduce(fusion_product, domain_legs)
  mat = BlockSparseArray{T}(row_axis, col_axis)
  return FusionTensor(codomain_legs, domain_legs, mat)
end

# empty matrix with concatenated axes
function FusionTensor{T,N,NCoAxes}(
  legs::L
) where {T,N,NCoAxes,L<:Tuple{Vararg{Union{GradedUnitRange,UnitRangeDual},N}}}
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
  @assert nca + nda == ndims(ft) "invalid ndims"

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  @assert size(m, 1) == prod(length.(codomain_axes(ft))) "invalid data_matrix row number"
  @assert size(m, 2) == prod(length.(domain_axes(ft))) "invalid data_matrix column number"

  @assert reduce(fusion_product, codomain_axes(ft)) == axes(m)[1] "data_matrix row axis does not match codomain axes"
  @assert reduce(fusion_product, domain_axes(ft)) == axes(m)[2] "data_matrix column axis does not match domain axes"
  return nothing
end
