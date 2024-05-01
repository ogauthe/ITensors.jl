# This file defines struct FusionTensor and constructors

# TBD remve NCoAxes and NDoAxes as explicit parameters?
struct FusionTensor{T,N,CoDomainAxes,DomainAxes,Mat} <: AbstractArray{T,N}
  codomain_axes::CoDomainAxes
  domain_axes::DomainAxes
  data_matrix::Mat

  # inner constructor to impose constraints on types
  function FusionTensor(
    codomain_legs::Tuple{Vararg{AbstractUnitRange}},
    domain_legs::Tuple{Vararg{AbstractUnitRange}},
    mat::BlockSparseArrays.BlockSparseArray{<:Any,2},
  )
    return new{
      eltype(mat),
      length(codomain_legs) + length(domain_legs),
      typeof(codomain_legs),
      typeof(domain_legs),
      typeof(mat),
    }(
      codomain_legs, domain_legs, mat
    )
  end
end

# empty matrix
function FusionTensor{T}(codomain_legs::Tuple, domain_legs::Tuple) where {T}
  row_axis = reduce(GradedAxes.fusion_product, codomain_legs)
  col_axis = reduce(GradedAxes.fusion_product, domain_legs)
  mat = BlockSparseArrays.BlockSparseArray{T}(row_axis, col_axis)
  return FusionTensor(codomain_legs, domain_legs, mat)
end

# getters
data_matrix(ft::FusionTensor) = ft.data_matrix
codomain_axes(ft::FusionTensor) = ft.codomain_axes
domain_axes(ft::FusionTensor) = ft.domain_axes

# misc
ndims_codomain(ft::FusionTensor) = length(codomain_axes(ft))
ndims_domain(ft::FusionTensor) = length(domain_axes(ft))

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

  @assert GradedAxes.gradedisequal(
    reduce(GradedAxes.fusion_product, codomain_axes(ft)), axes(m)[1]
  ) "data_matrix row axis does not match codomain axes"
  @assert GradedAxes.gradedisequal(
    reduce(GradedAxes.fusion_product, domain_axes(ft)), axes(m)[2]
  ) "data_matrix column axis does not match domain axes"
  return nothing
end

function matching_axes(axes1::T, axes2::T) where {T}
  if length(axes1) != length(axes2)  # in cases axes1 is a Vector
    return false
  end
  return all(GradedAxes.gradedisequal.(axes1, axes2))
end
function matching_dual(axes1, axes2)
  return matching_axes(axes1, GradedAxes.dual.(axes2))
end
