# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: block_stored_indices

struct FusionTensor{T,N,DomainAxes,CoDomainAxes,Mat} <: AbstractArray{T,N}
  data_matrix::Mat
  domain_axes::DomainAxes
  codomain_axes::CoDomainAxes

  # inner constructor to impose constraints on types
  # TBD replace domain_legs with FusedAxes(domain_legs)?
  function FusionTensor(
    mat::Union{BlockSparseMatrix,Adjoint{<:Number,<:BlockSparseMatrix}},
    domain_legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}},
    codomain_legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}},
  ) where {LA}
    return new{
      eltype(mat),
      length(domain_legs) + length(codomain_legs),
      typeof(domain_legs),
      typeof(codomain_legs),
      typeof(mat),
    }(
      mat, domain_legs, codomain_legs
    )
  end
end

# getters
data_matrix(ft::FusionTensor) = ft.data_matrix
domain_axes(ft::FusionTensor) = ft.domain_axes
codomain_axes(ft::FusionTensor) = ft.codomain_axes

# misc access
ndims_domain(ft::FusionTensor) = length(domain_axes(ft))
ndims_codomain(ft::FusionTensor) = length(codomain_axes(ft))

matrix_size(ft::FusionTensor) = quantum_dimension.(axes(data_matrix(ft)))
matrix_row_axis(ft::FusionTensor) = first(axes(data_matrix(ft)))
matrix_column_axis(ft::FusionTensor) = last(axes(data_matrix(ft)))

function unify_sector_type(
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange{LA}}},
) where {LA}  # nothing to do
  return domain_legs, codomain_legs
end

# TODO move this to SymmetrySectors or GradedAxes
function find_common_sector_type(sector_or_axes_enum)
  # fuse trivial sectors to produce unified type
  # avoid depending on SymmetrySectors internals
  return label_type(fusion_product(trivial.(sector_or_axes_enum)...))
end

function unify_sector_type(
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  T = find_common_sector_type((domain_legs..., codomain_legs...))
  unified_domain_legs = map(g -> unify_sector_type(T, g), domain_legs)
  unified_codomain_legs = map(g -> unify_sector_type(T, g), codomain_legs)
  return unified_domain_legs, unified_codomain_legs
end

function unify_sector_type(T::Type{<:SectorProduct}, g::AbstractGradedUnitRange)
  # fuse with trivial to insert all missing arguments inside each GradedAxis
  # avoid depending on SymmetrySectors internals
  glabels = map(s -> only(blocklabels(fusion_product(trivial(T), s))), blocklabels(g))
  # use labelled_blocks to preserve GradedUnitRange
  unified_g = labelled_blocks(unlabel_blocks(g), glabels)
  return isdual(g) ? flip(unified_g) : unified_g
end

# empty matrix
function FusionTensor(
  data_type::Type,
  domain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs_raw::Tuple{Vararg{AbstractGradedUnitRange}},
)
  domain_legs, codomain_legs = unify_sector_type(domain_legs_raw, codomain_legs_raw)
  domain_fused_axes = FusedAxes(domain_legs)
  codomain_fused_axes = FusedAxes(dual.(codomain_legs))
  mat = initialize_data_matrix(data_type, domain_fused_axes, codomain_fused_axes)
  return FusionTensor(mat, domain_legs, codomain_legs)
end

# init data_matrix
function initialize_data_matrix(
  data_type::Type{<:Number}, domain_fused_axes::FusedAxes, codomain_fused_axes::FusedAxes
)
  # fusion trees have Float64 eltype: need compatible type
  promoted = promote_type(data_type, Float64)
  mat_row_axis = fused_axis(domain_fused_axes)
  mat_col_axis = dual(fused_axis(codomain_fused_axes))
  return BlockSparseArray{promoted}(mat_row_axis, mat_col_axis)
end

function check_data_matrix_axes(
  mat::BlockSparseMatrix, domain_legs::Tuple, codomain_legs::Tuple
)
  ft0 = FusionTensor(Float64, domain_legs, codomain_legs)
  @assert space_isequal(matrix_row_axis(ft0), axes(mat, 1))
  @assert space_isequal(matrix_column_axis(ft0), axes(mat, 2))
end

function check_data_matrix_axes(mat::Adjoint, domain_legs::Tuple, codomain_legs::Tuple)
  return check_data_matrix_axes(adjoint(mat), dual.(codomain_legs), dual.(domain_legs))
end

function check_sanity(ft::FusionTensor)
  nca = ndims_codomain(ft)
  @assert nca == length(codomain_axes(ft)) "ndims_codomain does not match codomain_axes"
  @assert nca <= ndims(ft) "invalid ndims_codomain"

  nda = ndims_domain(ft)
  @assert nda == length(domain_axes(ft)) "ndims_domain does not match domain_axes"
  @assert nda <= ndims(ft) "invalid ndims_domain"
  @assert nda + nca == ndims(ft) "invalid ndims"

  @assert length(axes(ft)) == ndims(ft) "ndims does not match axes"
  @assert matching_axes(axes(ft)[begin:nda], domain_axes(ft)) "axes do not match domain_axes"
  @assert matching_axes(axes(ft)[(nda + 1):end], codomain_axes(ft)) "axes do not match codomain_axes"

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  row_axis = matrix_row_axis(ft)
  column_axis = matrix_column_axis(ft)
  @assert row_axis === axes(m, 1) "invalid row_axis"
  @assert column_axis === axes(m, 2) "invalid column_axis"
  check_data_matrix_axes(data_matrix(ft), domain_axes(ft), codomain_axes(ft))

  for it in eachindex(block_stored_indices(m))
    @assert dual(blocklabels(row_axis)[it[1]]) == blocklabels(column_axis)[it[2]] "forbidden block"
  end
  return nothing
end

matching_dual(axes1::Tuple, axes2::Tuple) = matching_axes(dual.(axes1), axes2)
matching_axes(axes1::Tuple, axes2::Tuple) = false
function matching_axes(axes1::T, axes2::T) where {T<:Tuple}
  return all(space_isequal.(axes1, axes2))
end
