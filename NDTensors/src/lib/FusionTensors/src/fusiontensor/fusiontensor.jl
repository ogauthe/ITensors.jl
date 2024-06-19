# This file defines struct FusionTensor and constructors

struct FusionTensor{T,N,CoDomainAxes,DomainAxes,Mat} <: AbstractArray{T,N}
  data_matrix::Mat
  codomain_axes::CoDomainAxes
  domain_axes::DomainAxes

  # inner constructor to impose constraints on types
  function FusionTensor(
    mat::BlockSparseArrays.BlockSparseMatrix,
    codomain_legs::Tuple{Vararg{AbstractUnitRange}},
    domain_legs::Tuple{Vararg{AbstractUnitRange}},
  )
    return new{
      eltype(mat),
      length(codomain_legs) + length(domain_legs),
      typeof(codomain_legs),
      typeof(domain_legs),
      typeof(mat),
    }(
      mat, codomain_legs, domain_legs
    )
  end

  function FusionTensor(
    mat::LinearAlgebra.Adjoint{<:Number,<:BlockSparseArrays.BlockSparseMatrix},
    codomain_legs::Tuple{Vararg{AbstractUnitRange}},
    domain_legs::Tuple{Vararg{AbstractUnitRange}},
  )
    return new{
      eltype(mat),
      length(codomain_legs) + length(domain_legs),
      typeof(codomain_legs),
      typeof(domain_legs),
      typeof(mat),
    }(
      mat, codomain_legs, domain_legs
    )
  end
end

# getters
data_matrix(ft::FusionTensor) = ft.data_matrix
codomain_axes(ft::FusionTensor) = ft.codomain_axes
domain_axes(ft::FusionTensor) = ft.domain_axes

# misc access
ndims_codomain(ft::FusionTensor) = length(codomain_axes(ft))
ndims_domain(ft::FusionTensor) = length(domain_axes(ft))

matrix_size(ft::FusionTensor) = Sectors.quantum_dimension.(axes(data_matrix(ft)))
matrix_row_axis(ft::FusionTensor) = first(axes(data_matrix(ft)))
matrix_column_axis(ft::FusionTensor) = last(axes(data_matrix(ft)))

# initialization
function initialize_matrix_axes(codomain_legs::Tuple, domain_legs::Tuple)
  mat_row_axis = GradedAxes.dual(
    GradedAxes.fusion_product(GradedAxes.dual.(codomain_legs)...)
  )
  mat_col_axis = GradedAxes.fusion_product(domain_legs...)
  return mat_row_axis, mat_col_axis
end

function initialize_matrix_axes(::Tuple{}, domain_legs::Tuple)
  mat_row_axis = GradedAxes.dual(Sectors.trivial(first(domain_legs)))
  mat_col_axis = GradedAxes.fusion_product(domain_legs...)
  return mat_row_axis, mat_col_axis
end

function initialize_matrix_axes(codomain_legs::Tuple, ::Tuple{})
  mat_row_axis = GradedAxes.dual(GradedAxes.fusion_product(codomain_legs...))
  mat_col_axis = Sectors.trivial(first(codomain_legs))
  return mat_row_axis, mat_col_axis
end

function initialize_matrix_axes(::Tuple{}, ::Tuple{})
  mat_col_axis = GradedAxes.gradedrange([Sectors.sector() => 1])
  mat_row_axis = GradedAxes.dual(mat_col_axis)
  return mat_row_axis, mat_col_axis
end

# empty matrix
function FusionTensor(data_type::Type, codomain_legs::Tuple, domain_legs::Tuple)
  mat = initialize_data_matrix(data_type, codomain_legs, domain_legs)
  return FusionTensor(mat, codomain_legs, domain_legs)
end

# init data_matrix
function initialize_data_matrix(
  data_type::Type{<:Number}, codomain_legs::Tuple, domain_legs::Tuple
)
  # fusion trees have Float64 eltype: need compatible type
  promoted = promote_type(data_type, Float64)
  mat_row_axis, mat_col_axis = initialize_matrix_axes(codomain_legs, domain_legs)
  return BlockSparseArrays.BlockSparseArray{promoted}(mat_row_axis, mat_col_axis)
end

function check_data_matrix_axes(
  mat::BlockSparseArrays.BlockSparseMatrix, codomain_legs::Tuple, domain_legs::Tuple
)
  rg, cg = initialize_matrix_axes(codomain_legs, domain_legs)
  @assert GradedAxes.gradedisequal(rg, axes(mat, 1))
  @assert GradedAxes.gradedisequal(cg, axes(mat, 2))
end

function check_data_matrix_axes(
  mat::LinearAlgebra.Adjoint, codomain_legs::Tuple, domain_legs::Tuple
)
  return check_data_matrix_axes(
    adjoint(mat), GradedAxes.dual.(domain_legs), GradedAxes.dual.(codomain_legs)
  )
end

function check_sanity(ft::FusionTensor)
  nca = ndims_codomain(ft)
  @assert nca == length(codomain_axes(ft)) "ndims_codomain does not match codomain_axes"
  @assert nca <= ndims(ft) "invalid ndims_codomain"

  nda = ndims_domain(ft)
  @assert nda == length(domain_axes(ft)) "ndims_domain does not match domain_axes"
  @assert nda <= ndims(ft) "invalid ndims_domain"
  @assert nca + nda == ndims(ft) "invalid ndims"

  @assert length(axes(ft)) == ndims(ft) "ndims does not match axes"
  @assert matching_axes(axes(ft)[begin:nca], codomain_axes(ft)) "axes do not match codomain_axes"
  @assert matching_axes(axes(ft)[(nca + 1):end], domain_axes(ft)) "axes do not match domain_axes"

  m = data_matrix(ft)
  @assert ndims(m) == 2 "invalid data_matrix ndims"
  @assert size(m, 1) == prod(length.(codomain_axes(ft))) "invalid data_matrix row number"
  @assert size(m, 2) == prod(length.(domain_axes(ft))) "invalid data_matrix column number"

  row_axis = matrix_row_axis(ft)
  column_axis = matrix_column_axis(ft)
  @assert row_axis === axes(m, 1) "invalid row_axis"
  @assert column_axis === axes(m, 2) "invalid column_axis"
  check_data_matrix_axes(data_matrix(ft), codomain_axes(ft), domain_axes(ft))

  for it in eachindex(BlockSparseArrays.block_stored_indices(m))
    @assert GradedAxes.dual(GradedAxes.blocklabels(row_axis)[it[1]]) ==
      GradedAxes.blocklabels(column_axis)[it[2]] "forbidden block"
  end
  return nothing
end

matching_dual(axes1::Tuple, axes2::Tuple) = matching_axes(GradedAxes.dual.(axes1), axes2)
matching_axes(axes1::Tuple, axes2::Tuple) = false
function matching_axes(axes1::T, axes2::T) where {T<:Tuple}
  return all(GradedAxes.gradedisequal.(axes1, axes2))
end
