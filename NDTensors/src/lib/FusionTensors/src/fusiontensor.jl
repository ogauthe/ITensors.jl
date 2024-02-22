# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray

struct FusionTensor{
  T<:Number,N,G<:AbstractUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  # TBD more type stable with only N fixed or with NRL and NCL as type parameters?
  # can also define N_ROW_LEG as type parameter
  # with N fixed and n_row_legs dynamic, permutedims, dagger and co preserve type
  # but tensor contraction output type is not knwon at compile time
  axes::Axes
  n_row_legs::Int
  matrix::Arr
end

# getters
matrix(ft::FusionTensor) = ft.matrix
axes(ft::FusionTensor) = ft.axes
n_row_legs(ft::FusionTensor) = ft.n_row_legs

# misc
codomain_axes(ft::FusionTensor) = axes(ft)[n_row_legs(ft):end]
domain_axes(ft::FusionTensor) = axes(ft)[begin:n_row_legs(ft)]
n_column_legs(ft::FusionTensor) = ndims(ft) - n_row_legs(ft)
matrix_size(ft::FusionTensor) = size(matrix(ft))
row_axis(ft::FusionTensor) = axes(matrix(ft))[1]
column_axis(ft::FusionTensor) = axes(matrix(ft))[2]

# constructors
function FusionTensor(codomain_axes, domain_axes, matrix)
  # TBD cannot disable assert globally with julia 1.10
  # remove these? Add explicit input validation with if wrong throw() end?
  @assert length(codomain_axes) > 0
  @assert length(domain_axes) > 0
  @assert prod(length.(codomain_axes)) == size(matrix, 1)
  @assert prod(length.(domain_axes)) == size(matrix, 2)
  axes = (codomain_axes..., domain_axes...)
  n_row_legs = length(codomain_axes)
  return FusionTensor(axes, n_row_legs, matrix)
end

# sanity check
function sanity_check(ft::FusionTensor)
  nca = length(codomain_axes)
  if !(0 < nca < ndims(ft))
    throw(DomainError(nca, "invalid codomain axes length"))
  end
  nda = length(domain_axes)
  if !(0 < nda < ndims(ft))
    throw(DomainError(nda, "invalid domain axes length"))
  end
  if nca + nda != ndims(ft)
    throw(DomainError(nca + nda, "invalid ndims"))
  end
  if ndims(matrix(ft)) != 2
    throw(DomainError(ndims(matrix(ft)), "invalid matrix ndims"))
  end
  if size(matrix, 1) != prod(length.(codomain_axes(ft)))
    throw(DomainError(size(matrix, 1), "invalid matrix row number"))
  end
  if size(matrix, 2) != prod(length.(domain_axes(ft)))
    throw(DomainError(size(matrix, 2), "invalid matrix column number"))
  end
  return nothing
end

# swap row and column axes, transpose matrix blocks, dual any axis. No basis change.
function dagger(ft::FusionTensor)  # TBD change name? TBD move to permutedims?
  return FusionTensor(
    dual.(domain_axes(ft)), dual.(codomain_axes(ft)), transpose(matrix(ft))
  )
end
