# This file defines struct FusionTensor and constructors

using NDTensors.BlockSparseArrays: BlockSparseArray

struct FusionTensor{
  T<:Number,N,G<:AbstractUnitRange,Axes<:NTuple{N,G},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  # TBD more type stable with only N fixed or with NRL and NCL as type parameters?
  # can also define N_ROW_LEG as type parameter
  # with N fixed and n_codomain_axes dynamic, permutedims, dagger and co preserve type
  # but tensor contraction output type is not knwon at compile time
  _axes::Axes
  _n_codomain_axes::Int
  _matrix::Arr
end

# getters
matrix(ft::FusionTensor) = ft._matrix
Base.axes(ft::FusionTensor) = ft._axes
n_codomain_axes(ft::FusionTensor) = ft._n_codomain_axes

# misc
codomain_axes(ft::FusionTensor) = axes(ft)[begin:n_codomain_axes(ft)]
domain_axes(ft::FusionTensor) = axes(ft)[(n_codomain_axes(ft) + 1):end]
n_domain_axes(ft::FusionTensor) = ndims(ft) - n_codomain_axes(ft)
matrix_size(ft::FusionTensor) = size(matrix(ft))
row_axis(ft::FusionTensor) = axes(matrix(ft))[1]
column_axis(ft::FusionTensor) = axes(matrix(ft))[2]

# alternative constructor from split codomain and domain
# works for both cast from dense or direct constructor from BlockSparseArray
function FusionTensor(codomain_legs::NTuple, domain_legs::NTuple, arr)
  # TBD cannot disable assert globally with julia 1.10
  # remove these? Add explicit input validation with if wrong throw() end?
  @assert length(codomain_legs) > 0
  @assert length(domain_legs) > 0
  axes_in = (codomain_legs..., domain_legs...)
  n_codomain_axes = length(codomain_legs)
  return FusionTensor(axes_in, n_codomain_axes, arr)
end

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
