# This files defines Base functions for FusionTensor

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(x * data_matrix(ft), domain_axes(ft), codomain_axes(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(x * data_matrix(ft), domain_axes(ft), codomain_axes(ft))
end

# tensor contraction is a block data_matrix product.
# allow to contract with different eltype and let BlockSparseArray ensure compatibility
# impose matching type and number of axes at compile time
# impose matching axes at run time
function Base.:*(left::FusionTensor, right::FusionTensor)
  if !matching_dual(codomain_axes(left), domain_axes(right))
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) * data_matrix(right)
  return FusionTensor(new_data_matrix, domain_axes(left), codomain_axes(right))
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block data_matrix add.
# impose matching axes, allow different eltypes
function Base.:+(left::FusionTensor, right::FusionTensor)
  if !matching_axes(axes(left), axes(right))
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) + data_matrix(right)
  return FusionTensor(new_data_matrix, domain_axes(left), codomain_axes(left))
end

function Base.:-(ft::FusionTensor)
  new_data_matrix = -data_matrix(ft)
  return FusionTensor(new_data_matrix, domain_axes(ft), codomain_axes(ft))
end

function Base.:-(left::FusionTensor, right::FusionTensor)
  if !matching_axes(axes(left), axes(right))
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) - data_matrix(right)
  return FusionTensor(new_data_matrix, domain_axes(left), codomain_axes(left))
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(data_matrix(ft) / x, domain_axes(ft), codomain_axes(ft))
end

Base.Array(ft::FusionTensor) = Array(cast_to_dense(ft))

# adjoint is costless: dual axes, swap domain and codomain, take data_matrix adjoint.
# data_matrix coeff are not modified (beyond complex conjugation)
function Base.adjoint(ft::FusionTensor)
  return FusionTensor(
    adjoint(data_matrix(ft)),
    GradedAxes.dual.(codomain_axes(ft)),
    GradedAxes.dual.(domain_axes(ft)),
  )
end

Base.axes(ft::FusionTensor) = (domain_axes(ft)..., codomain_axes(ft)...)

# conj is defined as coefficient wise complex conjugation, without axis dual
# same object for real element type
Base.conj(ft::FusionTensor{<:Real}) = ft

function Base.conj(ft::FusionTensor)
  return FusionTensor(conj(data_matrix(ft)), domain_axes(ft), codomain_axes(ft))
end

function Base.copy(ft::FusionTensor)
  new_data_matrix = copy(data_matrix(ft))
  new_domain_axes = copy.(domain_axes(ft))
  new_codomain_axes = copy.(codomain_axes(ft))
  return FusionTensor(new_data_matrix, new_domain_axes, new_codomain_axes)
end

function Base.deepcopy(ft::FusionTensor)
  new_data_matrix = deepcopy(data_matrix(ft))
  new_domain_axes = deepcopy.(domain_axes(ft))
  new_codomain_axes = deepcopy.(codomain_axes(ft))
  return FusionTensor(new_data_matrix, new_domain_axes, new_codomain_axes)
end

# eachindex is automatically defined for AbstractArray. We do now want it.
function Base.eachindex(::FusionTensor)
  throw(DomainError("eachindex", "eachindex not defined for FusionTensor"))
end

Base.ndims(::FusionTensor{T,N}) where {T,N} = N

Base.permutedims(ft::FusionTensor, args...) = fusiontensor_permutedims(ft, args...)

function Base.similar(ft::FusionTensor)
  mat = similar(data_matrix(ft))
  return FusionTensor(mat, domain_axes(ft), codomain_axes(ft))
end

function Base.similar(ft::FusionTensor, elt::Type)
  return FusionTensor(elt, domain_axes(ft), codomain_axes(ft))
end

function Base.similar(::FusionTensor, elt::Type, new_axes::Tuple{<:Tuple,<:Tuple})
  return FusionTensor(elt, new_axes[1], new_axes[2])
end

Base.show(io::IO, ft::FusionTensor) = print(io, "$(ndims(ft))-dim FusionTensor")

function Base.show(io::IO, ::MIME"text/plain", ft::FusionTensor)
  println(io, "$(ndims(ft))-dim FusionTensor with axes:")
  for ax in axes(ft)
    display(ax)
    println(io)
  end
  return nothing
end

Base.size(ft::FusionTensor) = Sectors.quantum_dimension.(axes(ft))
