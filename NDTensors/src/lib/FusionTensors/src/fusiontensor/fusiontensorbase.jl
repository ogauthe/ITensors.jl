# This files overloads Base functions for FusionTensor

using NDTensors.FusionTensors: FusionTensor, data_matrix, codomain_axes, domain_axes
using NDTensors.GradedAxes: dual

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), x * data_matrix(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), x * data_matrix(ft))
end

# tensor contraction is a block data_matrix product.
function Base.:*(
  left::FusionTensor{T,N,NCoAxes,NDoAxes}, right::FusionTensor{T,M,NDoAxes}
) where {T,N,NCoAxes,NDoAxes,M}

  # check consistency
  if domain_axes(left) != dual.(codomain_axes(right))
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) * data_matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(right), new_data_matrix)
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block data_matrix add.
function Base.:+(
  left::FusionTensor{T,N,NCoAxes,NDoAxes,G}, right::FusionTensor{T,N,NCoAxes,NDoAxes,G}
) where {T,N,NCoAxes,NDoAxes,G}
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) + data_matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(left), new_data_matrix)
end

function Base.:-(ft::FusionTensor)
  new_data_matrix = -data_matrix(ft)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), new_data_matrix)
end

function Base.:-(
  left::FusionTensor{T,N,NCoAxes,NDoAxes,G}, right::FusionTensor{T,N,NCoAxes,NDoAxes,G}
) where {T,N,NCoAxes,NDoAxes,G}
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end

  new_data_matrix = data_matrix(left) - data_matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(left), new_data_matrix)
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), data_matrix(ft) / x)
end

# adjoint is costless: dual axes, swap domain and codomain, take data_matrix adjoint.
# data_matrix coeff are not modified (beyond complex conjugation)
function Base.adjoint(ft::FusionTensor)
  return FusionTensor(
    dual.(domain_axes(ft)), dual.(codomain_axes(ft)), adjoint(data_matrix(ft))
  )
end

Base.axes(ft::FusionTensor) = (codomain_axes(ft)..., domain_axes(ft)...)

# conj is defined as coefficient wise complex conjugation, without axis dual
function Base.conj(ft::FusionTensor)
  return FusionTensor{M}(codomain_axes(ft), domain_axes(ft), conj(data_matrix(ft)))
end

function Base.copy(ft::FusionTensor)
  new_data_matrix = copy(data_matrix(ft))
  new_codomain_axes = copy.(codomain_axes(ft))
  new_domain_axes = copy.(domain_axes(ft))
  return FusionTensor(new_codomain_axes, new_domain_axes, new_data_matrix)
end

function Base.deepcopy(ft::FusionTensor)
  new_data_matrix = deepcopy(data_matrix(ft))
  new_codomain_axes = deepcopy.(codomain_axes(ft))
  new_domain_axes = deepcopy.(domain_axes(ft))
  return FusionTensor(new_codomain_axes, new_domain_axes, new_data_matrix)
end

# eachindex is automatically defined for AbstractArray. We do now want it.
function Base.eachindex(::FusionTensor)
  throw(DomainError("eachindex", "eachindex not defined for FusionTensor"))
end

Base.ndims(::FusionTensor{T,N}) where {T,N} = N

# Base.permutedims is defined in a separate file

Base.show(io::IO, ft::FusionTensor) = print(io, "$(ndims(ft))-dim FusionTensor")

function Base.show(io::IO, ::MIME"text/plain", ft::FusionTensor)
  println(io, "$(ndims(ft))-dim FusionTensor with axes:")
  for ax in axes(ft)
    display(ax)
    println(io)
  end
  return nothing
end

Base.size(ft::FusionTensor) = length.(axes(ft))
