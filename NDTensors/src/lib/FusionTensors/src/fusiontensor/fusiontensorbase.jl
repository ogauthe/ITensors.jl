# This files overloads Base functions for FusionTensor

using NDTensors.FusionTensors: FusionTensor, data_matrix, domain_axes, codomain_axes
using NDTensors.GradedAxes: dual

function Base.:*(x::Number, ft::FusionTensor{M}) where {M}
  return FusionTensor{M}(axes(ft), x * data_matrix(ft))
end

function Base.:*(ft::FusionTensor{M}, x::Number) where {M}
  return FusionTensor{M}(axes(ft), x * data_matrix(ft))
end

# tensor contraction is a block data_matrix product.
function Base.:*(left::FusionTensor{M,K}, right::FusionTensor{K}) where {M,K}

  # check consistency
  if domain_axes(left) != dual.(codomain_axes(right))
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) * data_matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(right), new_data_matrix)
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block data_matrix add.
function Base.:+(left::FusionTensor{M}, right::FusionTensor{M}) where {M}
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end
  new_data_matrix = data_matrix(left) + data_matrix(right)

  return FusionTensor{M}(axes(left), new_data_matrix)
end

function Base.:-(ft::FusionTensor{M}) where {M}
  new_data_matrix = -data_matrix(ft)
  return FusionTensor{M}(axes(ft), new_data_matrix)
end

function Base.:-(left::FusionTensor{M}, right::FusionTensor{M}) where {M}
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end

  new_data_matrix = data_matrix(left) - data_matrix(right)

  return FusionTensor{M}(axes(left), new_data_matrix)
end

function Base.:/(ft::FusionTensor{M}, x::Number) where {M}
  return FusionTensor{M}(axes(ft), data_matrix(ft) / x)
end

# adjoint is costless: dual axes, swap domain and codomain, take data_matrix adjoint.
# data_matrix coeff are not modified (beyond complex conjugation)
function Base.adjoint(ft::FusionTensor)
  return FusionTensor(
    dual.(domain_axes(ft)), dual.(codomain_axes(ft)), adjoint(data_matrix(ft))
  )
end

# Base.axes is defined in fusiontensor.jl as a getter

# conj is defined as coefficient wise complex conjugation, without axis dual
function Base.conj(ft::FusionTensor{M}) where {M}
  return FusionTensor{M}(axes(ft), conj(data_matrix(ft)))
end

function Base.copy(ft::FusionTensor{M}) where {M}
  new_data_matrix = copy(data_matrix(ft))
  new_axes = copy.(axes(ft))
  return FusionTensor{M}(new_axes, new_data_matrix)
end

function Base.deepcopy(ft::FusionTensor{M}) where {M}
  new_data_matrix = deepcopy(data_matrix(ft))
  new_axes = deepcopy(axes(ft))
  return FusionTensor{M}(new_axes, new_data_matrix)
end

function Base.eachindex(::FusionTensor)
  throw(DomainError("eachindex", "eachindex not defined for FusionTensor"))
end

Base.ndims(::FusionTensor{M,K,T,N}) where {M,K,T,N} = N

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
