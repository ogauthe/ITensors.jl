# This files overloads Base functions for FusionTensor

using NDTensors.FusionTensors: FusionTensor, domain_axes, codomain_axes

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(axes(ft), n_codomain_axes(ft), x * matrix(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(axes(ft), n_codomain_axes(ft), x * matrix(ft))
end

# tensor contraction is a block matrix product.
function Base.:*(left::FusionTensor, right::FusionTensor)

  # check consistency
  if domain_axes(left) != dual.(codomain_axes(right))  # TODO check dual behavior
    throw(DomainError("Incompatible tensor axes"))
  end
  new_matrix = matrix(left) * matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(right), new_matrix)
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block matrix add.
function Base.:+(left::FusionTensor, right::FusionTensor)
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end
  new_matrix = matrix(left) + matrix(right)

  return FusionTensor(axes(left), n_codomain_axes(left), new_matrix)
end

function Base.:-(ft::FusionTensor)
  new_matrix = -matrix(ft)
  return FusionTensor(axes(ft), n_codomain_axes(ft), new_matrix)
end

function Base.:-(left::FusionTensor, right::FusionTensor)
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end

  new_matrix = left.matrix - right.matrix

  return FusionTensor(axes(left), n_codomain_axes(left), new_matrix)
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(axes(ft), n_codomain_axes(ft), matrix(ft) / x)
end

# adjoint is costless: dual axes, swap domain and codomain, take matrix adjoint.
# matrix coeff are not modified (beyond complex conjugation)
function Base.adjoint(ft::FusionTensor)
  return FusionTensor(dual.(domain_axes(ft)), dual.(codomain_axes(ft)), adjoint(matrix(ft)))
end

# Base.axes is defined in fusiontensor.jl as a getter

# conj is defined as coefficient wise complex conjugation, without axis dual
function Base.conj(ft::FusionTensor)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), conj(matrix(ft)))
end

function Base.copy(ft::FusionTensor)
  new_matrix = copy(matrix(ft))
  new_axes = copy.(axes(ft))
  return FusionTensor(new_axes, n_codomain_axes(ft), new_matrix)
end

function Base.deepcopy(ft::FusionTensor)
  new_matrix = deepcopy(matrix(ft))
  new_axes = deepcopy(axes(ft))
  return FusionTensor(new_axes, n_codomain_axes(ft), new_matrix)
end

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
