# This files overloads Base functions for FusionTensor

using NDTensors.FusionTensors: FusionTensor, domain_axes, codomain_axes

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(axes(ft), n_row_legs(ft), x * matrix(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(axes(ft), n_row_legs(ft), x * matrix(ft))
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

  return FusionTensor(axes(left), n_row_legs(left), new_matrix)
end

function Base.:-(ft::FusionTensor)
  new_matrix = -matrix(ft)
  return FusionTensor(axes(ft), n_row_legs(ft), new_matrix)
end

function Base.:-(left::FusionTensor, right::FusionTensor)
  # check consistency
  if codomain_axes(left) != codomain_axes(right) || domain_axes(left) != domain_axes(right)
    throw(DomainError("Incompatible tensor axes"))
  end

  new_matrix = left.matrix - right.matrix

  return FusionTensor(axes(left), n_row_legs(left), new_matrix)
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(axes(ft), n_row_legs(ft), matrix(ft) / x)
end

# adjoint = dagger * conjugate
function Base.adjoint(ft::FusionTensor)
  ftdag = dagger(ft)
  return FusionTensor(axes(ftdag), n_domain_legs(ftdag), conj(matrix(ftdag)))
end

# complex conjugation, no dual
function Base.conj(ft::FusionTensor)
  return FusionTensor(codomain_axes(ft), domain_axes(ft), conj(matrix(ft)))
end

function Base.copy(ft::FusionTensor)
  new_matrix = copy(matrix(ft))
  new_axes = copy(axes(ft))
  return FusionTensor(new_axes, n_row_legs(ft), new_matrix)
end

function Base.deepcopy(ft::FusionTensor)
  new_matrix = deepcopy(matrix(ft))
  new_axes = deepcopy(axes(ft))
  return FusionTensor(new_axes, n_row_legs(ft), new_matrix)
end

Base.ndims(::FusionTensor{T,N}) where {T,N} = N
Base.show(::IO, ft::FusionTensor) = println("$(ndims(ft))-dim FusionTensor")
Base.size(ft::FusionTensor) = length.(axes(ft))
