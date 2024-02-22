# This files overloads Base functions for FusionTensor

using Printf
using NDTensors.FusionTensors: FusionTensor, domain_axes, codomain_axes
using ITensors: @debug_check

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(axes(ft), n_row_legs(ft), x * matrix(ft))
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(axes(ft), n_row_legs(ft), x * matrix(ft))
end

# tensor contraction is a block matrix product.
function Base.:*(lefft::FusionTensor, righft::FusionTensor)

  # check consistency
  if domain_axes(left) != dual.(codomain_axes(right))  # TODO check dual behavior
    throw(DomainError("Incompatible tensor axes"))
  end
  new_matrix = matrix(left) * matrix(right)

  return FusionTensor(codomain_axes(left), domain_axes(right), new_matrix)
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block matrix add.
function Base.:+(lefft::FusionTensor, righft::FusionTensor)
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

function Base.:-(lefft::FusionTensor, righft::FusionTensor)
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
  tdag = dagger(ft)
  return FusionTensor(axes(tdag), n_domain_legs(tdag), conj(matrix(tdag)))
end

# complex conjugation, no dual
function Base.conj(ft::FusionTensor)
  return FusionTensor(
    codomain_axes,
    t.domain_axes,
    conj(t.matrix),  # TBD impose sorting?
  )
end

function Base.copy(ft::FusionTensor)
  new_matrix = copy(matrix(ft))
  new_axes = copy(axes(ft))
  return FusionTensor(new_axes, n_row_legs(ft), new_matrix)
end

function Base.deecopy(ft::FusionTensor)
  new_matrix = deepcopy(matrix(ft))
  new_axes = deepcopy(axes(ft))
  return FusionTensor(new_axes, n_row_legs(ft), new_matrix)
end

Base.ndims(ft::FusionTensor) = N
Base.show(io::IO, ft::FusionTensor) = @printf(io, "%d-dim FusionTensor", ndims(ft))
Base.size(ft::FusionTensor) = length.(axes(ft))
