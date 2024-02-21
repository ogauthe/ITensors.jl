# This files overloads Base functions for FusionTensor

using NDTensors.FusionTensors: FusionTensor
using ITensors: @debug_check

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, x * ft.matrix_blocks)
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, x * ft.matrix_blocks)
end

# tensor contraction is a block matrix product.
function Base.:*(left::FusionTensor, right::FusionTensor)

  # check consistency
  if left.domain_axes != right.codomain_axes
    throw(DomainError("Incompatible tensor axes"))
  end

  new_blocks = left.matrix_blocks * right.matrix_blocks

  return FusionTensor(left.codomain_axes, right.domain_axes, new_blocks)
end

Base.:+(ft::FusionTensor) = ft

# tensor addition is a block matrix add.
function Base.:+(left::FusionTensor, right::FusionTensor)
  # check consistency
  if left.codomain_axes != right.codomain_axes || left.domain_axes != right.domain_axes
    throw(DomainError("Incompatible tensor axes"))
  end

  new_blocks = left.matrix_blocks + right.matrix_blocks

  return FusionTensor(left.codomain_axes, left.domain_axes, new_blocks)
end

function Base.:-(ft::FusionTensor)
  new_blocks = -ft.matrix_blocks
  return FusionTensor(ft.codomain_axes, ft.domain_axes, new_blocks)
end
function Base.:-(left::FusionTensor, right::FusionTensor)
  # check consistency
  if left.codomain_axes != right.codomain_axes || left.domain_axes != right.domain_axes
    throw(DomainError("Incompatible tensor axes"))
  end

  new_blocks = left.matrix_blocks - right.matrix_blocks

  return FusionTensor(left.codomain_axes, left.domain_axes, new_blocks)
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, ft.matrix_blocks / x)
end

# adjoint = dagger * conjugate
function Base.adjoint(t::FusionTensor)
  tdag = dagger(t)
  return FusionTensor(tdag.codomain_axes, tdag.domain_axes, conj.(tdag.blocks))
end

# TBD conjugate imposes dual?
function Base.conj(t::FusionTensor)
  return FusionTensor(
    dual.(t.codomain_axes),
    dual.(t.domain_axes),
    conj(t.matrix_blocks),  # TBD impose sorting?
  )
end

function Base.copy(ft::FusionTensor)
  new_blocks = copy(ft.matrix_blocks)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, new_blocks)
end

function Base.deecopy(ft::FusionTensor)
  # only copy matrix_blocks.
  # TBD Copy axes?
  new_blocks = deepcopy(ft.matrix_blocks)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, new_blocks)
end

Base.ndims(t::FusionTensor) = t.ndims  # clash with AbstractMatrix
Base.size(t::FusionTensor) = size(t.matrix_blocks)  # strange object
# length = prod(size(t)) has little meaning
# axes(a) is just wrong => overload it?
# eachindex(a) is wrong => It HAS to be impossible to write/access off-diagonal blocks
# stride(a)  meaningless - unsupported by BlockSparseArray anyway
# fill! => implemented by default? Implement it for diagonal blocks?

Base.getindex(t::FusionTensor, i::Int) = t.matrix_blocks[i]  # strange object
Base.getindex(t::FusionTensor, I::Vararg{Int,2}) = t.matrix_blocks[I]  # strange object
Base.IndexStyle(::FusionTensor) = IndexCartesian()
