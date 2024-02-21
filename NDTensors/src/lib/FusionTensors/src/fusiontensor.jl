# This file defines struct FusionTensor and constructors

using NDTensors.Sectors: AbstractCategory
using NDTensors.GradedAxes
using BlockArrays
using NDTensors.BlockSparseArrays: BlockSparseArray
using ITensors: @debug_check

struct FusionTensor{T<:number,C,D,E} <: AbstractMatrix{T}
  codomain_axes::C  # tuple / Vector of GradedAxes
  domain_axes::D    # tuple / Vector of GradedAxes

  matrix_blocks::E  # DiagonalBlockMatrix

  ndims::Integer  # may be different from 2!
  #nblocks::Integer
end where {T<:Number}

function check_consistency(t::FusionTensor)
  if length(t.codomain_axes) != t.n_codomain_legs
    return false
  end
  if length(t.domain_axes) != t.n_domain_legs
    return false
  end
  if t.n_codomain_legs < 1
    return false
  end
  if t.n_domain_legs < 1
    return false
  end
  if t.ndims != t.n_codomain_legs + t.n_domain_legs
    return false
  end
  if length(t.matrix_blocks) != t.nblocks
    return false
  end
  if blocklength(matrix_row_axis) != t.nblocks
    return false
  end
  if blocklength(matrix_column_axis) != t.nblocks
    return false
  end
  for i in 1:(t.nblocks)
    if size(matrix_blocks[i], 1) != blocklengths(t.matrix_row_axis)[i]
      return false
    end
    if size(matrix_blocks[i], 2) != blocklengths(t.matrix_column_axis)[i]
      return false
    end
  end

  fused_codomain = fuse(t.codomain_axes)  # this is slow
  fused_domain = fuse(t.domain_axes)  # this is slow
  # TODO define "included in"
  if !(t.matrix_row_axis < fused_codomain)
    return false
  end
  if !(t.matrix_column_axis < fused_domain)
    return false
  end
  return true
end

function FusionTensor(
  codomain_axes::Vector{GradedAxes.GradedUnitRange},
  domain_axes::Vector{GradedAxes.GradedUnitRange},
  matrix_blocks::BlockSparseArray,
)
  return FusionTensor(
    codomain_axes, domain_axes, matrix_blocks, length(codomain_axes) + length(domain_axes)
  )
end

matrix_size(t::FusionTensor) = size(t.matrix_blocks)
tensor_size(t::FusionTensor) = size(t.matrix_blocks)
n_codomain_legs(t::FusionTensor) = length(t.codomain_axes)
n_domain_legs(t::FusionTensor) = length(t.domain_axes)

# swap row and column axes, transpose matrix blocks, dual any axis. No basis change.
function dagger(t::FusionTensor)
  return FusionTensor(
    dual.(t.domain_axes),
    dual.(t.codomain_axes),
    transpose(t.matrix_blocks),  # TBD impose sorting? Currently crash BlockSparseArray
  )
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

function Base.:-(ft::FusionTensor)
  new_blocks = -ft.matrix_blocks
  return FusionTensor(ft.codomain_axes, ft.domain_axes, new_blocks)
end

# tensor addition is a block matrix add.
function Base.:+(left::FusionTensor, right::FusionTensor)
  # check consistency
  if left.codomain_axes != right.codomain_axes || left.domain_axes != right.domain_axes
    throw(DomainError("Incompatible tensor axes"))
  end

  new_blocks = left.matrix_blocks + right.matrix_blocks

  return FusionTensor(left.codomain_axes, left.domain_axes, new_blocks)
end

function Base.:-(left::FusionTensor, right::FusionTensor)
  # check consistency
  if left.codomain_axes != right.codomain_axes || left.domain_axes != right.domain_axes
    throw(DomainError("Incompatible tensor axes"))
  end

  new_blocks = left.matrix_blocks - right.matrix_blocks

  return FusionTensor(left.codomain_axes, left.domain_axes, new_blocks)
end

function Base.:*(x::Number, ft::FusionTensor)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, x * ft.matrix_blocks)
end

function Base.:*(ft::FusionTensor, x::Number)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, x * ft.matrix_blocks)
end

function Base.:/(ft::FusionTensor, x::Number)
  return FusionTensor(ft.codomain_axes, ft.domain_axes, ft.matrix_blocks / x)
end
