using BlockArrays:
  BlockArrays, AbstractBlockArray, Block, BlockIndex, BlockedUnitRange, blocks
using ..SparseArrayInterface: sparse_getindex, sparse_setindex!

# TODO: Delete this. This function was replaced
# by `nstored` but is still used in `NDTensors`.
function nonzero_keys end

abstract type AbstractBlockSparseArray{T,N} <: AbstractBlockArray{T,N} end

## Base `AbstractArray` interface

Base.axes(::AbstractBlockSparseArray) = error("Not implemented")

blockstype(::Type{<:AbstractBlockSparseArray}) = error("Not implemented")

## # Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.getindex(a::AbstractBlockSparseArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return blocksparse_getindex(a, I...)
end

## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray{<:Any,N}, I::Block{N}) where {N}
##   return ArrayLayouts.layout_getindex(a, I)
## end
##
## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray{<:Any,1}, I::Block{1})
##   return ArrayLayouts.layout_getindex(a, I)
## end
##
## # Fix ambiguity error with `BlockArrays`.
## function Base.getindex(a::AbstractBlockSparseArray, I::Vararg{AbstractVector})
##   ## return blocksparse_getindex(a, I...)
##   return ArrayLayouts.layout_getindex(a, I...)
## end

# Specialized in order to fix ambiguity error with `BlockArrays`.
function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  blocksparse_setindex!(a, value, I...)
  return a
end

function Base.setindex!(
  a::AbstractBlockSparseArray{<:Any,N}, value, I::Vararg{Block{1},N}
) where {N}
  blocksize = ntuple(dim -> length(axes(a, dim)[I[dim]]), N)
  if size(value) ≠ blocksize
    throw(
      DimensionMismatch(
        "Trying to set block $(Block(Int.(I)...)), which has a size $blocksize, with data of size $(size(value)).",
      ),
    )
  end
  blocks(a)[Int.(I)...] = value
  return a
end
