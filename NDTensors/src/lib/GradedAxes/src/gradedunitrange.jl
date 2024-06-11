using BlockArrays:
  BlockArrays,
  Block,
  BlockedUnitRange,
  BlockIndex,
  BlockRange,
  BlockSlice,
  BlockVector,
  blockedrange,
  BlockIndexRange,
  blockfirsts,
  blockisequal,
  blocklength,
  blocklengths,
  findblock,
  findblockindex,
  mortar
using ..LabelledNumbers: LabelledNumbers, LabelledInteger, label, labelled, unlabel

const GradedUnitRange{BlockLasts<:Vector{<:LabelledInteger}} = BlockedUnitRange{BlockLasts}

# == is just a range comparison that ignores labels. Need dedicated function to check equality.
function gradedisequal(a1::AbstractUnitRange, a2::AbstractUnitRange)
  return blockisequal(a1, a2) && (blocklabels(a1) == blocklabels(a2))
end

# TODO: Use `TypeParameterAccessors`.
Base.eltype(::Type{<:GradedUnitRange{<:Vector{T}}}) where {T} = T

function gradedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  brange = blockedrange(unlabel.(lblocklengths))
  lblocklasts = labelled.(blocklasts(brange), label.(lblocklengths))
  # TODO: `first` is forced to be `Int` in `BlockArrays.BlockedUnitRange`,
  # so this doesn't do anything right now. Make a PR to generalize it.
  firstlength = first(lblocklengths)
  lfirst = oneunit(firstlength)
  return BlockArrays._BlockedUnitRange(lfirst, lblocklasts)
end

# To help with generic code.
function BlockArrays.blockedrange(lblocklengths::AbstractVector{<:LabelledInteger})
  return gradedrange(lblocklengths)
end

Base.last(a::GradedUnitRange) = isempty(a.lasts) ? first(a) - 1 : last(a.lasts)

# TODO: This needs to be defined to circumvent an issue
# in the `BlockArrays.BlocksView` constructor. This
# is likely caused by issues around `BlockedUnitRange` constraining
# the element type to be `Int`, which is being fixed in:
# https://github.com/JuliaArrays/BlockArrays.jl/pull/337
# Remove this definition once that is fixed.
function BlockArrays.blocks(a::GradedUnitRange)
  # TODO: Fix `BlockRange`, try using `BlockRange` instead.
  return [a[Block(i)] for i in 1:blocklength(a)]
end

function gradedrange(lblocklengths::AbstractVector{<:Pair{<:Any,<:Integer}})
  return gradedrange(labelled.(last.(lblocklengths), first.(lblocklengths)))
end

function labelled_blocks(a::BlockedUnitRange, labels)
  return BlockArrays._BlockedUnitRange(a.first, labelled.(a.lasts, labels))
end

function BlockArrays.findblock(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblock(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblock(unlabel_blocks(a), index)
end

function blockedunitrange_findblockindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

function BlockArrays.findblockindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_findblockindex(unlabel_blocks(a), index)
end

## Block label interface

# Internal function
function get_label(a::BlockedUnitRange, index::Block{1})
  return label(blocklasts(a)[Int(index)])
end

# Internal function
function get_label(a::BlockedUnitRange, index::Integer)
  return get_label(a, blockedunitrange_findblock(a, index))
end

function blocklabels(a::BlockVector)
  return map(BlockRange(a)) do block
    return label(@view(a[block]))
  end
end

function blocklabels(a::BlockedUnitRange)
  # Using `a.lasts` here since that is what is stored
  # inside of `BlockedUnitRange`, maybe change that.
  # For example, it could be something like:
  #
  # map(BlockRange(a)) do block
  #   return label(@view(a[block]))
  # end
  #
  return label.(a.lasts)
end

# TODO: This relies on internals of `BlockArrays`, maybe redesign
# to try to avoid that.
# TODO: Define `set_grades`, `set_sector_labels`, `set_labels`.
function unlabel_blocks(a::BlockedUnitRange)
  return BlockArrays._BlockedUnitRange(a.first, unlabel.(a.lasts))
end

## BlockedUnitRage interface

function Base.axes(ga::GradedUnitRange)
  return map(axes(unlabel_blocks(ga))) do a
    return labelled_blocks(a, blocklabels(ga))
  end
end

function BlockArrays.blockfirsts(a::GradedUnitRange)
  return labelled.(blockfirsts(unlabel_blocks(a)), blocklabels(a))
end

function BlockArrays.blocklasts(a::GradedUnitRange)
  return labelled.(blocklasts(unlabel_blocks(a)), blocklabels(a))
end

function BlockArrays.blocklengths(a::GradedUnitRange)
  return labelled.(blocklengths(unlabel_blocks(a)), blocklabels(a))
end

function Base.first(a::GradedUnitRange)
  return labelled(first(unlabel_blocks(a)), label(a[Block(1)]))
end

Base.iterate(a::GradedUnitRange) = isempty(a) ? nothing : (first(a), first(a))
function Base.iterate(a::GradedUnitRange, i)
  i == last(a) && return nothing
  next = a[i + step(a)]
  return (next, next)
end

function firstblockindices(a::GradedUnitRange)
  return labelled.(firstblockindices(unlabel_blocks(a)), blocklabels(a))
end

function blockedunitrange_getindex(a::GradedUnitRange, index)
  # This uses `blocklasts` since that is what is stored
  # in `BlockedUnitRange`, maybe abstract that away.
  return labelled(unlabel_blocks(a)[index], get_label(a, index))
end

# The block labels of the corresponding slice.
function blocklabels(a::AbstractUnitRange, indices)
  return map(_blocks(a, indices)) do block
    return label(a[block])
  end
end

function blockedunitrange_getindices(
  ga::GradedUnitRange, indices::AbstractUnitRange{<:Integer}
)
  a_indices = blockedunitrange_getindices(unlabel_blocks(ga), indices)
  return labelled_blocks(a_indices, blocklabels(ga, indices))
end

# Fixes ambiguity error with:
# ```julia
# blockedunitrange_getindices(::GradedUnitRange, ::AbstractUnitRange{<:Integer})
# ```
# TODO: Try removing once GradedAxes is rewritten for BlockArrays v1.
function blockedunitrange_getindices(a::GradedUnitRange, indices::BlockSlice)
  return a[indices.block]
end

function blockedunitrange_getindices(ga::GradedUnitRange, indices::BlockRange)
  return labelled_blocks(unlabel_blocks(ga)[indices], blocklabels(ga, indices))
end

function blockedunitrange_getindices(a::GradedUnitRange, indices::BlockIndex{1})
  return a[block(indices)][blockindex(indices)]
end

function Base.getindex(a::GradedUnitRange, index::Integer)
  return blockedunitrange_getindex(a, index)
end

function Base.getindex(a::GradedUnitRange, index::Block{1})
  return blockedunitrange_getindex(a, index)
end

function Base.getindex(a::GradedUnitRange, indices::BlockIndexRange)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(
  a::GradedUnitRange, indices::BlockRange{1,<:Tuple{AbstractUnitRange{Int}}}
)
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity error with `BlockArrays`.
function Base.getindex(a::GradedUnitRange, indices::BlockRange{1,Tuple{Base.OneTo{Int}}})
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices::BlockIndex{1})
  return blockedunitrange_getindices(a, indices)
end

# Fixes ambiguity issues with:
# ```julia
# getindex(::BlockedUnitRange, ::BlockSlice)
# getindex(::GradedUnitRange, ::AbstractUnitRange{<:Integer})
# getindex(::GradedUnitRange, ::Any)
# getindex(::AbstractUnitRange, ::AbstractUnitRange{<:Integer})
# ```
# TODO: Maybe not needed once GradedAxes is rewritten
# for BlockArrays v1.
function Base.getindex(a::GradedUnitRange, indices::BlockSlice)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices)
  return blockedunitrange_getindices(a, indices)
end

function Base.getindex(a::GradedUnitRange, indices::AbstractUnitRange{<:Integer})
  return blockedunitrange_getindices(a, indices)
end
