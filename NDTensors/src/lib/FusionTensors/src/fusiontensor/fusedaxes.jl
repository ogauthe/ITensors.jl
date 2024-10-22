# This file defines helper functions to access FusionTensor internal structures

struct FusedAxes{A,B,C,D}
  outer_axes::A
  fused_axis::B
  index_matrix::C
  inner_block_indices::D

  function FusedAxes(
    outer_legs::NTuple{N,AbstractUnitRange},
    fused_axis::AbstractUnitRange,
    index_matrix::Matrix{UnitRange{Int64}},
    inner_block_indices::CartesianIndices{N},
  ) where {N}
    return new{
      typeof(outer_legs),typeof(fused_axis),typeof(index_matrix),typeof(inner_block_indices)
    }(
      outer_legs, fused_axis, index_matrix, inner_block_indices
    )
  end
end

# getters
fused_axis(fa::FusedAxes) = fa.fused_axis
index_matrix(fa::FusedAxes) = fa.index_matrix
inner_block_indices(fa::FusedAxes) = fa.inner_block_indices

# Base interface
Base.axes(fa::FusedAxes) = fa.outer_axes
Base.ndims(fa::FusedAxes) = length(axes(fa))

# constructors
function FusedAxes(outer_legs::Tuple{Vararg{AbstractUnitRange}})
  fused_axis, index_matrix = compute_inner_ranges(outer_legs)
  inner_block_indices = CartesianIndices(BlockArrays.blocklength.(outer_legs))
  return FusedAxes(outer_legs, fused_axis, index_matrix, inner_block_indices)
end

function FusedAxes(::Tuple{})
  fused_axis = GradedAxes.gradedrange([SymmetrySectors.TrivialSector() => 1])
  index_matrix = range.(ones(Int, 1, 1), ones(Int, 1, 1))
  inner_block_indices = CartesianIndices(())
  return FusedAxes((), fused_axis, index_matrix, inner_block_indices)
end

function GradedAxes.dual(fa::FusedAxes)
  outer_legs = GradedAxes.dual.(axes(fa))
  dual_fused_axis = GradedAxes.dual(fused_axis(fa))
  # once dualed, dual_fused_axis != fusion_product(dual_outer_legs). TBD change this?
  dual_inner_block_indices = inner_block_indices(fa)
  dual_index_matrix = index_matrix(fa)
  return FusedAxes(outer_legs, dual_fused_axis, dual_index_matrix, dual_inner_block_indices)
end

function compute_inner_ranges(outer_legs::Tuple{Vararg{AbstractUnitRange}})
  fused_axis = GradedAxes.fusion_product(outer_legs...)
  allowed_sectors = GradedAxes.blocklabels(fused_axis)
  blocklengths = BlockArrays.blocklengths.(outer_legs)
  blocklabels = GradedAxes.blocklabels.(outer_legs)
  n_sec = length(allowed_sectors)
  n_blocks = prod(length.(blocklengths))
  m = zeros(Int, n_blocks + 1, n_sec)
  for (i_block, it) in
      enumerate(Tuple.(CartesianIndices(BlockArrays.blocklength.(outer_legs))))
    # TODO fuse directly blocklengths
    # requires fixed blocklengths(dual) && fusion_product(::LabelledNumber,::GradedUnitRange)
    # TBD if fusion_product is too slow, compute it recursively
    block_axis = GradedAxes.fusion_product(getindex.(blocklabels, Tuple(it))...)
    block_sectors = GradedAxes.blocklabels(block_axis)

    indices1, indices2 = find_shared_indices(allowed_sectors, block_sectors)
    for (i1, i2) in zip(indices1, indices2)
      m[i_block + 1, i1] =
        prod(getindex.(blocklengths, it)) * length(block_axis[BlockArrays.Block(i2)])
    end
  end
  mcs = cumsum(m; dims=1)
  index_matrix = range.(mcs[begin:(end - 1), :] .+ 1, mcs[2:end, :])

  return fused_axis, index_matrix
end

function translate_inner_block_to_outer(inner_block, fa::FusedAxes)
  return translate_inner_block_to_outer(Tuple(inner_block), fa)
end

function translate_inner_block_to_outer(inner_block::Tuple{Int}, fa::FusedAxes)
  return Tuple(inner_block_indices(fa)[inner_block...])
end

function translate_outer_block_to_inner(outer_block, fa::FusedAxes)
  return translate_outer_block_to_inner(Tuple(outer_block), fa)
end

function translate_outer_block_to_inner(outer_block::Tuple{Vararg{Int}}, fa::FusedAxes)
  return LinearIndices(inner_block_indices(fa))[outer_block...]
end

# TBD speed up: a and b are sorted
function find_shared_indices(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
  indices1 = findall(in(b), a)
  indices2 = findall(in(a), b)
  return indices1, indices2
end

function find_block_range(fa::FusedAxes, outer_block, s::SymmetrySectors.AbstractSector)
  i_block = translate_outer_block_to_inner(outer_block, fa)
  return find_block_range(fa, i_block, s)
end

function find_block_range(
  fa::FusedAxes, i_block::Integer, s::SymmetrySectors.AbstractSector
)
  # use == instead of a hash function to ensure e.g. TrivialSector() can be found from U1(0)
  # as well as the opposite (evaluate as equal, but hash differ)
  i_sector = findfirst(==(s), GradedAxes.blocklabels(fused_axis(fa)))
  return find_block_range(fa, i_block, i_sector)
end

function find_block_range(fa::FusedAxes, i_block::Integer, i_sector::Integer)
  return index_matrix(fa)[i_block, i_sector]
end

function kept_indices(left_labels::AbstractVector, right_labels::AbstractVector)
  # cannot use intersect/searchsort in case e.g. left = Trivial, right = U1(0)
  # TBD use searchsort? What of adjoint?
  matches = left_labels .== reshape(right_labels, (1, :))
  kept_ind = reinterpret(Tuple{Int,Int}, findall(matches))
  return kept_ind
end

function Base.intersect(left::FusedAxes, right::FusedAxes)
  left_labels = GradedAxes.blocklabels(fused_axis(left))
  right_labels = GradedAxes.blocklabels(fused_axis(right))
  return kept_indices(left_labels, right_labels)
end

function Base.intersect(
  left::FusedAxes,
  right::FusedAxes,
  existing_sectors::Vector{<:SymmetrySectors.AbstractSector},
)
  # assume existing_sectors ⊂ left_labels && existing_sectors ⊂ right_labels
  left_labels = GradedAxes.blocklabels(fused_axis(left))
  right_labels = GradedAxes.blocklabels(fused_axis(right))

  kept_left = kept_indices(left_labels, existing_sectors)
  kept_right = kept_indices(right_labels, existing_sectors)
  matches = kept_indices(last.(kept_left), last.(kept_right))
  kept_inds = map(c -> (first(kept_left[first(c)]), first(kept_right[last(c)])), matches)
  return kept_inds
end

function allowed_outer_blocks_sectors(
  left::FusedAxes, right::FusedAxes, kept_indices::AbstractVector
)
  left_labels = GradedAxes.blocklabels(fused_axis(left))
  left_indices = first.(kept_indices)
  reduced_left = .!isempty.(index_matrix(left)[:, left_indices])
  reduced_right = .!isempty.(index_matrix(right)[:, last.(kept_indices)])

  allowed_outer_blocks = Vector{NTuple{ndims(left) + ndims(right),Int}}()
  allowed_outer_block_sectors = Vector{Vector{eltype(left_labels)}}()
  for i in 1:size(reduced_left, 1), j in 1:size(reduced_right, 1)
    intersection = findall(>(0), reduced_left[i, :] .* reduced_right[j, :])  #  ASSUME SORTED LABELS ON BOTH SIDES
    isempty(intersection) && continue
    push!(
      allowed_outer_blocks,
      (
        translate_inner_block_to_outer(i, left)...,
        translate_inner_block_to_outer(j, right)...,
      ),
    )
    push!(allowed_outer_block_sectors, left_labels[left_indices[intersection]])
  end
  return allowed_outer_blocks .=> allowed_outer_block_sectors
end
