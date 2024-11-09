# This file defines helper functions to access FusionTensor internal structures

struct FusedAxes{A,B,C,D}
  outer_axes::A
  fused_axis::B
  index_matrix::C
  inner_block_indices::D

  function FusedAxes(
    outer_legs::NTuple{N,AbstractGradedUnitRange},
    fused_axis::AbstractGradedUnitRange,
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

# GradedAxes interface
GradedAxes.blocklabels(fa::FusedAxes) = blocklabels(fused_axis(fa))

function GradedAxes.dual(fa::FusedAxes)
  outer_legs = dual.(axes(fa))
  dual_fused_axis = dual(fused_axis(fa))
  # once dualed, dual_fused_axis != fusion_product(dual_outer_legs). TBD change this?
  dual_inner_block_indices = inner_block_indices(fa)
  dual_index_matrix = index_matrix(fa)
  return FusedAxes(outer_legs, dual_fused_axis, dual_index_matrix, dual_inner_block_indices)
end

# constructors
function FusedAxes(::Tuple{})
  fused_axis = gradedrange([TrivialSector() => 1])
  index_matrix = range.(ones(Int, 1, 1), ones(Int, 1, 1))
  inner_block_indices = CartesianIndices(())
  return FusedAxes((), fused_axis, index_matrix, inner_block_indices)
end

function FusedAxes(outer_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  fused_axis, index_matrix = compute_inner_ranges(outer_legs)
  inner_block_indices = CartesianIndices(blocklength.(outer_legs))
  return FusedAxes(outer_legs, fused_axis, index_matrix, inner_block_indices)
end

function compute_inner_ranges(outer_legs::Tuple{Vararg{AbstractGradedUnitRange}})
  fused_axis = fusion_product(outer_legs...)
  allowed_sectors = blocklabels(fused_axis)
  outer_blocklengths = blocklengths.(outer_legs)
  outer_blocklabels = blocklabels.(outer_legs)
  n_sec = length(allowed_sectors)
  n_blocks = prod(length.(outer_blocklengths))
  m = zeros(Int, n_blocks + 1, n_sec)
  for (i_block, it) in enumerate(Tuple.(CartesianIndices(blocklength.(outer_legs))))
    # TODO fuse directly blocklengths
    # requires fixed blocklengths(dual) && fusion_product(::LabelledNumber,::GradedUnitRange)
    # TBD if fusion_product is too slow, compute it recursively
    block_axis = fusion_product(getindex.(outer_blocklabels, Tuple(it))...)
    block_sectors = blocklabels(block_axis)
    for (i1, i2) in find_shared_indices(allowed_sectors, block_sectors)
      m[i_block + 1, i1] =
        prod(getindex.(outer_blocklengths, it)) * length(block_axis[Block(i2)])
    end
  end
  mcs = cumsum(m; dims=1)
  index_matrix = range.(mcs[begin:(end - 1), :] .+ 1, mcs[2:end, :])

  return fused_axis, index_matrix
end

# allow Block{1}, Tuple{Int} or Int as input
function unravel_index(inner_block, fa::FusedAxes)
  return Tuple(inner_block_indices(fa)[Tuple(inner_block)...])
end

# allow Block{N}, NTuple{N,Int} or CartesianIndex{N} as input
function ravel_multi_index(outer_block, fa::FusedAxes)
  return LinearIndices(inner_block_indices(fa))[Tuple(outer_block)...]
end

function find_sector_index(s::AbstractSector, fa::FusedAxes)
  return findfirst(==(s), blocklabels(fa))
end

function find_block_range(fa::FusedAxes, outer_block, s::AbstractSector)
  i_block = ravel_multi_index(outer_block, fa)
  return find_block_range(fa, i_block, s)
end

function find_block_range(fa::FusedAxes, i_block::Integer, s::AbstractSector)
  # use == instead of a hash function to ensure e.g. TrivialSector() can be found from U1(0)
  # as well as the opposite (evaluate as equal, but hash differ)

  return find_block_range(fa, i_block, find_sector_index(s, fa))
end

function find_block_range(fa::FusedAxes, i_block::Integer, i_sector::Integer)
  return index_matrix(fa)[i_block, i_sector]
end

function block_external_multiplicities(fa::FusedAxes, i_outer_block::Int)
  return block_external_multiplicities(fa, unravel_index(i_outer_block, fa))
end
function block_external_multiplicities(fa::FusedAxes, outer_block)
  return block_external_multiplicities(fa, Int.(Tuple(outer_block)))
end
function block_external_multiplicities(fa::FusedAxes, outer_block::NTuple{N,Int}) where {N}
  return ntuple(i -> length(axes(fa)[i][Block(outer_block[i])]), N)
end

function block_structural_multiplicity(fa::FusedAxes, outer_block, s::AbstractSector)
  i_outer_block = ravel_multi_index(outer_block, fa)
  i_sec = find_sector_index(s, fa)
  return block_structural_multiplicity(fa, i_outer_block, i_sec)
end
function block_structural_multiplicity(fa::FusedAxes, i_outer_block::Int, i_sec::Int)
  return length(index_matrix(fa)[i_outer_block, i_sec]) ÷
         prod(block_external_multiplicities(fa, i_outer_block))
end
block_structural_multiplicity(::FusedAxes, ::Int, ::Nothing) = 0

function find_shared_indices(left_labels::AbstractVector, right_labels::AbstractVector)
  # cannot use intersect in case e.g. left = Trivial, right = U1(0)
  # TBD use searchsort? What of dual?
  matches = left_labels .== reshape(right_labels, (1, :))
  shared = reinterpret(Tuple{Int,Int}, findall(matches))
  return shared
end

function Base.intersect(left::FusedAxes, right::FusedAxes)
  left_labels = blocklabels(left)
  right_labels = blocklabels(right)
  return find_shared_indices(left_labels, right_labels)
end

function allowed_outer_blocks_sectors(
  left::FusedAxes, right::FusedAxes, shared_indices::AbstractVector
)
  left_labels = blocklabels(left)
  left_indices = first.(shared_indices)
  right_indices = last.(shared_indices)
  @assert left_labels[left_indices] == blocklabels(right)[right_indices]
  reduced_left = .!isempty.(index_matrix(left)[:, left_indices])
  reduced_right = .!isempty.(index_matrix(right)[:, right_indices])

  allowed_outer_blocks = Vector{NTuple{ndims(left) + ndims(right),Int}}()
  allowed_outer_block_sectors = Vector{Vector{eltype(left_labels)}}()
  for i in axes(reduced_left, 1), j in axes(reduced_right, 1)
    intersection = findall(>(0), reduced_left[i, :] .* reduced_right[j, :])
    isempty(intersection) && continue
    push!(allowed_outer_blocks, (unravel_index(i, left)..., unravel_index(j, right)...))
    push!(allowed_outer_block_sectors, left_labels[left_indices[intersection]])
  end
  return Dict(allowed_outer_blocks .=> allowed_outer_block_sectors)
end
