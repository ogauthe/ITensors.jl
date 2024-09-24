# This file defines helper functions to access FusionTensor internal structures

struct FusedAxes{A,B,C,D}
  outer_axes::A
  fused_axis::B
  inner_ranges::C
  inner_block_indices::D

  function FusedAxes(
    outer_legs::Tuple{Vararg{AbstractUnitRange}}, t::Type{<:Sectors.AbstractCategory}
  )
    @assert all(eltype.(GradedAxes.blocklabels.(outer_legs)) .== t)
    fused_axis, inner_ranges = compute_inner_ranges(outer_legs)
    inner_block_indices = Tuple.(CartesianIndices(BlockArrays.blocklength.(outer_legs)))
    return new{
      typeof(outer_legs),typeof(fused_axis),typeof(inner_ranges),typeof(inner_block_indices)
    }(
      outer_legs, fused_axis, inner_ranges, inner_block_indices
    )
  end

  function FusedAxes(::Tuple{}, t::Type{<:Sectors.AbstractCategory})
    fused_axis, inner_ranges = compute_inner_ranges(t)
    inner_block_indices = [()]
    return new{Tuple{},typeof(fused_axis),typeof(inner_ranges),typeof(inner_block_indices)}(
      (), fused_axis, inner_ranges, inner_block_indices
    )
  end
end

# getters
fused_axis(fa::FusedAxes) = fa.fused_axis
inner_ranges(fa::FusedAxes) = fa.inner_ranges
inner_block_indices(fa::FusedAxes) = fa.inner_block_indices

# Base interface
Base.axes(fa::FusedAxes) = fa.outer_axes
Base.iterate(fa::FusedAxes) = iterate(inner_block_indices(fa))
Base.iterate(fa::FusedAxes, x) = iterate(inner_block_indices(fa), x)
Base.ndims(fa::FusedAxes) = length(outer_axes(fa))

function compute_inner_ranges(t::Type{<:Sectors.AbstractCategory})
  fused_axis = GradedAxes.gradedrange([Sectors.trivial(t) => 1])
  inner_ranges = Dict([(1, Sectors.trivial(t)) => 1:1])
  return fused_axis, inner_ranges
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

  inner_ranges = Dict{Tuple{Int,eltype(allowed_sectors)},UnitRange{Int64}}()
  for it in CartesianIndices(index_matrix)
    length(index_matrix[it]) < 1 && continue
    inner_ranges[it[1], allowed_sectors[it[2]]] = index_matrix[it]
  end
  return fused_axis, inner_ranges
end

function translate_inner_block_to_outer(inner_block, fa::FusedAxes)
  return translate_inner_block_to_outer(Tuple(inner_block), fa)
end

function translate_inner_block_to_outer(inner_block::Tuple{Int}, fa::FusedAxes)
  return inner_block_indices(fa)[inner_block...]
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

function find_block_range(fa::FusedAxes, outer_block, c::Sectors.AbstractCategory)
  i_block = translate_outer_block_to_inner(outer_block, fa)
  return find_block_range(fa, i_block, c)
end

function find_block_range(fa::FusedAxes, i_block::Integer, c::Sectors.AbstractCategory)
  return inner_ranges(fa)[i_block, c]
end
