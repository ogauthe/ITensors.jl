# This file defines helper functions to access FusionTensor internal structures

function translate_outer_block_to_inner(outer_block, blocklength::NTuple{<:Any,Integer})
  return BlockArrays.Block(LinearIndices(blocklength)[(Int.(Tuple(outer_block))...)])
end

function translate_inner_block_to_outer(inner_block, blocklength::NTuple{<:Any,Integer})
  return BlockArrays.Block(Tuple(CartesianIndices(blocklength)[Int(inner_block)]))
end

struct FusionTensorBlockStructure{A,B,C,D}
  allowed_sectors::A
  domain_blocklength::B
  codomain_blocklength::C
  row_blockranges::D
  col_blockranges::D
end

get_allowed_sectors(ftbs::FusionTensorBlockStructure) = ftbs.allowed_sectors
get_row_blockranges(ftbs::FusionTensorBlockStructure) = ftbs.row_blockranges
get_col_blockranges(ftbs::FusionTensorBlockStructure) = ftbs.col_blockranges
get_domain_blocklength(ftbs::FusionTensorBlockStructure) = ftbs.domain_blocklength
get_codomain_blocklength(ftbs::FusionTensorBlockStructure) = ftbs.codomain_blocklength

function FusionTensorBlockStructure(
  domain_legs::Tuple{Vararg{AbstractUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractUnitRange}},
)
  row_axis = GradedAxes.dual(GradedAxes.fusion_product(GradedAxes.dual.(domain_legs)...))
  col_axis = GradedAxes.fusion_product(codomain_legs...)
  allowed_sectors = intersect_sectors(
    GradedAxes.blocklabels(row_axis), GradedAxes.blocklabels(col_axis)
  )
  return FusionTensorBlockStructure(domain_legs, codomain_legs, allowed_sectors)
end

function FusionTensorBlockStructure(
  domain_legs::Tuple{Vararg{AbstractUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  allowed_sectors::Vector{<:Sectors.AbstractCategory},
)
  row_blockranges = compute_inner_ranges(GradedAxes.dual.(domain_legs), allowed_sectors)
  col_blockranges = compute_inner_ranges(codomain_legs, allowed_sectors)
  domain_blocklength = BlockArrays.blocklength.(domain_legs)
  codomain_blocklength = BlockArrays.blocklength.(codomain_legs)
  return FusionTensorBlockStructure(
    allowed_sectors,
    domain_blocklength,
    codomain_blocklength,
    row_blockranges,
    col_blockranges,
  )
end

function find_block_ranges(
  ftbs::FusionTensorBlockStructure,
  domain_block,
  codomain_block,
  c::Sectors.AbstractCategory,
)
  i_sec = findfirst(==(c), get_allowed_sectors(ftbs))
  return find_block_ranges(ftbs, domain_block, codomain_block, i_sec)
end

function find_block_ranges(
  ftbs::FusionTensorBlockStructure,
  domain_block,  # either Tuple{N_DO} or Block{N_DO}
  codomain_block,  # either Tuple{N_CO} or Block{N_CO}
  i_sec::Integer,
)
  row_block = translate_outer_block_to_inner(domain_block, get_domain_blocklength(ftbs))
  row_range = access_symmetric_block_range(get_row_blockranges(ftbs), row_block, i_sec)
  col_block = translate_outer_block_to_inner(codomain_block, get_codomain_blocklength(ftbs))
  col_range = access_symmetric_block_range(get_col_blockranges(ftbs), col_block, i_sec)
  return row_range, col_range
end

function access_symmetric_block_range(
  inner_blockranges::AbstractArray, inner_block, i_sec::Integer
)
  return inner_blockranges[Int(inner_block), i_sec]
end

# TBD speed up: a and b are sorted
function find_shared_indices(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
  indices1 = findall(in(b), a)
  indices2 = findall(in(a), b)
  return indices1, indices2
end

function compute_inner_ranges(legs::Tuple{Vararg{AbstractUnitRange}})
  allowed_sectors = GradedAxes.blocklabels(GradedAxes.fusion_product(legs...))
  return compute_inner_ranges(legs, allowed_sectors)
end

function compute_inner_ranges(
  ::Tuple{}, allowed_sectors::Vector{<:Sectors.AbstractCategory}
)
  return reshape([1:Sectors.istrivial(s) for s in allowed_sectors], 1, :)
end

function compute_inner_ranges(
  legs::Tuple{Vararg{AbstractUnitRange}},
  allowed_sectors::Vector{<:Sectors.AbstractCategory},
)
  blocklengths = BlockArrays.blocklengths.(legs)
  blocklabels = GradedAxes.blocklabels.(legs)
  n_sec = length(allowed_sectors)
  n_blocks = prod(length.(blocklengths))
  m = zeros(Int, n_blocks + 1, n_sec)
  for (i_block, it) in enumerate(Tuple.(CartesianIndices(BlockArrays.blocklength.(legs))))
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
  return index_matrix
end
