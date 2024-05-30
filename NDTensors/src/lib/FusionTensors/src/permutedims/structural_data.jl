# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD
# * dual in codomain
# * BlockArray?

struct StructuralData{P,NCoAxesIn,NDoAxesIn,C}
  permutation::P

  # inner constructor to impose constraints on types
  function StructuralData(
    perm::TensorAlgebra.BlockedPermutation{2,N},
    sectors_codomain_in::NTuple{NCoAxesIn,Vector{C}},
    sectors_domain_in::NTuple{NDoAxesIn,Vector{C}},
    arrow_directions_in::NTuple{N,Bool},
    #isometries::Matrix{Mat},  # or Dict(int)
  ) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
    if NCoAxesIn + NDoAxesIn != N
      return error("permutation incompatible with axes")
    end
    return new{typeof(perm),NCoAxesIn,NCoAxesIn,C}(perm)
  end
end

# getters
permutation(sd::StructuralData) = sd.permutation

function Base.ndims(::StructuralData{<:Any,NCoAxesIn,NDoAxesIn}) where {NCoAxesIn,NDoAxesIn}
  return NCoAxesIn + NDoAxesIn
end
ndims_codomain_in(::StructuralData{<:Any,NCoAxesIn}) where {NCoAxesIn} = NCoAxesIn
ndims_domain_in(::StructuralData{<:Any,<:Any,NDoAxesIn}) where {NDoAxesIn} = NDoAxesIn

function ndims_codomain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[1]
end

function ndims_domain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[2]
end

########################  Constructor from Clebsch-Gordan trees ############################
function contract_projector(
  tree_codomain_config_sector::Array{Float64},
  tree_domain_config_sector::Array{Float64},
  irreps_perm::NTuple{N,Int},
) where {N}
  # compile time values
  NCoAxes = ndims(tree_codomain_config_sector) - 2
  NDoAxes = ndims(tree_domain_config_sector) - 2
  @assert NDoAxes + NCoAxes == N

  irrep_dims_prod =
    prod(size(tree_codomain_config_sector)[begin:NCoAxes]) *
    prod(size(tree_domain_config_sector)[begin:NDoAxes])
  if length(tree_codomain_config_sector) > 0 && length(tree_domain_config_sector) > 0
    labels_codomain = (ntuple(identity, NCoAxes)..., N + 3, N + 1)
    labels_domain = (ntuple(i -> i + NCoAxes, NDoAxes)..., N + 3, N + 2)
    labels_dest = (irreps_perm..., N + 1, N + 2)

    #          ----------------dim_sec---------
    #          |                              |
    #          |   ndof_codomain_sec          |  ndof_domain_sec
    #           \  /                           \  /
    #            \/                             \/
    #            /                              /
    #           /                              /
    #          /\                             /\
    #         /  \                           /  \
    #        /\   \                         /\   \
    #       /  \   \                       /  \   \
    #     dim1 dim2 dim3                 dim4 dim5 dim6
    #
    projector = TensorAlgebra.contract(
      labels_dest,
      tree_codomain_config_sector,
      labels_codomain,
      tree_domain_config_sector,
      labels_domain,
    )

    # reshape as a matrix
    #           ---------------projector_sector---------
    #           |                                      |
    #   dim1*dim2*...*dim6               ndof_codomain_sec*ndof_domain_sec
    #
    return reshape(projector, (irrep_dims_prod, :))
  end
  return zeros((irrep_dims_prod, 0))  # some trees are empty: return projector on null space
end

function overlap_cg_trees(
  trees_codomain_in_config::Vector{<:Array{Float64}},
  trees_domain_in_config::Vector{<:Array{Float64}},
  trees_codomain_out_config::Vector{<:Array{Float64}},
  trees_domain_out_config::Vector{<:Array{Float64}},
  irreps_perm::NTuple{N,Int},
) where {N}
  # compile time
  NCoAxesIn = ndims(eltype(trees_codomain_in_config)) - 2
  NDoAxesIn = ndims(eltype(trees_domain_in_config)) - 2
  NCoAxesOut = ndims(eltype(trees_codomain_out_config)) - 2
  NDoAxesOut = ndims(eltype(trees_domain_out_config)) - 2
  @assert NCoAxesIn + NDoAxesIn == N
  @assert NCoAxesOut + NDoAxesOut == N
  @assert N > 0

  # initialize output as a BlockArray with
  # blocksize: (n_sectors_out, n_sectors_in)
  # blocksizes: (ndof_config_sectors_out, ndof_config_sectors_in)
  block_rows =
    size.(trees_codomain_out_config, NCoAxesOut + 2) .*
    size.(trees_domain_out_config, NDoAxesOut + 2)
  block_cols =
    size.(trees_codomain_in_config, NCoAxesIn + 2) .*
    size.(trees_domain_in_config, NDoAxesIn + 2)
  isometry = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  # contract codomain and domain CG trees to construct projector on each allowed sector
  projectors_out =
    contract_projector.(
      trees_codomain_out_config, trees_domain_out_config, Ref(ntuple(identity, N))
    )
  projectors_in =
    contract_projector.(trees_codomain_in_config, trees_domain_in_config, Ref(irreps_perm))

  for (i, j) in Iterators.product(eachindex.((projectors_out, projectors_in))...)
    # contract projectors in and out to construct singlet space base change matrix
    # construction is made blockwise. One block (i,j) corresponds to fusing incoming axes
    # over sector sec_j and outcoming axes over sector sec_i. It has a 4-dim tensor
    # internal structure which is compressed into a matrix.
    #
    #          --------------dim_sec_j---------
    #          |                              |
    #          |   ndof_codomain_sec_in       |  ndof_domain_sec_in
    #           \  /                           \  /
    #            \/                             \/
    #            /                              /
    #           /                              /
    #          /\                             /\
    #         /  \                           /  \
    #        /\   \                         /\   \
    #       /  \   \                       /  \   \
    #     dim1 dim2 dim3                 dim4 dim5 dim6
    #      |    |   |                      |   |   |
    #     -------------------------------------------
    #     --------------- irreps_perm ---------------
    #     -------------------------------------------
    #      |    |    |    |              |     |
    #     dim4 dim1 dim2 dim6           dim3  dim5
    #       \   /   /    /                \   /
    #        \ /   /    /                  \ /
    #         \   /    /                    \
    #          \ /    /                      \
    #           \    /                        \
    #            \  /                          \
    #             \/                            \
    #              \                             \
    #               \                             \
    #               /\                            /\
    #              /  \                          /  \
    #             |   ndof_codomain_sec_out     |  ndof_domain_sec_out
    #             |                             |
    #             -------------dim_sec_i---------
    #
    dim_sec_i = size(trees_codomain_out_config[i], NCoAxesOut + 1)
    isometry[BlockArrays.Block(i, j)] = (projectors_out[i]'projectors_in[j]) / dim_sec_i
  end

  return isometry
end

function compute_isometries_CG(
  biperm::TensorAlgebra.BlockedPermutation{2,N},
  sectors_codomain_in::NTuple{NCoAxesIn,Vector{C}},
  sectors_domain_in::NTuple{NDoAxesIn,Vector{C}},
  arrow_directions_in::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}

  # compile time
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
  NCoAxesOut, NDoAxesOut = BlockArrays.blocklengths(biperm)
  irreps_perm = Tuple(biperm)
  perm1, perm2 = BlockArrays.blocks(biperm)

  # define axes, isdual and allowed sectors
  allowed_sectors_in = intersect_sectors(
    sectors_codomain_in, sectors_domain_in, arrow_directions_in
  )
  sectors_in = (sectors_codomain_in..., sectors_domain_in...)
  sectors_codomain_out = getindex.(Ref(sectors_in), perm1)
  sectors_domain_out = getindex.(Ref(sectors_in), perm2)
  arrow_directions_out = getindex.(Ref(arrow_directions_in), irreps_perm)
  isdual_codomain_in = .!arrow_directions_in[begin:NCoAxesIn]  # TBD
  isdual_domain_in = arrow_directions_in[(NCoAxesIn + 1):end]
  isdual_codomain_out = .!getindex.(Ref(arrow_directions_in), perm1)  # TBD
  isdual_domain_out = getindex.(Ref(arrow_directions_in), perm2)
  allowed_sectors_out = intersect_sectors(
    sectors_codomain_out, sectors_domain_out, arrow_directions_out
  )

  # initialize output
  isometries = Vector{
    BlockArrays.BlockMatrix{  # TBD use BlockSparseArray 4-dim?
      Float64,
      Matrix{Matrix{Float64}},
      Tuple{
        BlockArrays.BlockedUnitRange{Vector{Int64}},
        BlockArrays.BlockedUnitRange{Vector{Int64}},
      },
    },
  }()
  configurations_in = Vector{NTuple{N,Int}}()

  # cache computed Clebsch-Gordan trees
  # TBD compress out trees?
  trees_codomain_in = Dict{NTuple{NCoAxesIn,Int},Vector{Array{Float64,NCoAxesIn + 2}}}()
  trees_domain_in = Dict{NTuple{NDoAxesIn,Int},Vector{Array{Float64,NDoAxesIn + 2}}}()
  trees_codomain_out = Dict{NTuple{NCoAxesOut,Int},Vector{Array{Float64,NCoAxesOut + 2}}}()
  trees_domain_out = Dict{NTuple{NDoAxesOut,Int},Vector{Array{Float64,NDoAxesOut + 2}}}()

  # loop over all sector configuration
  for it in Iterators.product(eachindex.(sectors_in)...)
    if !isempty(
      intersect_sectors(
        getindex.(sectors_codomain_in, it[begin:NCoAxesIn]),
        getindex.(sectors_domain_in, it[(NCoAxesIn + 1):end]),
      ),
    )
      trees_codomain_in_config = get_tree!(
        trees_codomain_in,
        it[begin:NCoAxesIn],
        sectors_codomain_in,
        isdual_codomain_in,
        allowed_sectors_in,
      )
      trees_domain_in_config = get_tree!(
        trees_domain_in,
        it[(NCoAxesIn + 1):end],
        sectors_domain_in,
        isdual_domain_in,
        allowed_sectors_in,
      )
      trees_codomain_out_config = get_tree!(
        trees_codomain_out,
        getindex.(Ref(it), perm1),
        sectors_codomain_out,
        isdual_codomain_out,
        allowed_sectors_out,
      )
      trees_domain_out_config = get_tree!(
        trees_domain_out,
        getindex.(Ref(it), perm2),
        sectors_domain_out,
        isdual_domain_out,
        allowed_sectors_out,
      )

      isometry = overlap_cg_trees(
        trees_codomain_in_config,
        trees_domain_in_config,
        trees_codomain_out_config,
        trees_domain_out_config,
        irreps_perm,
      )
      push!(isometries, isometry)
      push!(configurations_in, it)
    end
  end
  return isometries, configurations_in
end

###################################  Constructor from 6j  ##################################
function compute_isometries_6j(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  codomain_sectors_in::NTuple{NCoAxesIn,Vector{C}},
  domain_sectors_in::NTuple{NDoAxesIn,Vector{C}},
  arrow_directions::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
  # this function has the same inputs and the same outputs as compute_isometries_CG
end
