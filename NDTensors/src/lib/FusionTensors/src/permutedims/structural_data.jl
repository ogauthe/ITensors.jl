# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD
# * BlockArray?

struct StructuralData{P,NCoAxesIn,NDoAxesIn,C}
  permutation::P

  # inner constructor to impose constraints on types
  function StructuralData(
    perm::TensorAlgebra.BlockedPermutation{2,N},
    codomain_in_nondual_irreps::NTuple{NCoAxesIn,Vector{C}},
    domain_in_nondual_irreps::NTuple{NDoAxesIn,Vector{C}},
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
  codomain_tree::Array{Float64}, domain_tree::Array{Float64}, irreps_perm::NTuple{N,Int}
) where {N}
  # compile time values
  NCoAxes = ndims(codomain_tree) - 2
  NDoAxes = ndims(domain_tree) - 2
  @assert NDoAxes + NCoAxes == N

  irrep_dims_prod =
    prod(size(codomain_tree)[begin:NCoAxes]) * prod(size(domain_tree)[begin:NDoAxes])
  if length(codomain_tree) > 0 && length(domain_tree) > 0
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
      labels_dest, codomain_tree, labels_codomain, domain_tree, labels_domain
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
  codomain_in_block_trees::Vector{<:Array{Float64}},
  domain_in_block_trees::Vector{<:Array{Float64}},
  codomain_out_block_trees::Vector{<:Array{Float64}},
  domain_out_block_trees::Vector{<:Array{Float64}},
  irreps_perm::NTuple{N,Int},
) where {N}
  # compile time
  NCoAxesIn = ndims(eltype(codomain_in_block_trees)) - 2
  NDoAxesIn = ndims(eltype(domain_in_block_trees)) - 2
  NCoAxesOut = ndims(eltype(codomain_out_block_trees)) - 2
  NDoAxesOut = ndims(eltype(domain_out_block_trees)) - 2
  @assert NCoAxesIn + NDoAxesIn == N
  @assert NCoAxesOut + NDoAxesOut == N
  @assert N > 0

  # initialize output as a BlockArray with
  # blocksize: (n_sectors_out, n_sectors_in)
  # blocksizes: (ndof_config_sectors_out, ndof_config_sectors_in)
  block_rows =
    size.(codomain_out_block_trees, NCoAxesOut + 2) .*
    size.(domain_out_block_trees, NDoAxesOut + 2)
  block_cols =
    size.(codomain_in_block_trees, NCoAxesIn + 2) .*
    size.(domain_in_block_trees, NDoAxesIn + 2)
  isometry = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  # contract codomain and domain CG trees to construct projector on each allowed sector
  projectors_out =
    contract_projector.(
      codomain_out_block_trees, domain_out_block_trees, Ref(ntuple(identity, N))
    )
  projectors_in =
    contract_projector.(codomain_in_block_trees, domain_in_block_trees, Ref(irreps_perm))

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
    dim_sec_i = size(codomain_out_block_trees[i], NCoAxesOut + 1)
    isometry[BlockArrays.Block(i, j)] = (projectors_out[i]'projectors_in[j]) / dim_sec_i
  end

  return isometry
end

function compute_isometries_CG(
  biperm::TensorAlgebra.BlockedPermutation{2,N},
  codomain_in_irreps::NTuple{NCoAxesIn,Vector{C}},
  domain_in_irreps::NTuple{NDoAxesIn,Vector{C}},
  in_arrows::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}

  # compile time
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
  NCoAxesOut, NDoAxesOut = BlockArrays.blocklengths(biperm)
  irreps_perm = Tuple(biperm)
  perm1, perm2 = BlockArrays.blocks(biperm)

  # define axes, isdual and allowed sectors
  in_allowed_sectors = intersect_sectors(codomain_in_irreps, domain_in_irreps)
  in_irreps = (codomain_in_sectors..., domain_in_sectors...)
  codomain_out_irreps = getindex.(Ref(in_irreps), perm1)
  domain_out_irreps = getindex.(Ref(in_irreps), perm2)
  codomain_in_arrows = .!in_arrows[begin:NCoAxesIn]
  domain_in_arrows = in_arrows[(NCoAxesIn + 1):end]
  codomain_out_arrows = .!getindex.(Ref(in_arrows), perm1)
  domain_out_arrows = getindex.(Ref(in_arrows), perm2)
  out_allowed_sectors = intersect_sectors(codomain_out_irreps, domain_out_irreps)

  # initialize output
  isometries = Vector{
    BlockArrays.BlockMatrix{   # TBD use BlockSparseArray 4-dim?
      Float64,
      Matrix{Matrix{Float64}},
      Tuple{
        BlockArrays.BlockedUnitRange{Vector{Int64}},
        BlockArrays.BlockedUnitRange{Vector{Int64}},
      },
    },
  }()
  in_blocks = Vector{NTuple{N,Int}}()

  # cache computed Clebsch-Gordan trees
  # TBD compress out trees?
  codomain_in_trees = Dict{NTuple{NCoAxesIn,Int},Vector{Array{Float64,NCoAxesIn + 2}}}()
  domain_in_trees = Dict{NTuple{NDoAxesIn,Int},Vector{Array{Float64,NDoAxesIn + 2}}}()
  codomain_out_trees = Dict{NTuple{NCoAxesOut,Int},Vector{Array{Float64,NCoAxesOut + 2}}}()
  domain_out_trees = Dict{NTuple{NDoAxesOut,Int},Vector{Array{Float64,NDoAxesOut + 2}}}()

  # loop over all sector configuration
  for it in Iterators.product(eachindex.(sectors_in)...)
    if !isempty(
      intersect_sectors(
        getindex.(codomain_in_irreps, it[begin:NCoAxesIn]),
        getindex.(domain_in_irreps, it[(NCoAxesIn + 1):end]),
      ),
    )
      codomain_in_block_trees = get_tree!(
        codomain_in_trees,
        it[begin:NCoAxesIn],
        codomain_in_irreps,
        codomain_in_arrows,
        in_allowed_sectors,
      )
      domain_in_block_trees = get_tree!(
        domain_in_trees,
        it[(NCoAxesIn + 1):end],
        domain_in_irreps,
        domain_in_arrows,
        in_allowed_sectors,
      )
      codomain_out_block_trees = get_tree!(
        codomain_out_trees,
        getindex.(Ref(it), perm1),
        codomain_out_irreps,
        codomain_out_arrows,
        out_allowed_sectors,
      )
      domain_out_block_trees = get_tree!(
        domain_out_trees,
        getindex.(Ref(it), perm2),
        domain_out_irreps,
        domain_out_arrows,
        out_allowed_sectors,
      )

      isometry = overlap_cg_trees(
        codomain_in_block_trees,
        domain_in_block_trees,
        codomain_out_block_trees,
        domain_out_block_trees,
        irreps_perm,
      )
      push!(isometries, isometry)
      push!(in_blocks, it)
    end
  end
  return isometries, in_blocks
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
