# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD
# * BlockArray?

# Current implementation:
# a FusionTensor is a minimal container that does not cache its internal structure
# this structure instead is stored in a StructuralData that works as a 2-level cache
#
# first cache: isometries, as defined by
# perm::TensorAlgebra.BlockedPermutation{2,N},
# irreps::NTuple{N,AbstractCategory}
# arrows::NTuple{N,Bool}
# stored as a BlockArray
# TBD IF I WANT TO REUSE ONE ISOMETRY FOR ANOTHER StructuralData,
# I NEED TO CHANGE ITS BLOCK STRUCTURE AS IT DEPENDS ON IN/OUT_SECTORS
#
#
# second cache: StructuralData, defined for many blocks
# contains list of allowed blocks + isometries for each block

struct StructuralData{N,P,Mat}
  biperm::P  # not strictly necessary but convenient
  old_block_indices::Vector{NTuple{N,Int}}
  old_domain_structural_multiplicities::Matrix{Int}
  old_codomain_structural_multiplicities::Matrix{Int}
  new_domain_structural_multiplicities::Matrix{Int}
  new_codomain_structural_multiplicities::Matrix{Int}
  isometries::Vector{Mat}
  # currently minimalistic, store strict necessary to minimize memory use
  # TBD not parametrized by C?
  # TBD store old_irreps?

  function StructuralData(
    biperm::TensorAlgebra.BlockedPermutation{2,N},
    old_block_indices::Vector{NTuple{N,Int}},
    old_domain_structural_multiplicities::Matrix{Int},
    old_codomain_structural_multiplicities::Matrix{Int},
    new_domain_structural_multiplicities::Matrix{Int},
    new_codomain_structural_multiplicities::Matrix{Int},
    isometries::Vector{<:AbstractMatrix},
  ) where {N}
    @assert size(old_domain_structural_multiplicities, 1) ==
      size(old_codomain_structural_multiplicities, 1)
    @assert size(new_domain_structural_multiplicities, 1) ==
      size(new_codomain_structural_multiplicities, 1)
    @assert size(new_domain_structural_multiplicities, 1) == length(new_allowed_sectors)
    @assert length(old_block_indices) == length(isometries)
    return new{N,typeof(biperm),eltype(isometries)}(
      biperm,
      old_block_indices,
      old_domain_structural_multiplicities,
      old_codomain_structural_multiplicities,
      new_domain_structural_multiplicities,
      new_codomain_structural_multiplicities,
      isometries,
    )
  end
end

# getters
#old_blocks(sd::StructuralData) = sd.old_blocks
Base.ndims(::StructuralData{N}) where {N} = N
get_biperm(sd::StructuralData) = sd.biperm
#structural_multiplicities_domain_ion(sd::StructuralData) = sd.structural_multiplicities
#isometries(sd::StructuralData) = sd.isometries
#new_allowed_sectors(sd::StructuralData) = sd.new_allowed_sectors

function StructuralData(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  old_domain_irreps = GradedAxes.blocklabels.(domain_axes(ft))
  old_codomain_irreps = GradedAxes.blocklabels.(codomain_axes(ft))
  old_arrows = GradedAxes.isdual.(axes(ft))
  return StructuralData(biperm, old_domain_irreps, old_codomain_irreps, old_arrows)
end

function StructuralData(
  biperm::TensorAlgebra.BlockedPermutation{2,N},
  old_domain_irreps::NTuple{OldNCoAxes,Vector{C}},
  old_codomain_irreps::NTuple{OldNDoAxes,Vector{C}},
  old_arrows::NTuple{N,Bool},
) where {N,OldNCoAxes,OldNDoAxes,C<:Sectors.AbstractCategory}
  @assert OldNCoAxes + OldNDoAxes == N
  @assert N > 0
  return StructuralData(
    biperm,
    compute_isometries_CG(biperm, old_domain_irreps, old_codomain_irreps, old_arrows)...,
  )
end

########################  Constructor from Clebsch-Gordan trees ############################
function contract_projector(
  domain_tree::AbstractArray{<:Real},
  codomain_tree::AbstractArray{<:Real},
  irreps_perm::NTuple{N,Int},
) where {N}
  # compile time values
  NCoAxes = ndims(domain_tree) - 2
  NDoAxes = ndims(codomain_tree) - 2
  @assert NDoAxes + NCoAxes == N

  irrep_dims_prod =
    prod(size(domain_tree)[begin:NCoAxes]) * prod(size(codomain_tree)[begin:NDoAxes])
  if length(domain_tree) > 0 && length(codomain_tree) > 0
    labels_domain = (ntuple(identity, NCoAxes)..., N + 3, N + 1)
    labels_codomain = (ntuple(i -> i + NCoAxes, NDoAxes)..., N + 3, N + 2)
    labels_dest = (irreps_perm..., N + 1, N + 2)

    #          ----------------dim_sec---------
    #          |                              |
    #          |  struct_mult_domain_sec    |  struct_mult_codomain_sec
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
      labels_dest, domain_tree, labels_domain, codomain_tree, labels_codomain
    )

    # reshape as a matrix
    #           ---------------projector_sector----------------
    #           |                                             |
    #   dim1*dim2*...*dim6               struct_mult_domain_sec*struct_mult_codomain_sec
    #
    return reshape(projector, (irrep_dims_prod, :))
  end
  return zeros((irrep_dims_prod, 0))  # some trees are empty: return projector on null space
end

function overlap_cg_trees(
  old_domain_block_trees::Vector{<:AbstractArray{<:Real}},
  old_codomain_block_trees::Vector{<:AbstractArray{<:Real}},
  new_domain_block_trees::Vector{<:AbstractArray{<:Real}},
  new_codomain_block_trees::Vector{<:AbstractArray{<:Real}},
  irreps_perm::NTuple{N,Int},
) where {N}
  # compile time
  OldNCoAxes = ndims(eltype(old_domain_block_trees)) - 2
  OldNDoAxes = ndims(eltype(old_codomain_block_trees)) - 2
  NCoAxesNew = ndims(eltype(new_domain_block_trees)) - 2
  NDoAxesNew = ndims(eltype(new_codomain_block_trees)) - 2
  @assert OldNCoAxes + OldNDoAxes == N
  @assert NCoAxesNew + NDoAxesNew == N
  @assert N > 0

  # initialize output as a BlockArray with
  # blocksize: (n_new_sectors, n_old_sectors)
  # blocksizes: (ndof_new_block_sectors, ndof_old_block_sectors)
  block_rows =
    size.(new_domain_block_trees, NCoAxesNew + 2) .*
    size.(new_codomain_block_trees, NDoAxesNew + 2)
  block_cols =
    size.(old_domain_block_trees, OldNCoAxes + 2) .*
    size.(old_codomain_block_trees, OldNDoAxes + 2)
  isometry = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  # contract domain and codomain CG trees to construct projector on each allowed sector
  new_projectors =
    contract_projector.(
      new_domain_block_trees, new_codomain_block_trees, Ref(ntuple(identity, N))
    )
  old_projectors =
    contract_projector.(old_domain_block_trees, old_codomain_block_trees, Ref(irreps_perm))

  for (i, j) in Iterators.product(eachindex.((new_projectors, old_projectors))...)
    # Contract new and old projectors to construct singlet space basis change matrix.
    # Construction is blockwise. One block (i,j) corresponds to fusing old axes
    # over sector sec_j and new axes over sector sec_i. It has a 4-dim tensor
    # internal structure which is compressed into a matrix.
    #
    #          --------------dim_sec_j---------
    #          |                              |
    #          |   struct_mult                |   struct_mult_old_codomain_sec_j
    #           \  / _old_domain_sec_j       \  /
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
    #             | struct_mult                 |  struct_mult_new_codomain_sec_i
    #             |   _old_domain_sec_j       |
    #             |                             |
    #             -------------dim_sec_i---------
    #
    dim_sec_i = size(new_domain_block_trees[i], NCoAxesNew + 1)
    isometry[BlockArrays.Block(i, j)] = (new_projectors[i]'old_projectors[j]) / dim_sec_i
  end

  return isometry
end

function compute_isometries_CG(
  biperm::TensorAlgebra.BlockedPermutation{2,N},
  old_domain_irreps::NTuple{OldNCoAxes,Vector{C}},
  old_codomain_irreps::NTuple{OldNDoAxes,Vector{C}},
  old_arrows::NTuple{N,Bool},
) where {N,OldNCoAxes,OldNDoAxes,C<:Sectors.AbstractCategory}

  # compile time
  NCoAxesNew, NDoAxesNew = BlockArrays.blocklengths(biperm)
  irreps_perm = Tuple(biperm)
  perm1, perm2 = BlockArrays.blocks(biperm)

  # define axes, isdual and allowed sectors
  old_allowed_sectors = intersect_sectors(
    broadcast.(GradedAxes.dual, old_domain_irreps), old_codomain_irreps
  )
  in_irreps = (old_domain_irreps..., old_codomain_irreps...)
  new_domain_irreps = getindex.(Ref(in_irreps), perm1)
  new_codomain_irreps = getindex.(Ref(in_irreps), perm2)
  old_domain_arrows = .!old_arrows[begin:OldNCoAxes]
  old_codomain_arrows = old_arrows[(OldNCoAxes + 1):end]
  new_domain_arrows = .!getindex.(Ref(old_arrows), perm1)
  new_codomain_arrows = getindex.(Ref(old_arrows), perm2)
  new_allowed_sectors = intersect_sectors(
    broadcast.(GradedAxes.dual, new_domain_irreps), new_codomain_irreps
  )

  # initialize output
  old_block_indices = Vector{NTuple{N,Int}}()
  old_domain_structural_multiplicities = Matrix{Int}(undef, length(old_allowed_sectors), 0)
  old_codomain_structural_multiplicities = Matrix{Int}(
    undef, length(old_allowed_sectors), 0
  )
  new_domain_structural_multiplicities = Matrix{Int}(undef, length(new_allowed_sectors), 0)
  new_codomain_structural_multiplicities = Matrix{Int}(
    undef, length(new_allowed_sectors), 0
  )

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

  # cache computed Clebsch-Gordan trees. Could compress new trees, not worth it.
  old_domain_trees = Dict{NTuple{OldNCoAxes,Int},Vector{Array{Float64,OldNCoAxes + 2}}}()
  old_codomain_trees = Dict{NTuple{OldNDoAxes,Int},Vector{Array{Float64,OldNDoAxes + 2}}}()
  new_domain_trees = Dict{NTuple{NCoAxesNew,Int},Vector{Array{Float64,NCoAxesNew + 2}}}()
  new_codomain_trees = Dict{NTuple{NDoAxesNew,Int},Vector{Array{Float64,NDoAxesNew + 2}}}()

  # loop over all sector configuration
  for it in Iterators.product(eachindex.(in_irreps)...)
    if !isempty(
      intersect_sectors(
        getindex.(old_domain_irreps, it[begin:OldNCoAxes]),
        getindex.(old_codomain_irreps, it[(OldNCoAxes + 1):end]),
      ),
    )
      old_domain_block_trees = get_tree!(
        old_domain_trees,
        it[begin:OldNCoAxes],
        old_domain_irreps,
        old_domain_arrows,
        old_allowed_sectors,
      )
      old_codomain_block_trees = get_tree!(
        old_codomain_trees,
        it[(OldNCoAxes + 1):end],
        old_codomain_irreps,
        old_codomain_arrows,
        old_allowed_sectors,
      )
      new_domain_block_trees = get_tree!(
        new_domain_trees,
        getindex.(Ref(it), perm1),
        new_domain_irreps,
        new_domain_arrows,
        new_allowed_sectors,
      )
      new_codomain_block_trees = get_tree!(
        new_codomain_trees,
        getindex.(Ref(it), perm2),
        new_codomain_irreps,
        new_codomain_arrows,
        new_allowed_sectors,
      )

      isometry = overlap_cg_trees(
        old_domain_block_trees,
        old_codomain_block_trees,
        new_domain_block_trees,
        new_codomain_block_trees,
        irreps_perm,
      )
      push!(old_block_indices, it)
      push!(isometries, isometry)
      # TODO update multiplicities
    end
  end
  return (
    old_block_indices,
    old_domain_structural_multiplicities,
    old_codomain_structural_multiplicities,
    new_domain_structural_multiplicities,
    new_codomain_structural_multiplicities,
    isometries,
  )
end

###################################  Constructor from 6j  ##################################
function compute_isometries_6j(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  old_codomain_sectors::NTuple{OldNCoAxes,Vector{C}},
  old_domain_sectors::NTuple{OldNDoAxes,Vector{C}},
  old_arrows::NTuple{N,Bool},
) where {N,OldNCoAxes,OldNDoAxes,C<:Sectors.AbstractCategory}
  # this function has the same inputs and the same outputs as compute_isometries_CG

  # dummy
end
