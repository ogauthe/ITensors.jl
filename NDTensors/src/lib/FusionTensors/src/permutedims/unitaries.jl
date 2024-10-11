# This file defines unitaries to be used in permutedims

# Current implementation:
# * Unitary = BlockMatrix
# * no unitary cache

# ===========================  Constructor from Clebsch-Gordan  ============================
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

  # some trees are empty: return projector on null space
  if length(domain_tree) == 0 || length(codomain_tree) == 0
    return zeros((irrep_dims_prod, 0))
  end

  labels_domain = (ntuple(identity, NCoAxes)..., N + 3, N + 1)
  labels_codomain = (ntuple(i -> i + NCoAxes, NDoAxes)..., N + 3, N + 2)
  labels_dest = (irreps_perm..., N + 1, N + 2)

  #          ----------------dim_sec---------
  #          |                              |
  #          |  struct_mult_domain_sec      |  struct_mult_codomain_sec
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
  unitary = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

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
    #           \  / _old_domain_sec_j         \  /
    #            \/                             \/
    #            /                              /
    #           /                              /
    #          /\                             /\
    #         /  \                           /  \
    #        /\   \                         /\   \
    #       /  \   \                       /  \   \
    #     dim1 dim2 dim3                 dim4 dim5 dim6
    #      |    |   |                      |   |    |
    #      ----------------- irreps_perm ------------
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
    #             |   _old_domain_sec_j         |
    #             |                             |
    #             -------------dim_sec_i---------
    #
    dim_sec_i = size(new_domain_block_trees[i], NCoAxesNew + 1)
    unitary[BlockArrays.Block(i, j)] = (new_projectors[i]'old_projectors[j]) / dim_sec_i
  end

  return unitary
end

function compute_unitaries_CG(
  old_domain_irreps::NTuple{OldNDoAxes,Vector{C}},
  old_codomain_irreps::NTuple{OldNCoAxes,Vector{C}},
  new_domain_irreps::NTuple{NewNDoAxes,Vector{C}},
  new_codomain_irreps::NTuple{NewNCoAxes,Vector{C}},
  old_arrows::NTuple{N,Bool},
  flat_permutation::NTuple{N,Int},
) where {N,OldNDoAxes,OldNCoAxes,NewNDoAxes,NewNCoAxes,C<:SymmetrySectors.AbstractSector}
  perm1 = flat_permutation[begin:NewNDoAxes]
  perm2 = flat_permutation[(NewNDoAxes + 1):end]

  # define axes, isdual and allowed sectors
  old_allowed_sectors = intersect_sectors(
    broadcast.(GradedAxes.dual, old_domain_irreps), old_codomain_irreps
  )
  in_irreps = (old_domain_irreps..., old_codomain_irreps...)
  old_domain_arrows = .!old_arrows[begin:OldNCoAxes]
  old_codomain_arrows = old_arrows[(OldNCoAxes + 1):end]
  new_domain_arrows = .!map(i -> old_arrows[i], perm1)
  new_codomain_arrows = map(i -> old_arrows[i], perm2)
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

  unitaries = Vector{
    BlockArrays.BlockMatrix{   # TBD use BlockSparseArray 4-dim?
      Float64,
      Matrix{Matrix{Float64}},
      Tuple{
        BlockArrays.BlockedUnitRange{Vector{Int64}},
        BlockArrays.BlockedUnitRange{Vector{Int64}},
      },
    },
  }()

  # cache computed Clebsch-Gordan trees.
  old_domain_trees_cache = Dict{
    NTuple{OldNCoAxes,Int},Vector{Array{Float64,OldNCoAxes + 2}}
  }()
  old_codomain_trees_cache = Dict{
    NTuple{OldNDoAxes,Int},Vector{Array{Float64,OldNDoAxes + 2}}
  }()
  new_domain_trees_cache = Dict{
    NTuple{NCoAxesNew,Int},Vector{Array{Float64,NewNDoAxes + 2}}
  }()
  new_codomain_trees_cache = Dict{
    NTuple{NDoAxesNew,Int},Vector{Array{Float64,NewNCoAxes + 2}}
  }()

  # loop over all sector configuration
  for it in Iterators.product(eachindex.(in_irreps)...)
    isempty(
      intersect_sectors(
        map(i -> old_domain_irreps[i], it[begin:OldNCoAxes]),
        map(i -> old_codomain_irreps[i], it[(OldNCoAxes + 1):end]),
      ),
    ) && continue
    old_domain_block_trees = get_tree!(
      old_domain_trees_cache,
      it[begin:OldNCoAxes],
      old_domain_irreps,
      old_domain_arrows,
      old_allowed_sectors,
    )
    old_codomain_block_trees = get_tree!(
      old_codomain_trees_cache,
      it[(OldNCoAxes + 1):end],
      old_codomain_irreps,
      old_codomain_arrows,
      old_allowed_sectors,
    )
    new_domain_block_trees = get_tree!(
      new_domain_trees_cache,
      map(i -> it[i], perm1),
      new_domain_irreps,
      new_domain_arrows,
      new_allowed_sectors,
    )
    new_codomain_block_trees = get_tree!(
      new_codomain_trees_cache,
      map(i -> it[i], perm2),
      new_codomain_irreps,
      new_codomain_arrows,
      new_allowed_sectors,
    )

    unitary = overlap_cg_trees(
      old_domain_block_trees,
      old_codomain_block_trees,
      new_domain_block_trees,
      new_codomain_block_trees,
      flat_permutation,
    )
    push!(old_block_indices, it)
    push!(unitaries, unitary)
    # TODO update multiplicities
  end

  return (
    old_block_indices,
    old_domain_structural_multiplicities,
    old_codomain_structural_multiplicities,
    new_domain_structural_multiplicities,
    new_codomain_structural_multiplicities,
    unitaries,
  )
end

# =================================  Constructor from 6j  ==================================
function compute_unitaries_6j(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  old_codomain_sectors::NTuple{OldNCoAxes,Vector{C}},
  old_domain_sectors::NTuple{OldNDoAxes,Vector{C}},
  old_arrows::NTuple{N,Bool},
) where {N,OldNCoAxes,OldNDoAxes,C<:SymmetrySectors.AbstractSector}
  # this function has the same inputs and the same outputs as compute_unitaries_CG

  # dummy
end
