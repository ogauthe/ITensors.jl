# This file defines unitaries to be used in permutedims

# Current implementation:
# * Unitary = BlockMatrix
# * no unitary cache

# ======================================  Interface  =======================================
function compute_unitaries(
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  flat_permutation::Tuple{Vararg{Int}},
)
  @assert length(old_domain_legs) + length(old_codomain_legs) == length(flat_permutation)
  @assert length(new_domain_legs) + length(new_codomain_legs) == length(flat_permutation)
  return compute_unitaries_clebsch_gordan(
    old_domain_legs, old_codomain_legs, new_domain_legs, new_codomain_legs, flat_permutation
  )
end

# ===========================  Constructor from Clebsch-Gordan  ============================
function contract_singlet_space_projector(
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

function overlap_fusion_trees(
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
  # blocksizes: (struct_mult_new_block_sectors, struct_mult_old_block_sectors)
  block_rows =
    size.(new_domain_block_trees, NCoAxesNew + 2) .*
    size.(new_codomain_block_trees, NDoAxesNew + 2)
  block_cols =
    size.(old_domain_block_trees, OldNCoAxes + 2) .*
    size.(old_codomain_block_trees, OldNDoAxes + 2)
  unitary = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  # contract domain and codomain fusion trees to construct projector on each allowed sector
  new_projectors =
    contract_singlet_space_projector.(
      new_domain_block_trees, new_codomain_block_trees, Ref(ntuple(identity, N))
    )
  old_projectors =
    contract_singlet_space_projector.(
      old_domain_block_trees, old_codomain_block_trees, Ref(irreps_perm)
    )

  for (i, j) in Iterators.product(eachindex.((new_projectors, old_projectors))...)
    # Contract new and old projectors to construct singlet space basis change matrix.
    # Construction is blockwise. One block (i,j) corresponds to fusing old axes
    # over sector sec_j and new axes over sector sec_i. It has a 4-dim tensor
    # internal structure which is reshaped into a matrix.
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

function compute_unitaries_clebsch_gordan(
  old_domain_legs::NTuple{OldNDoAxes,AbstractUnitRange},
  old_codomain_legs::NTuple{OldNCoAxes,AbstractUnitRange},
  new_domain_legs::NTuple{NewNDoAxes,AbstractUnitRange},
  new_codomain_legs::NTuple{NewNCoAxes,AbstractUnitRange},
  flat_permutation::NTuple{N,Int},
) where {N,OldNDoAxes,OldNCoAxes,NewNDoAxes,NewNCoAxes}
  perm1 = flat_permutation[begin:NewNDoAxes]
  perm2 = flat_permutation[(NewNDoAxes + 1):end]

  new_row_labels = GradedAxes.blocklabels(GradedAxes.fusion_product(new_domain_legs...))
  new_column_labels = GradedAxes.blocklabels(
    GradedAxes.fusion_product(new_codomain_legs...)
  )
  new_allowed_sectors = new_row_labels[first.(
    find_shared_indices(new_row_labels, new_column_labels)
  )]

  # TBD use FusedAxes as input?
  old_domain_fused_axes = FusedAxes(old_domain_legs)
  old_codomain_fused_axes = FusedAxes(GradedAxes.dual.(old_codomain_legs))
  old_matrix_block_indices = intersect(old_domain_fused_axes, old_codomain_fused_axes)
  old_allowed_sectors = GradedAxes.blocklabels(old_domain_fused_axes)[first.(
    old_matrix_block_indices
  )]
  old_allowed_outer_blocks = allowed_outer_blocks_sectors(
    old_domain_fused_axes, old_codomain_fused_axes, old_matrix_block_indices
  )

  # initialize output
  unitaries = Dict{
    NTuple{N,Int64},
    BlockArrays.BlockMatrix{   # TBD use BlockSparseArray 4-dim?
      Float64,
      Matrix{Matrix{Float64}},
      Tuple{
        BlockArrays.BlockedOneTo{Int64,Vector{Int64}},
        BlockArrays.BlockedOneTo{Int64,Vector{Int64}},
      },
    },
  }()

  # cache computed Clebsch-Gordan trees.
  old_domain_trees_cache = Dict{
    NTuple{OldNDoAxes,Int},Vector{Array{Float64,OldNDoAxes + 2}}
  }()
  old_codomain_trees_cache = Dict{
    NTuple{OldNCoAxes,Int},Vector{Array{Float64,OldNCoAxes + 2}}
  }()
  new_domain_trees_cache = Dict{
    NTuple{NewNDoAxes,Int},Vector{Array{Float64,NewNDoAxes + 2}}
  }()
  new_codomain_trees_cache = Dict{
    NTuple{NewNCoAxes,Int},Vector{Array{Float64,NewNCoAxes + 2}}
  }()

  # loop over all allowed outer blocks.
  for (it, _) in old_allowed_outer_blocks
    old_domain_block_trees = get_tree!(
      old_domain_trees_cache, it[begin:OldNDoAxes], old_domain_legs, old_allowed_sectors
    )
    old_codomain_block_trees = get_tree!(
      old_codomain_trees_cache,
      it[(OldNDoAxes + 1):end],
      old_codomain_legs,
      old_allowed_sectors,
    )
    new_domain_block_trees = get_tree!(
      new_domain_trees_cache, map(i -> it[i], perm1), new_domain_legs, new_allowed_sectors
    )
    new_codomain_block_trees = get_tree!(
      new_codomain_trees_cache,
      map(i -> it[i], perm2),
      new_codomain_legs,
      new_allowed_sectors,
    )

    u = overlap_fusion_trees(
      old_domain_block_trees,
      old_codomain_block_trees,
      new_domain_block_trees,
      new_codomain_block_trees,
      flat_permutation,
    )
    unitaries[it] = u
  end

  return unitaries
end

# =================================  Constructor from 6j  ==================================
# dummy
