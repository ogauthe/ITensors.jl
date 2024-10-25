# This file defines unitaries to be used in permutedims

# Current implementation:
# * Unitary = BlockMatrix
# * no unitary cache

# Notes:
# - The interface uses AbstractGradedUnitRanges as input for interface simplicity
#   however only blocklabels are used and blocklengths are never used.

# ======================================  Interface  =======================================
function compute_unitaries(
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  biperm::TensorAlgebra.BlockedPermutation{2},
)
  @assert length(old_domain_legs) + length(old_codomain_legs) == length(biperm)
  return compute_unitaries_clebsch_gordan(old_domain_legs, old_codomain_legs, biperm)
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

function intersect_trees(
  domain_trees::Vector{<:AbstractArray{<:Real}},
  codomain_trees::Vector{<:AbstractArray{<:Real}},
)
  kept_indices = findall(.!isempty.(domain_trees) .* .!isempty.(codomain_trees))
  return domain_trees[kept_indices] .=> codomain_trees[kept_indices]
end

function overlap_fusion_trees(
  old_domain_block_trees::Vector{<:AbstractArray{<:Real}},
  old_codomain_block_trees::Vector{<:AbstractArray{<:Real}},
  new_domain_block_trees::Vector{<:AbstractArray{<:Real}},
  new_codomain_block_trees::Vector{<:AbstractArray{<:Real}},
  irreps_perm::NTuple{N,Int},
) where {N}
  # filter trees that do not share the same sectors
  old_trees = intersect_trees(old_domain_block_trees, old_codomain_block_trees)
  new_trees = intersect_trees(new_domain_block_trees, new_codomain_block_trees)
  return overlap_filtered_fusion_trees(old_trees, new_trees, irreps_perm)
end

function overlap_filtered_fusion_trees(
  old_trees::Vector{<:Pair{<:AbstractArray{<:Real},<:AbstractArray{<:Real}}},
  new_trees::Vector{<:Pair{<:AbstractArray{<:Real},<:AbstractArray{<:Real}}},
  irreps_perm::NTuple{N,Int},
) where {N}
  old_domain_block_trees = first.(old_trees)
  old_codomain_block_trees = last.(old_trees)
  new_domain_block_trees = first.(new_trees)
  new_codomain_block_trees = last.(new_trees)
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

  for j in eachindex(block_rows)
    old_proj = contract_singlet_space_projector(
      old_domain_block_trees[j], old_codomain_block_trees[j], irreps_perm
    )
    for (i, new_proj) in enumerate(new_projectors)

      # Contract new and old projectors to construct singlet space basis change matrix.
      # Construction is blockwise. One block (i,j) corresponds to fusing old axes
      # over sector sec_j and new axes over sector sec_i. It has a 4-dim tensor
      # internal structure which is reshaped into a matrix.
      #
      #         --------------dim_sec_j---------
      #         |                              |
      #         |   struct_mult                |   struct_mult_old_codomain_sec_j
      #          \  / _old_domain_sec_j         \  /
      #           \/                             \/
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
      #              /\                            /\
      #             /  \                          /  \
      #            | struct_mult                 |  struct_mult_new_codomain_sec_i
      #            |   _old_domain_sec_j         |
      #            |                             |
      #            -------------dim_sec_i---------
      #
      dim_sec_i = size(new_domain_block_trees[i], NCoAxesNew + 1)
      unitary[BlockArrays.Block(i, j)] = (new_proj'old_proj) / dim_sec_i
    end
  end

  return unitary
end

function compute_unitaries_clebsch_gordan(
  old_domain_legs::NTuple{OldNDoAxes,AbstractUnitRange},
  old_codomain_legs::NTuple{OldNCoAxes,AbstractUnitRange},
  biperm::TensorAlgebra.BlockedPermutation{2,N},
) where {OldNDoAxes,OldNCoAxes,N}
  @assert OldNDoAxes + OldNCoAxes == N

  new_domain_legs, nondual_new_codomain_legs = TensorAlgebra.blockpermute(
    (old_domain_legs..., old_codomain_legs...), biperm
  )
  new_codomain_legs = GradedAxes.dual.(nondual_new_codomain_legs)
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
    NTuple{length(new_domain_legs),Int},Vector{Array{Float64,length(new_domain_legs) + 2}}
  }()
  new_codomain_trees_cache = Dict{
    NTuple{length(new_codomain_legs),Int},
    Vector{Array{Float64,length(new_codomain_legs) + 2}},
  }()

  flat_permutation = Tuple(biperm)

  # loop over all allowed outer blocks.
  for (old_outer_block, _) in old_allowed_outer_blocks
    new_domain_outer_block, new_codomain_outer_block = TensorAlgebra.blockpermute(
      old_outer_block, biperm
    )
    old_domain_block_trees = get_tree!(
      old_domain_trees_cache,
      old_outer_block[begin:OldNDoAxes],
      old_domain_legs,
      old_allowed_sectors,
    )
    old_codomain_block_trees = get_tree!(
      old_codomain_trees_cache,
      old_outer_block[(OldNDoAxes + 1):end],
      GradedAxes.dual.(old_codomain_legs),
      old_allowed_sectors,
    )
    new_domain_block_trees = get_tree!(
      new_domain_trees_cache, new_domain_outer_block, new_domain_legs, new_allowed_sectors
    )
    new_codomain_block_trees = get_tree!(
      new_codomain_trees_cache,
      new_codomain_outer_block,
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
    unitaries[old_outer_block] = u
  end

  return unitaries
end

# =================================  Constructor from 6j  ==================================
# dummy
