# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD: Unitary format
#      * BlockMatrix?
#      * 4-dim BlockSparseArray?
#      * other?

# TBD: cache format
#       * global Dict of Dict{(N,C,OldNDo,OldNCo,NewNDo,NewNCo,OldArrows,flatperm), Dict}
#          + unitary Dict{NTuple{C<:AbstractCategory}, Unitary}

# TBD: inner structure of a matrix block
#       * (struct, ext) or its transpose

# TBD: cache of FusionTensor inner structure
#       * cache in FT (TensorKit choice)
#       * cache in StructuralData  (froSTspin choice)
#       * no cache

# Current implementation:
# * Unitary = BlockMatrix
# * no unitary cache
# * inner structure = (struct, ext)
# * no cache of internal structure
#

struct StructuralData{N,C,OldNDo,OldNCo,NewNDo,NewNCo,Unitaries}
  old_domain_labels::NTuple{OldNDo,Vector{C}}
  old_codomain_labels::Tuple{OldNCo,Vector{C}}
  new_domain_labels::Tuple{NewNDo,Vector{C}}
  new_codomain_labels::Tuple{NewNCo,Vector{C}}
  old_arrows::NTuple{N,Bool}
  flat_permutation::NTuple{N,Int}
  unitaries::Unitaries

  function StructuralData(
    old_domain_labels::Tuple{Vararg{Vector{C}}},
    old_codomain_labels::Tuple{Vararg{Vector{C}}},
    new_domain_labels::Tuple{Vararg{Vector{C}}},
    new_codomain_labels::Tuple{Vararg{Vector{C}}},
    old_arrows::NTuple{N,Bool},
    flat_permutation::NTuple{N,Int},
    unitaries,
  ) where {N,C<:Sectors.AbstractCategory}
    @assert length(old_domain_labels) + length(old_codomain_labels) == N
    @assert length(new_domain_labels) + length(new_codomain_labels_codomain_labels) == N
    @assert N > 0

    return new{
      N,
      C,
      length(old_domain_labels),
      length(old_codomain_labels),
      length(new_domain_labels),
      length(new_codomain_labels),
      eltype(unitaries),
    }(
      old_domain_labels,
      old_codomain_labels,
      new_domain_labels,
      new_codomain_labels,
      old_arrows,
      flat_permutation,
      unitaries,
    )
  end
end

function StructuralData(
  old_domain_labels::Tuple{Vararg{Vector{C}}},
  old_codomain_labels::Tuple{Vararg{Vector{C}}},
  new_domain_labels::Tuple{Vararg{Vector{C}}},
  new_codomain_labels::Tuple{Vararg{Vector{C}}},
  old_arrows::NTuple{N,Bool},
  flat_permutation::NTuple{N,Int},
) where {N,C<:Sectors.AbstractCategory}
  unitaries = compute_unitaries_CG(
    old_domain_labels,
    old_codomain_labels,
    new_domain_labels,
    new_codomain_labels,
    old_arrows,
    flat_permutation,
  )
  return StructuralData(
    old_domain_labels,
    old_codomain_labels,
    new_domain_labels,
    new_codomain_labels,
    old_arrows,
    flat_permutation,
    unitaries,
  )
end

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
) where {N,OldNDoAxes,OldNCoAxes,NewNDoAxes,NewNCoAxes,C<:Sectors.AbstractCategory}
  perm1 = flat_permutation[begin:NewNDoAxes]
  perm2 = flat_permutation[(NewNDoAxes + 1):end]

  # define axes, isdual and allowed sectors
  old_allowed_sectors = intersect_sectors(
    broadcast.(GradedAxes.dual, old_domain_irreps), old_codomain_irreps
  )
  in_irreps = (old_domain_irreps..., old_codomain_irreps...)
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

  # cache computed Clebsch-Gordan trees. Could compress new trees, not worth it.
  old_domain_trees = Dict{NTuple{OldNCoAxes,Int},Vector{Array{Float64,OldNCoAxes + 2}}}()
  old_codomain_trees = Dict{NTuple{OldNDoAxes,Int},Vector{Array{Float64,OldNDoAxes + 2}}}()
  new_domain_trees = Dict{NTuple{NCoAxesNew,Int},Vector{Array{Float64,NewNDoAxes + 2}}}()
  new_codomain_trees = Dict{NTuple{NDoAxesNew,Int},Vector{Array{Float64,NewNCoAxes + 2}}}()

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
) where {N,OldNCoAxes,OldNDoAxes,C<:Sectors.AbstractCategory}
  # this function has the same inputs and the same outputs as compute_unitaries_CG

  # dummy
end
