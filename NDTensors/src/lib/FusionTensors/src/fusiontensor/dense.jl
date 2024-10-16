# This file defines interface to cast from and to dense array

# =================================  High level interface  =================================

#### cast from dense to symmetric
function FusionTensor(dense::AbstractArray, domain_legs::Tuple, codomain_legs::Tuple)
  bounds = SymmetrySectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockArrays.BlockedArray(dense, bounds...)
  return FusionTensor(blockarray, domain_legs, codomain_legs)
end

function FusionTensor(
  blockarray::BlockArrays.AbstractBlockArray, domain_legs::Tuple, codomain_legs::Tuple
)
  data_mat = cast_from_dense(blockarray, domain_legs, codomain_legs)
  return FusionTensor(data_mat, domain_legs, codomain_legs)
end

#### cast from symmetric to dense
function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  return cast_to_dense(ft)
end

# =================================  Low level interface  ==================================
function cast_from_dense(
  blockarray::BlockArrays.AbstractBlockArray, domain_legs::Tuple, codomain_legs::Tuple
)
  # input validation
  if length(domain_legs) + length(codomain_legs) != ndims(blockarray)  # compile time
    throw(DomainError("legs are incompatible with array ndims"))
  end
  if SymmetrySectors.quantum_dimension.((domain_legs..., codomain_legs...)) !=
    size(blockarray)
    throw(DomainError("legs dimensions are incompatible with array"))
  end

  data_mat = initialize_data_matrix(eltype(blockarray), domain_legs, codomain_legs)
  fill_matrix_blocks!(data_mat, blockarray, domain_legs, codomain_legs)
  return data_mat
end

function cast_to_dense(ft::FusionTensor)
  return cast_to_dense(data_matrix(ft), domain_axes(ft), codomain_axes(ft))
end

function cast_to_dense(data_mat::AbstractMatrix, domain_legs::Tuple, codomain_legs::Tuple)
  bounds = SymmetrySectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockSparseArrays.BlockSparseArray{eltype(data_mat)}(
    BlockArrays.blockedrange.(bounds)
  )
  fill_blockarray!(blockarray, data_mat, domain_legs, codomain_legs)
  return blockarray
end

# =====================================  Internals  ========================================

##################################  utility tools  #########################################

function find_allowed_blocks(data_mat::BlockSparseArrays.AbstractBlockSparseMatrix)
  row_axis, col_axis = axes(data_mat)
  col_sectors = GradedAxes.blocklabels(col_axis)
  dual_row_sectors = GradedAxes.blocklabels(GradedAxes.dual(row_axis))
  allowed_sectors = intersect_sectors(dual_row_sectors, col_sectors)
  col_shared_indices = findall(in(allowed_sectors), col_sectors)
  row_shared_indices = findall(in(allowed_sectors), dual_row_sectors)
  allowed_blocks = BlockArrays.Block.(row_shared_indices, col_shared_indices)
  allowed_matrix_blocks = [BlockSparseArrays.view!(data_mat, b) for b in allowed_blocks]
  return allowed_sectors, allowed_matrix_blocks
end

function find_existing_blocks(data_mat::AbstractMatrix)
  col_sectors = GradedAxes.blocklabels(axes(data_mat, 2))
  existing_blocks = sort(collect(BlockSparseArrays.block_stored_indices(data_mat)))
  existing_sectors = [col_sectors[Int(Tuple(b)[2])] for b in existing_blocks]
  existing_matrix_blocks = [view(data_mat, b) for b in existing_blocks]
  return existing_sectors, existing_matrix_blocks
end

function split_axes(legs::Tuple)
  arrows = GradedAxes.isdual.(legs)
  irreps = GradedAxes.blocklabels.(legs)
  degens = BlockArrays.blocklengths.(legs)
  dimensions = broadcast.(SymmetrySectors.quantum_dimension, irreps)
  return arrows, irreps, degens, dimensions
end

function split_degen_dims(
  dense_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  dense_block_split_shape = (
    braid_tuples(domain_block_degens, domain_block_dims)...,
    braid_tuples(codomain_block_degens, codomain_block_dims)...,
  )
  split_dense_block = reshape(dense_block, dense_block_split_shape)
  return split_dense_block
end

function merge_degen_dims(split_dense_block::AbstractArray)
  s0 = size(split_dense_block)
  dense_shape =
    ntuple(i -> s0[2 * i - 1], length(s0) ÷ 2) .* ntuple(i -> s0[2 * i], length(s0) ÷ 2)
  dense_block = reshape(split_dense_block, dense_shape)
  return dense_block
end

function permute_split_dense_block(split_dense_block::AbstractArray)
  N = ndims(split_dense_block) ÷ 2
  dense_data_perm = (ntuple(i -> 2 * i - 1, N)..., ntuple(i -> 2 * i, N)...)
  permuted_split_dense_block = permutedims(split_dense_block, dense_data_perm)
  return permuted_split_dense_block
end

function unpermute_split_dense_block(permuted_split_dense_block::AbstractArray)
  twoN = ndims(permuted_split_dense_block)
  N = twoN ÷ 2
  inverse_dense_data_perm = ntuple(i -> fld1(i, 2) + (1 - i % 2) * N, twoN)
  split_dense_block = permutedims(permuted_split_dense_block, inverse_dense_data_perm)
  return split_dense_block
end

function reshape_permuted_to_fused(
  permuted_split_dense_block::AbstractArray, ::Val{N_CO}
) where {N_CO}
  N = ndims(permuted_split_dense_block) ÷ 2
  permuted_dense_shape = size(permuted_split_dense_block)
  fused_dense_block_shape = (
    prod(permuted_dense_shape[begin:N_CO]),
    prod(permuted_dense_shape[(N_CO + 1):N]),
    prod(permuted_dense_shape[(N + 1):(N + N_CO)]),
    prod(permuted_dense_shape[(N + N_CO + 1):end]),
  )
  fused_dense_block = reshape(permuted_split_dense_block, fused_dense_block_shape)
  return fused_dense_block
end

function reshape_fused_to_permuted(
  fused_dense_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  degen_dim_shape = (
    domain_block_degens...,
    codomain_block_degens...,
    domain_block_dims...,
    codomain_block_dims...,
  )
  permuted_split_dense_block = reshape(fused_dense_block, degen_dim_shape)
  return permuted_split_dense_block
end

function fuse_dense_block(
  dense_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  # start from a dense outer block with e.g. N=6 axes divided into N_DO=3 ndims_domain
  # and N_CO=3 ndims_codomain. Each leg k can be decomposed as a product of external an
  # multiplicity extk and a quantum dimension dimk
  #
  #        ------------------------------dense_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #

  # each leg of this this dense outer block can now be opened to form a 2N-dims tensor.
  # note that this 2N-dims form is only defined at the level of the outer block,
  # not for a larger block.
  #
  #        ------------------------------split_dense_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #
  split_dense_block = split_degen_dims(
    dense_block,
    domain_block_degens,
    domain_block_dims,
    codomain_block_degens,
    codomain_block_dims,
  )

  # Now we permute the axes to group together degenearacies on one side and irrep
  # dimensions on the other side. This is the bottleneck.
  #
  #     -------------------permuted_split_dense_block-----------------------------------
  #     |      |       |       |        |      |      |      |      |      |     |     |
  #   ext1   ext2    ext3    ext4     ext5   ext6    dim1   dim2   dim3   dim4  dim5  dim6
  #
  permuted_split_dense_block = permute_split_dense_block(split_dense_block)

  # Finally, it is convenient to merge together legs corresponding to domain or
  # to codomain and produce a 4-dims tensor
  #
  #        ---------------------fused_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  fused_dense_block = reshape_permuted_to_fused(
    permuted_split_dense_block, Val(length(domain_block_degens))
  )
  return fused_dense_block
end

function contract_fusion_trees(
  fused_dense_block::AbstractArray{<:Number,4},
  tree_domain::AbstractArray{<:Real,3},
  tree_codomain::AbstractArray{<:Real,3},
)
  # Input:
  #
  #        ---------------------fused_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  #
  #
  #         ---------------tree_domain------------
  #         |                      |             |
  #        dim1*dim2*dim3        dim_sec    struct_sec_domain
  #
  #
  #         ----------------tree_codomain-----------
  #         |                      |               |
  #        dim4*dim5*dim6         dim_sec     struct_sec_codomain
  #
  # in this form, we can apply fusion trees on both the codomain and the domain.
  #

  # contract codomain tree
  #           -------------------------data_1tree---------------------------
  #           |               |                 |             |            |
  #     ext1*ext2*ext3   ext4*ext5*ext6   dim1*dim2*dim3    dim_sec    struct_sec_codomain
  #
  data_1tree = TensorAlgebra.contract(
    (1, 2, 3, 5, 6), fused_dense_block, (1, 2, 3, 4), tree_codomain, (4, 5, 6)
  )

  # contract domain tree
  #             -----------------------sym_data----------------------------
  #             |                  |                    |                 |
  #       ext1*ext2*ext3    struct_sec_codomain   ext4*ext5*ext6   struct_sec_domain
  #
  T = promote_type(eltype(fused_dense_block), Float64)
  dim_sec = size(tree_domain, 2)
  sym_data::Array{T,4} = TensorAlgebra.contract(
    (1, 7, 2, 6),   # HERE WE SET INNER STRUCTURE FOR MATRIX BLOCKS
    data_1tree,
    (1, 2, 3, 5, 6),
    tree_domain,
    (3, 5, 7),
    1 / dim_sec,  # normalization factor
  )

  #             ----------------------sym_block_sec---------------
  #             |                                                |
  #       ext1*ext2*ext3*struct_sec_codomain   ext4*ext5*ext6*struct_sec_domain
  #
  sym_shape = (size(sym_data, 1) * size(sym_data, 2), size(sym_data, 3) * size(sym_data, 4))
  sym_block_sec = reshape(sym_data, sym_shape)
  return sym_block_sec
end

#################################  cast from dense array  ##################################

function fill_matrix_blocks!(
  data_mat::BlockSparseArrays.AbstractBlockSparseMatrix,
  blockarray::BlockArrays.AbstractBlockArray,
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  # find sectors
  allowed_sectors, allowed_matrix_blocks = find_allowed_blocks(data_mat)

  # domain needs to be dualed in fusion tree
  domain_arrows, domain_irreps, domain_degens, domain_dims = split_axes(
    GradedAxes.dual.(domain_legs)
  )
  codomain_arrows, codomain_irreps, codomain_degens, codomain_dims = split_axes(
    codomain_legs
  )

  # cache computed trees
  domain_trees_cache = Dict{NTuple{length(domain_legs),Int},Vector{Array{Float64,3}}}()

  # precompute internal structure
  # TODO cache FusedAxes inside FusionTensor
  domain_fused_axes = FusedAxes(domain_legs)
  codomain_fused_axes = FusedAxes(codomain_legs)

  # Below, we loop over every allowed outer block, contract domain and codomain fusion trees
  # for each allowed sector and write the result inside a symmetric matrix block
  #
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
  #      |    |    |                    |    |    |
  #      ------------------dense_block-------------
  #      |    |    |                    |    |    |
  #     ext1 ext2 ext3                 ext4 ext5 ext6

  # loop for each codomain irrep configuration
  for iter_co in codomain_fused_axes
    codomain_block_irreps = getindex.(codomain_irreps, iter_co)
    codomain_block_allowed_sectors = intersect_sectors(
      codomain_block_irreps, allowed_sectors
    )
    isempty(codomain_block_allowed_sectors) && continue
    codomain_block_trees = compute_pruned_leavesmerged_fusion_trees(
      codomain_block_irreps, codomain_arrows, allowed_sectors
    )

    # loop for each domain irrep configuration
    for iter_do in domain_fused_axes
      block_allowed_sectors = intersect_sectors(
        getindex.(domain_irreps, iter_do), codomain_block_allowed_sectors
      )
      isempty(block_allowed_sectors) && continue
      domain_block_trees = get_tree!(
        domain_trees_cache, iter_do, domain_irreps, domain_arrows, allowed_sectors
      )
      fused_dense_block = fuse_dense_block(
        view(blockarray, BlockArrays.Block(iter_do..., iter_co...)),
        getindex.(domain_degens, iter_do),
        getindex.(domain_dims, iter_do),
        getindex.(codomain_degens, iter_co),
        getindex.(codomain_dims, iter_co),
      )

      # loop for each symmetry sector allowed in this configuration
      for i_sec in findall(in(block_allowed_sectors), allowed_sectors)

        # actual implementation: legs are conveniently merged
        #
        #          ----------------dim_sec---------
        #          |                              |
        #          |  struct_mult_domain_sec      |  struct_mult_codomain_sec
        #           \  /                           \  /
        #            \/                             \/
        #            /                              /
        #           |                               |
        #     dim1*dim2*dim3                 dim4*dim5*dim6
        #           |                               |
        #           ------------fused_dense_block----
        #           |                               |
        #     ext1*ext2*ext3                 ext4*ext5*ext6

        # contract fusion trees and reshape symmetric block as a matrix
        # Note: a final permutedims is needed after the last contract
        # therefore cannot efficiently use contract!(allowed_matrix_blocks[...], ...)
        # TBD something like permutedims!(reshape(view), sym_block, (1,3,2,4))?
        sym_block_sec = contract_fusion_trees(
          fused_dense_block, domain_block_trees[i_sec], codomain_block_trees[i_sec]
        )

        # find outer block location inside this matrix block && write it
        row_range = find_block_range(
          domain_fused_axes, iter_do, GradedAxes.dual(allowed_sectors[i_sec])
        )
        col_range = find_block_range(codomain_fused_axes, iter_co, allowed_sectors[i_sec])
        @views allowed_matrix_blocks[i_sec][row_range, col_range] = sym_block_sec
      end
    end
  end
end

##################################  cast to dense array  ###################################
function add_sector_block!(
  fused_dense_block::AbstractArray{<:Number,4},
  sym_block_sec::AbstractMatrix,
  tree_domain::AbstractArray{<:Real,3},
  tree_codomain::AbstractArray{<:Real,3},
)
  domain_block_struct_sector = size(tree_domain, 3)
  codomain_block_struct_sector = size(tree_codomain, 3)
  #             ----------------------sym_block_sec---------------
  #             |                                                |
  #       ext1*ext2*ext3*struct_sec_codomain   ext4*ext5*ext6*struct_sec_domain
  #
  sym_data_shape = (
    size(sym_block_sec, 1) ÷ domain_block_struct_sector,
    domain_block_struct_sector,
    size(sym_block_sec, 2) ÷ codomain_block_struct_sector,
    codomain_block_struct_sector,
  )

  #             -----------------------sym_data----------------------------
  #             |                  |                    |                 |
  #       ext1*ext2*ext3    struct_sec_codomain   ext4*ext5*ext6   struct_sec_domain
  #
  sym_data = reshape(sym_block_sec, sym_data_shape)

  # contract domain tree
  #            -----------------------------data_1tree------------------------------
  #            |               |                    |              |               |
  #      ext1*ext2*ext3   ext4*ext5*ext6    struct_sec_domain  dim1*dim2*dim3   dim_sec
  #
  data_1tree = TensorAlgebra.contract(
    (1, 2, 6, 3, 5), sym_data, (1, 7, 2, 6), tree_domain, (3, 5, 7)
  )

  # contract codomain tree
  #        ---------------------fused_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  return TensorAlgebra.contract!(
    fused_dense_block,
    (1, 2, 3, 4),
    data_1tree,
    (1, 2, 6, 3, 5),
    tree_codomain,
    (4, 5, 6),
    1.0,
    1.0,
  )
end

function unfuse_dense_block(
  fused_dense_block::AbstractArray{<:Number,4},
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  #        ---------------------fused_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #

  #     -------------------permuted_split_dense_block-----------------------------------
  #     |      |       |       |        |      |      |      |      |      |     |     |
  #   ext1   ext2    ext3    ext4     ext5   ext6    dim1   dim2   dim3   dim4  dim5  dim6
  #
  permuted_split_dense_block = reshape_fused_to_permuted(
    fused_dense_block,
    domain_block_degens,
    domain_block_dims,
    codomain_block_degens,
    codomain_block_dims,
  )

  #        ------------------------------split_dense_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #
  split_dense_block = unpermute_split_dense_block(permuted_split_dense_block)

  #
  #        ------------------------------dense_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #
  dense_block = merge_degen_dims(split_dense_block)
  return dense_block
end

function fill_blockarray!(
  blockarray::BlockArrays.AbstractBlockArray,
  data_mat::AbstractMatrix,
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  existing_sectors, existing_matrix_blocks = find_existing_blocks(data_mat)

  # domain needs to be dualed in fusion tree
  domain_arrows, domain_irreps, domain_degens, domain_dims = split_axes(
    GradedAxes.dual.(domain_legs)
  )
  codomain_arrows, codomain_irreps, codomain_degens, codomain_dims = split_axes(
    codomain_legs
  )

  # TODO cache FusedAxes inside FusionTensor
  domain_fused_axes = FusedAxes(domain_legs)
  codomain_fused_axes = FusedAxes(codomain_legs)

  # cache computed trees
  domain_trees_cache = Dict{NTuple{length(domain_legs),Int},Vector{Array{Float64,3}}}()

  # loop for each codomain irrep configuration
  for iter_co in codomain_fused_axes
    codomain_block_irreps = getindex.(codomain_irreps, iter_co)
    codomain_block_existing_sectors = intersect_sectors(
      codomain_block_irreps, existing_sectors
    )
    isempty(codomain_block_existing_sectors) && continue
    codomain_block_trees = compute_pruned_leavesmerged_fusion_trees(
      codomain_block_irreps, codomain_arrows, existing_sectors
    )
    codomain_block_length = prod(getindex.(codomain_degens, iter_co))
    codomain_block_dims = getindex.(codomain_dims, iter_co)

    # loop for each domain irrep configuration
    for iter_do in domain_fused_axes
      block_existing_sectors = intersect_sectors(
        getindex.(domain_irreps, iter_do), codomain_block_existing_sectors
      )
      isempty(block_existing_sectors) && continue
      domain_block_trees = get_tree!(
        domain_trees_cache, iter_do, domain_irreps, domain_arrows, existing_sectors
      )
      domain_block_length = prod(getindex.(domain_degens, iter_do))
      domain_block_dims = getindex.(domain_dims, iter_do)

      #        ---------------------fused_dense_block--------------------
      #        |                   |                 |                  |
      #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
      #
      fused_dense_block_shape = (
        domain_block_length,
        codomain_block_length,
        prod(domain_block_dims),
        prod(codomain_block_dims),
      )
      fused_dense_block = zeros(eltype(blockarray), fused_dense_block_shape)

      # loop for each symmetry sector inside this configuration
      for i_sec in findall(in(block_existing_sectors), existing_sectors)
        row_range = find_block_range(
          domain_fused_axes, iter_do, GradedAxes.dual(existing_sectors[i_sec])
        )
        col_range = find_block_range(codomain_fused_axes, iter_co, existing_sectors[i_sec])
        sym_block_sec = view(existing_matrix_blocks[i_sec], row_range, col_range)
        add_sector_block!(
          fused_dense_block,
          sym_block_sec,
          domain_block_trees[i_sec],
          codomain_block_trees[i_sec],
        )
      end

      blockarray[BlockArrays.Block(iter_do..., iter_co...)] = unfuse_dense_block(
        fused_dense_block,
        getindex.(domain_degens, iter_do),
        domain_block_dims,
        getindex.(codomain_degens, iter_co),
        codomain_block_dims,
      )
    end
  end
end
