# This file defines interface to cast from and to generic array

# =================================  High level interface  =================================

#### cast from array to symmetric
function FusionTensor(
  array::AbstractArray,
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  data_mat = cast_from_array(array, domain_legs, codomain_legs)
  return FusionTensor(data_mat, domain_legs, codomain_legs)
end

#### cast from symmetric to array
function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  return cast_to_array(ft)
end

# =================================  Low level interface  ==================================
function cast_from_array(
  array::AbstractArray,
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  bounds = SymmetrySectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockArrays.BlockedArray(array, bounds...)
  return cast_from_array(blockarray, domain_legs, codomain_legs)
end

function cast_from_array(
  blockarray::BlockArrays.AbstractBlockArray,
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  # input validation
  if length(domain_legs) + length(codomain_legs) != ndims(blockarray)  # compile time
    throw(DomainError("legs are incompatible with array ndims"))
  end
  if SymmetrySectors.quantum_dimension.((domain_legs..., codomain_legs...)) !=
    size(blockarray)
    throw(DomainError("legs dimensions are incompatible with array"))
  end

  # precompute internal structure
  # TODO cache FusedAxes inside FusionTensor
  domain_fused_axes = FusedAxes(domain_legs)
  codomain_fused_axes = FusedAxes(GradedAxes.dual.(codomain_legs))
  data_mat = initialize_data_matrix(
    eltype(blockarray), domain_fused_axes, codomain_fused_axes
  )

  fill_matrix_blocks!(data_mat, blockarray, domain_fused_axes, codomain_fused_axes)
  return data_mat
end

function cast_to_array(ft::FusionTensor)
  return cast_to_array(data_matrix(ft), domain_axes(ft), codomain_axes(ft))
end

function cast_to_array(
  data_mat::AbstractMatrix,
  domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
)
  bounds = SymmetrySectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockSparseArrays.BlockSparseArray{eltype(data_mat)}(
    BlockArrays.blockedrange.(bounds)
  )
  domain_fused_axes = FusedAxes(domain_legs)
  codomain_fused_axes = FusedAxes(GradedAxes.dual.(codomain_legs))
  fill_blockarray!(blockarray, data_mat, domain_fused_axes, codomain_fused_axes)
  return blockarray
end

# =====================================  Internals  ========================================

#------------------------------------  utility tools ---------------------------------------
function split_axes(fa::FusedAxes)
  legs = axes(fa)
  degens = BlockArrays.blocklengths.(legs)
  dimensions = broadcast.(SymmetrySectors.quantum_dimension, GradedAxes.blocklabels.(legs))
  return legs, degens, dimensions
end

function split_degen_dims(
  array_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  array_block_split_shape = (
    braid_tuples(domain_block_degens, domain_block_dims)...,
    braid_tuples(codomain_block_degens, codomain_block_dims)...,
  )
  split_array_block = reshape(array_block, array_block_split_shape)
  return split_array_block
end

function merge_degen_dims(split_array_block::AbstractArray)
  s0 = size(split_array_block)
  array_shape =
    ntuple(i -> s0[2 * i - 1], length(s0) ÷ 2) .* ntuple(i -> s0[2 * i], length(s0) ÷ 2)
  array_block = reshape(split_array_block, array_shape)
  return array_block
end

function permute_split_array_block(split_array_block::AbstractArray)
  N = ndims(split_array_block) ÷ 2
  array_data_perm = (ntuple(i -> 2 * i - 1, N)..., ntuple(i -> 2 * i, N)...)
  permuted_split_array_block = permutedims(split_array_block, array_data_perm)
  return permuted_split_array_block
end

function unpermute_split_array_block(permuted_split_array_block::AbstractArray)
  twoN = ndims(permuted_split_array_block)
  N = twoN ÷ 2
  inverse_array_data_perm = ntuple(i -> fld1(i, 2) + (1 - i % 2) * N, twoN)
  split_array_block = permutedims(permuted_split_array_block, inverse_array_data_perm)
  return split_array_block
end

function reshape_permuted_to_fused(
  permuted_split_array_block::AbstractArray, ::Val{N_CO}
) where {N_CO}
  N = ndims(permuted_split_array_block) ÷ 2
  permuted_array_shape = size(permuted_split_array_block)
  fused_array_block_shape = (
    prod(permuted_array_shape[begin:N_CO]),
    prod(permuted_array_shape[(N_CO + 1):N]),
    prod(permuted_array_shape[(N + 1):(N + N_CO)]),
    prod(permuted_array_shape[(N + N_CO + 1):end]),
  )
  fused_array_block = reshape(permuted_split_array_block, fused_array_block_shape)
  return fused_array_block
end

function reshape_fused_to_permuted(
  fused_array_block::AbstractArray,
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
  permuted_split_array_block = reshape(fused_array_block, degen_dim_shape)
  return permuted_split_array_block
end

function fuse_array_block(
  array_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  # start from an array outer block with e.g. N=6 axes divided into N_DO=3 ndims_domain
  # and N_CO=3 ndims_codomain. Each leg k can be decomposed as a product of external an
  # multiplicity extk and a quantum dimension dimk
  #
  #        ------------------------------array_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #

  # each leg of this this array outer block can now be opened to form a 2N-dims tensor.
  # note that this 2N-dims form is only defined at the level of the outer block,
  # not for a larger block.
  #
  #        ------------------------------split_array_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #
  split_array_block = split_degen_dims(
    array_block,
    domain_block_degens,
    domain_block_dims,
    codomain_block_degens,
    codomain_block_dims,
  )

  # Now we permute the axes to group together degenearacies on one side and irrep
  # dimensions on the other side. This is the bottleneck.
  #
  #     -------------------permuted_split_array_block-----------------------------------
  #     |      |       |       |        |      |      |      |      |      |     |     |
  #   ext1   ext2    ext3    ext4     ext5   ext6    dim1   dim2   dim3   dim4  dim5  dim6
  #
  permuted_split_array_block = permute_split_array_block(split_array_block)

  # Finally, it is convenient to merge together legs corresponding to domain or
  # to codomain and produce a 4-dims tensor
  #
  #        ---------------------fused_array_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  fused_array_block = reshape_permuted_to_fused(
    permuted_split_array_block, Val(length(domain_block_degens))
  )
  return fused_array_block
end

function contract_fusion_trees(
  fused_array_block::AbstractArray{<:Number,4},
  tree_domain::AbstractArray{<:Real,3},
  tree_codomain::AbstractArray{<:Real,3},
)
  # Input:
  #
  #        ---------------------fused_array_block--------------------
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
    (1, 2, 3, 5, 6), fused_array_block, (1, 2, 3, 4), tree_codomain, (4, 5, 6)
  )

  # contract domain tree
  #             -----------------------sym_data----------------------------
  #             |                  |                    |                 |
  #       ext1*ext2*ext3    struct_sec_codomain   ext4*ext5*ext6   struct_sec_domain
  #
  T = promote_type(eltype(fused_array_block), Float64)
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

#----------------------------------  cast from array ---------------------------------------

function fill_matrix_blocks!(
  data_mat::BlockSparseArrays.AbstractBlockSparseMatrix,
  blockarray::BlockArrays.AbstractBlockArray,
  domain_fused_axes::FusedAxes,
  codomain_fused_axes::FusedAxes,
)
  domain_legs, domain_degens, domain_dims = split_axes(domain_fused_axes)
  codomain_legs, codomain_degens, codomain_dims = split_axes(codomain_fused_axes)

  matrix_block_indices = intersect(domain_fused_axes, codomain_fused_axes)
  allowed_matrix_blocks = [
    BlockSparseArrays.view!(data_mat, BlockArrays.Block(bi)) for bi in matrix_block_indices
  ]
  allowed_sectors = GradedAxes.blocklabels(domain_fused_axes)[first.(matrix_block_indices)]
  allowed_outer_blocks = allowed_outer_blocks_sectors(
    domain_fused_axes, codomain_fused_axes, matrix_block_indices
  )

  # cache computed trees
  domain_trees_cache = Dict{NTuple{ndims(domain_fused_axes),Int},Vector{Array{Float64,3}}}()
  codomain_trees_cache = Dict{
    NTuple{ndims(codomain_fused_axes),Int},Vector{Array{Float64,3}}
  }()

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
  #      ------------------array_block-------------
  #      |    |    |                    |    |    |
  #     ext1 ext2 ext3                 ext4 ext5 ext6

  # loop for each allowed outer block
  for (outer_block, outer_block_sectors) in allowed_outer_blocks
    iter_do = outer_block[begin:ndims(domain_fused_axes)]
    domain_block_trees = get_tree!(
      domain_trees_cache, iter_do, domain_legs, allowed_sectors
    )

    iter_co = outer_block[(ndims(domain_fused_axes) + 1):end]
    codomain_block_trees = get_tree!(
      codomain_trees_cache, iter_co, codomain_legs, allowed_sectors
    )

    fused_array_block = fuse_array_block(
      view(blockarray, BlockArrays.Block(iter_do..., iter_co...)),
      getindex.(domain_degens, iter_do),
      getindex.(domain_dims, iter_do),
      getindex.(codomain_degens, iter_co),
      getindex.(codomain_dims, iter_co),
    )

    # loop for each symmetry sector allowed in this outer block
    for sect in outer_block_sectors

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
      #           ------------fused_array_block----
      #           |                               |
      #     ext1*ext2*ext3                 ext4*ext5*ext6

      # contract fusion trees and reshape symmetric block as a matrix
      # Note: a final permutedims is needed after the last contract
      # therefore cannot efficiently use contract!(allowed_matrix_blocks[...], ...)
      i_sec = findfirst(==(sect), allowed_sectors)
      sym_block_sec = contract_fusion_trees(
        fused_array_block, domain_block_trees[i_sec], codomain_block_trees[i_sec]
      )

      # find outer block location inside this matrix block && write it
      row_range = find_block_range(domain_fused_axes, iter_do, sect)
      col_range = find_block_range(codomain_fused_axes, iter_co, sect)
      @views allowed_matrix_blocks[i_sec][row_range, col_range] = sym_block_sec
    end
  end
end

#-----------------------------------  cast to array ----------------------------------------
function add_sector_block!(
  fused_array_block::AbstractArray{<:Number,4},
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
  #        ---------------------fused_array_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  return TensorAlgebra.contract!(
    fused_array_block,
    (1, 2, 3, 4),
    data_1tree,
    (1, 2, 6, 3, 5),
    tree_codomain,
    (4, 5, 6),
    1.0,
    1.0,
  )
end

function unfuse_array_block(
  fused_array_block::AbstractArray{<:Number,4},
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  #        ---------------------fused_array_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #

  #     -------------------permuted_split_array_block-----------------------------------
  #     |      |       |       |        |      |      |      |      |      |     |     |
  #   ext1   ext2    ext3    ext4     ext5   ext6    dim1   dim2   dim3   dim4  dim5  dim6
  #
  permuted_split_array_block = reshape_fused_to_permuted(
    fused_array_block,
    domain_block_degens,
    domain_block_dims,
    codomain_block_degens,
    codomain_block_dims,
  )

  #        ------------------------------split_array_block-------------------------
  #        |             |              |             |             |             |
  #       / \           / \            / \           / \           / \           / \
  #      /   \         /   \          /   \         /   \         /   \         /   \
  #    ext1  dim1    ext2  dim2     ext3  dim3    ext4  dim4    ext5  dim5    ext6 dim6
  #
  split_array_block = unpermute_split_array_block(permuted_split_array_block)

  #
  #        ------------------------------array_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #
  array_block = merge_degen_dims(split_array_block)
  return array_block
end

function fill_blockarray!(
  blockarray::BlockArrays.AbstractBlockArray,
  data_mat::AbstractMatrix,
  domain_fused_axes::FusedAxes,
  codomain_fused_axes::FusedAxes,
)
  domain_legs, domain_degens, domain_dims = split_axes(domain_fused_axes)
  codomain_legs, codomain_degens, codomain_dims = split_axes(codomain_fused_axes)

  matrix_block_blocks = sort(collect(BlockSparseArrays.block_stored_indices(data_mat)))
  existing_matrix_blocks = [view(data_mat, b) for b in matrix_block_blocks]
  matrix_block_indices = reinterpret(Tuple{Int,Int}, matrix_block_blocks)
  existing_sectors = GradedAxes.blocklabels(domain_fused_axes)[first.(matrix_block_indices)]
  existing_outer_blocks = allowed_outer_blocks_sectors(
    domain_fused_axes, codomain_fused_axes, matrix_block_indices
  )

  # cache computed trees
  domain_trees_cache = Dict{NTuple{ndims(domain_fused_axes),Int},Vector{Array{Float64,3}}}()
  codomain_trees_cache = Dict{
    NTuple{ndims(codomain_fused_axes),Int},Vector{Array{Float64,3}}
  }()

  # loop for each existing outer block
  for (outer_block, outer_block_sectors) in existing_outer_blocks
    iter_do = outer_block[begin:ndims(domain_fused_axes)]
    domain_block_degens = getindex.(domain_degens, iter_do)
    domain_block_dims = getindex.(domain_dims, iter_do)
    domain_block_trees = get_tree!(
      domain_trees_cache, iter_do, domain_legs, existing_sectors
    )

    iter_co = outer_block[(ndims(domain_fused_axes) + 1):end]
    codomain_block_degens = getindex.(codomain_degens, iter_co)
    codomain_block_dims = getindex.(codomain_dims, iter_co)
    codomain_block_trees = get_tree!(
      codomain_trees_cache, iter_co, codomain_legs, existing_sectors
    )

    #        ---------------------fused_array_block--------------------
    #        |                   |                 |                  |
    #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
    #
    fused_array_block_shape = (
      prod(domain_block_degens),
      prod(codomain_block_degens),
      prod(domain_block_dims),
      prod(codomain_block_dims),
    )
    fused_array_block = zeros(eltype(blockarray), fused_array_block_shape)

    # loop for each symmetry sector inside this configuration
    for sect in outer_block_sectors
      i_sec = findfirst(==(sect), existing_sectors)
      row_range = find_block_range(domain_fused_axes, iter_do, sect)
      col_range = find_block_range(codomain_fused_axes, iter_co, sect)
      sym_block_sec = view(existing_matrix_blocks[i_sec], row_range, col_range)
      add_sector_block!(
        fused_array_block,
        sym_block_sec,
        domain_block_trees[i_sec],
        codomain_block_trees[i_sec],
      )
    end

    blockarray[BlockArrays.Block(iter_do..., iter_co...)] = unfuse_array_block(
      fused_array_block,
      domain_block_degens,
      domain_block_dims,
      codomain_block_degens,
      codomain_block_dims,
    )
  end
end
