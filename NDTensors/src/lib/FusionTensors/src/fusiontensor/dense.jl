# This file defines interface to cast from and to dense array

##################################  utility tools  #########################################
function find_allowed_blocks(data_mat::BlockSparseArrays.AbstractBlockSparseMatrix)
  return find_allowed_blocks(axes(data_mat)...)
end

function find_allowed_blocks(row_axis::AbstractUnitRange, col_axis::AbstractUnitRange)
  col_sectors = GradedAxes.blocklabels(col_axis)
  dual_row_sectors = GradedAxes.blocklabels(GradedAxes.dual(row_axis))
  allowed_sectors = intersect_sectors(dual_row_sectors, col_sectors)
  col_shared_indices = findall(in(allowed_sectors), col_sectors)
  row_shared_indices = findall(in(allowed_sectors), dual_row_sectors)
  allowed_blocks = BlockArrays.Block.(row_shared_indices, col_shared_indices)
  return allowed_sectors, allowed_blocks
end

function find_existing_blocks(data_mat::BlockSparseArrays.AbstractBlockSparseMatrix)
  col_sectors = GradedAxes.blocklabels(axes(data_mat, 2))
  existing_blocks = BlockSparseArrays.block_stored_indices(data_mat)
  existing_sectors = [col_sectors[it[2]] for it in eachindex(existing_blocks)]
  return existing_sectors, existing_blocks
end

function split_axes(legs::Tuple)
  arrows = GradedAxes.isdual.(legs)
  irreps = GradedAxes.blocklabels.(legs)
  degens = BlockArrays.blocklengths.(legs)
  dimensions = broadcast.(Sectors.quantum_dimension, irreps)
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

function reshape_permuted_to_compressed(
  permuted_split_dense_block::AbstractArray, ::Val{N_CO}
) where {N_CO}
  N = ndims(permuted_split_dense_block) ÷ 2
  permuted_dense_shape = size(permuted_split_dense_block)
  compressed_dense_block_shape = (
    prod(permuted_dense_shape[begin:N_CO]),
    prod(permuted_dense_shape[(N_CO + 1):N]),
    prod(permuted_dense_shape[(N + 1):(N + N_CO)]),
    prod(permuted_dense_shape[(N + N_CO + 1):end]),
  )
  compressed_dense_block = reshape(permuted_split_dense_block, compressed_dense_block_shape)
  return compressed_dense_block
end

function reshape_compressed_to_permuted(
  compressed_dense_block::AbstractArray,
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
  permuted_split_dense_block = reshape(compressed_dense_block, degen_dim_shape)
  return permuted_split_dense_block
end

function compress_dense_block(
  dense_block::AbstractArray,
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  # start from a dense tensor block with e.g. N=6 axes divided into N_DO=3 ndims_domain
  # and N_CO=3 ndims_codomain. Each leg k can be decomposed as a product of external an
  # multiplicity extk and a quantum dimension dimk
  #
  #        ------------------------------dense_block-------------------------------
  #        |             |             |              |               |           |
  #       ext1*dim1   ext2*dim2     ext3*dim3      ext4*dim4       ext5*dim5   ext6*dim6
  #

  # each leg of this this dense block can now be opened to form a 2N-dims tensor.
  # note that this 2N-dims form is only defined at the level of the irrep
  # configuration, not for a larger dense block.
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
  #        ----------------compressed_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  compressed_dense_block = reshape_permuted_to_compressed(
    permuted_split_dense_block, Val(length(domain_block_degens))
  )
  return compressed_dense_block
end

function contract_fusion_trees(
  compressed_dense_block::AbstractArray{<:Number,4},
  tree_domain::AbstractArray{<:Real,3},
  tree_codomain::AbstractArray{<:Real,3},
  sec_dim::Int,
)
  # Input:
  #
  #        ----------------compressed_dense_block--------------------
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
    (1, 2, 3, 5, 6), compressed_dense_block, (1, 2, 3, 4), tree_codomain, (4, 5, 6)
  )

  # contract domain tree
  #             -----------------------sym_data----------------------------
  #             |                  |                    |                 |
  #       ext1*ext2*ext3    struct_sec_codomain   ext4*ext5*ext6   struct_sec_domain
  #
  T = promote_type(eltype(compressed_dense_block), Float64)
  sym_data::Array{T,4} = TensorAlgebra.contract(
    (1, 7, 2, 6),   # HERE WE SET INNER STRUCTURE FOR MATRIX BLOCKS
    data_1tree,
    (1, 2, 3, 5, 6),
    tree_domain,
    (3, 5, 7),
    1 / sec_dim,  # normalization factor
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
function FusionTensor(dense::AbstractArray, domain_legs::Tuple, codomain_legs::Tuple)
  bounds = Sectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockArrays.BlockedArray(dense, bounds...)
  return FusionTensor(blockarray, domain_legs, codomain_legs)
end

function FusionTensor(
  blockarray::BlockArrays.AbstractBlockArray, domain_legs::Tuple, codomain_legs::Tuple
)
  # input validation
  if length(domain_legs) + length(codomain_legs) != ndims(blockarray)  # compile time
    throw(DomainError("legs are incompatible with array ndims"))
  end
  if Sectors.quantum_dimension.((domain_legs..., codomain_legs...)) != size(blockarray)
    throw(DomainError("legs dimensions are incompatible with array"))
  end

  data_mat = initialize_data_matrix(eltype(blockarray), domain_legs, codomain_legs)
  fill_matrix_blocks!(data_mat, blockarray, domain_legs, codomain_legs)
  return FusionTensor(data_mat, domain_legs, codomain_legs)
end

function fill_matrix_blocks!(
  data_mat::BlockSparseArrays.AbstractBlockSparseMatrix,
  blockarray::BlockArrays.AbstractBlockArray,
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  # find sectors
  allowed_sectors, allowed_blocks = find_allowed_blocks(data_mat)
  allowed_matrix_blocks = [BlockSparseArrays.view!(data_mat, b) for b in allowed_blocks]
  return fill_matrix_blocks!(
    allowed_matrix_blocks, allowed_sectors, blockarray, domain_legs, codomain_legs
  )
end

function fill_matrix_blocks!(
  allowed_matrix_blocks::Vector{<:AbstractMatrix},
  allowed_sectors::Vector{<:Sectors.AbstractCategory},
  blockarray::BlockArrays.AbstractBlockArray,
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  # domain needs to be dualed in fusion tree
  domain_arrows, domain_irreps, domain_degens, domain_dims = split_axes(
    GradedAxes.dual.(domain_legs)
  )
  codomain_arrows, codomain_irreps, codomain_degens, codomain_dims = split_axes(
    codomain_legs
  )

  # precompute matrix block normalization factor
  allowed_sectors_dims = Sectors.quantum_dimension.(allowed_sectors)

  # cache computed trees
  domain_trees = Dict{NTuple{length(domain_legs),Int},Vector{Array{Float64,3}}}()

  # Below, we loop over every allowed dense block, contract domain and codomain fusion trees
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
  block_shifts_columns = zeros(Int, length(allowed_sectors))
  for iter_co in Iterators.product(eachindex.(codomain_irreps)...)
    codomain_block_irreps = getindex.(codomain_irreps, iter_co)
    codomain_block_allowed_sectors = intersect_sectors(
      codomain_block_irreps, allowed_sectors
    )
    if !isempty(codomain_block_allowed_sectors)
      codomain_block_trees = prune_fusion_trees_compressed(
        codomain_block_irreps, codomain_arrows, allowed_sectors
      )

      # loop for each domain irrep configuration
      block_shifts_rows = zeros(Int, length(allowed_sectors))
      for iter_do in Iterators.product(eachindex.(domain_irreps)...)
        block_allowed_sectors = intersect_sectors(
          getindex.(domain_irreps, iter_do), codomain_block_allowed_sectors
        )
        if !isempty(block_allowed_sectors)
          domain_block_trees = get_tree!(
            domain_trees, iter_do, domain_irreps, domain_arrows, allowed_sectors
          )
          compressed_dense_block = compress_dense_block(
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
            #           -------compressed_dense_block----
            #           |                               |
            #     ext1*ext2*ext3                 ext4*ext5*ext6

            # contract fusion trees and reshape symmetric block as a matrix
            # Note: a final permutedims is needed after the last contract
            # therefore cannot efficiently use contract!(allowed_matrix_blocks[...], ...)
            # TBD something like permutedims!(reshape(view), sym_block, (1,3,2,4))?
            sym_block_sec = contract_fusion_trees(
              compressed_dense_block,
              domain_block_trees[i_sec],
              codomain_block_trees[i_sec],
              allowed_sectors_dims[i_sec],
            )

            # find position and write matrix block
            r1 = block_shifts_rows[i_sec]
            r2 = r1 + size(compressed_dense_block, 1) * size(domain_block_trees[i_sec], 3)
            c1 = block_shifts_columns[i_sec]
            c2 = c1 + size(compressed_dense_block, 2) * size(codomain_block_trees[i_sec], 3)
            @views allowed_matrix_blocks[i_sec][(r1 + 1):r2, (c1 + 1):c2] = sym_block_sec
            block_shifts_rows[i_sec] = r2
          end
        end
      end

      codomain_block_length = prod(getindex.(codomain_degens, iter_co))
      block_shifts_columns += codomain_block_length * size.(codomain_block_trees, 3)
    end
  end
end

##################################  cast to dense array  ###################################
function add_sector_block!(
  compressed_dense_block::AbstractArray{<:Number,4},
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
  #        ----------------compressed_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #
  return TensorAlgebra.contract!(
    compressed_dense_block,
    (1, 2, 3, 4),
    data_1tree,
    (1, 2, 6, 3, 5),
    tree_codomain,
    (4, 5, 6),
    1.0,
    1.0,
  )
end

function decompress_dense_block(
  compressed_dense_block::AbstractArray{<:Number,4},
  domain_block_degens::Tuple,
  domain_block_dims::Tuple,
  codomain_block_degens::Tuple,
  codomain_block_dims::Tuple,
)
  #        ----------------compressed_dense_block--------------------
  #        |                   |                 |                  |
  #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
  #

  #     -------------------permuted_split_dense_block-----------------------------------
  #     |      |       |       |        |      |      |      |      |      |     |     |
  #   ext1   ext2    ext3    ext4     ext5   ext6    dim1   dim2   dim3   dim4  dim5  dim6
  #
  permuted_split_dense_block = reshape_compressed_to_permuted(
    compressed_dense_block,
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

Base.Array(ft::FusionTensor) = Array(BlockSparseArrays.BlockSparseArray(ft))

function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  domain_legs = domain_axes(ft)
  codomain_legs = codomain_axes(ft)
  bounds = Sectors.block_dimensions.((domain_legs..., codomain_legs...))
  blockarray = BlockSparseArrays.BlockSparseArray{eltype(ft)}(
    BlockArrays.blockedrange.(bounds)
  )
  fill_blockarray!(blockarray, data_matrix(ft), domain_legs, codomain_legs)
  return blockarray
end

function fill_blockarray!(
  blockarray::BlockArrays.AbstractBlockArray,
  data_mat::BlockSparseArrays.AbstractBlockSparseMatrix,
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  existing_sectors, existing_blocks = find_existing_blocks(data_mat)
  existing_matrix_blocks = [view(data_mat, b) for b in existing_blocks]
  return fill_blockarray!(
    blockarray, existing_sectors, existing_matrix_blocks, domain_legs, codomain_legs
  )
end

function fill_blockarray!(
  blockarray::BlockArrays.AbstractBlockArray,
  existing_sectors::Vector{<:Sectors.AbstractCategory},
  existing_matrix_blocks::Vector{<:AbstractMatrix},
  domain_legs::Tuple,
  codomain_legs::Tuple,
)
  # domain needs to be dualed in fusion tree
  domain_arrows, domain_irreps, domain_degens, domain_dims = split_axes(
    GradedAxes.dual.(domain_legs)
  )
  codomain_arrows, codomain_irreps, codomain_degens, codomain_dims = split_axes(
    codomain_legs
  )

  # cache computed trees
  domain_trees = Dict{NTuple{length(domain_legs),Int},Vector{Array{Float64,3}}}()

  # loop for each codomain irrep configuration
  block_shifts_column = zeros(Int, length(existing_sectors))
  for iter_co in Iterators.product(eachindex.(codomain_irreps)...)
    codomain_block_irreps = getindex.(codomain_irreps, iter_co)
    codomain_block_existing_sectors = intersect_sectors(
      codomain_block_irreps, existing_sectors
    )
    if !isempty(codomain_block_existing_sectors)
      codomain_block_trees = prune_fusion_trees_compressed(
        codomain_block_irreps, codomain_arrows, existing_sectors
      )
      block_shifts_row = zeros(Int, length(existing_sectors))
      codomain_block_length = prod(getindex.(codomain_degens, iter_co))
      codomain_block_dims = getindex.(codomain_dims, iter_co)

      # loop for each domain irrep configuration
      for iter_do in Iterators.product(eachindex.(domain_irreps)...)
        block_existing_sectors = intersect_sectors(
          getindex.(domain_irreps, iter_do), codomain_block_existing_sectors
        )
        if !isempty(block_existing_sectors)
          domain_block_trees = get_tree!(
            domain_trees, iter_do, domain_irreps, domain_arrows, existing_sectors
          )
          domain_block_length = prod(getindex.(domain_degens, iter_do))
          domain_block_dims = getindex.(domain_dims, iter_do)

          #        ----------------compressed_dense_block--------------------
          #        |                   |                 |                  |
          #  ext1*ext2*ext3      ext4*ext5*ext6    dim1*dim2*dim3    dim4*dim5*dim6
          #
          compressed_dense_block_shape = (
            domain_block_length,
            codomain_block_length,
            prod(domain_block_dims),
            prod(codomain_block_dims),
          )
          compressed_dense_block = zeros(eltype(blockarray), compressed_dense_block_shape)

          # loop for each symmetry sector inside this configuration
          for i_sec in findall(in(block_existing_sectors), existing_sectors)
            c1 = block_shifts_column[i_sec]
            r1 = block_shifts_row[i_sec]
            r2 = r1 + size(domain_block_trees[i_sec], 3) * domain_block_length
            c2 = c1 + size(codomain_block_trees[i_sec], 3) * codomain_block_length

            sym_block_sec = view(existing_matrix_blocks[i_sec], (r1 + 1):r2, (c1 + 1):c2)
            add_sector_block!(
              compressed_dense_block,
              sym_block_sec,
              domain_block_trees[i_sec],
              codomain_block_trees[i_sec],
            )

            block_shifts_row[i_sec] = r2
          end

          blockarray[BlockArrays.Block(iter_do..., iter_co...)] = decompress_dense_block(
            compressed_dense_block,
            getindex.(domain_degens, iter_do),
            domain_block_dims,
            getindex.(codomain_degens, iter_co),
            codomain_block_dims,
          )
        end
      end
      block_shifts_column += codomain_block_length * size.(codomain_block_trees, 3)
    end
  end
end
