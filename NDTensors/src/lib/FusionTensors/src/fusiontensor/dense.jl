# This file defines interface to cast from and to dense array

##################################  utility tools  #########################################
function shape_split_degen_dims(legs::Tuple, it::Tuple)
  config_degens = GradedAxes.unlabel.(getindex.(GradedAxes.blocklengths.(legs), it))
  config_dims = Sectors.quantum_dimension.(getindex.(GradedAxes.blocklabels.(legs), it))
  return braid_tuples(config_degens, config_dims)
end

function swap_dense_block(
  blockarray::BlockArrays.AbstractBlockArray,
  codomain_legs::Tuple,
  domain_legs::Tuple,
  iter_co::Tuple,
  iter_do::Tuple,
)
  # compile time checks
  N_CO = length(codomain_legs)
  N_DO = length(domain_legs)
  N = ndims(blockarray)
  @assert length(iter_co) == N_CO
  @assert length(iter_do) == N_DO
  @assert N_CO + N_DO == N

  # start from a dense tensor with e.g. N=4 axes divided into N_CO=2 ndims_codomain
  # and N_DO=2 ndims_domain. It may have several irrep configurations, select one
  # of them. The associated dense block has shape
  #
  #        ----------------------dense_block------------------
  #        |                  |                |             |
  #       degen1*dim1    degen2*dim2      degen3*dim3     degen4*dim4
  #
  dense_block = @view blockarray[BlockArrays.Block(iter_co..., iter_do...)]

  # each leg of this this dense block can now be opened to form a 2N-dims tensor.
  # note that this 2N-dims form is only defined at the level of the irrep
  # configuration, not for a larger dense block.
  #
  #        -------------dense_block_split_degen_dim------------
  #        |                 |                |               |
  #       / \               / \              / \             / \
  #      /   \             /   \            /   \           /   \
  #  degen1  dim1      degen2  dim2     degen3  dim3    degen4  dim4
  #
  dense_block_split_shape = (
    shape_split_degen_dims(codomain_legs, iter_co)...,
    shape_split_degen_dims(domain_legs, iter_do)...,
  )
  dense_block_split_degen_dim = reshape(dense_block, dense_block_split_shape)

  # Now we permute the axes to group together degenearacies on one side and irrep
  # dimensions on the other side. This is the bottleneck.
  #
  #        ------------------dense_block_permuted-----------------
  #        |        |       |       |       |      |      |      |
  #       degen1  degen2  degen3  degen4   dim1   dim2   dim3   dim4
  #
  perm_dense_data = (
    ntuple(i -> 2 * i - 1, ndims(blockarray))..., ntuple(i -> 2 * i, ndims(blockarray))...
  )
  dense_block_permuted = permutedims(dense_block_split_degen_dim, perm_dense_data)

  # Finally, it is convenient to merge together legs corresponding to codomain or
  # to codomain and produce a 4-dims tensor
  #
  #        -----------------dense_block_config--------------
  #        |                 |               |             |
  #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
  #
  permuted_dense_shape = size(dense_block_permuted)
  shape_config = (
    prod(permuted_dense_shape[begin:N_CO]),
    prod(permuted_dense_shape[(N_CO + 1):N]),
    prod(permuted_dense_shape[(N + 1):(N + N_CO)]),
    prod(permuted_dense_shape[(N + N_CO + 1):end]),
  )
  dense_block_config = reshape(dense_block_permuted, shape_config)
  return dense_block_config
end

function contract_fusion_trees(
  dense_block_config::AbstractArray{<:Any,4},
  tree_codomain::AbstractArray{Float64,3},
  tree_domain::AbstractArray{Float64,3},
  sec_dim::Int,
)
  #        -----------------dense_block_config--------------
  #        |                 |               |             |
  #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
  #
  # in this form, we can apply fusion trees on both the domain and the codomain.

  # contract domain tree
  #             -------------------------data_1tree----------------------
  #             |               |              |           |            |
  #       degen1*degen2   degen3*degen4    dim1*dim2    sec_dim    ndof_sec_domain
  #
  data_1tree = TensorAlgebra.contract(
    (1, 2, 3, 5, 6), dense_block_config, (1, 2, 3, 4), tree_domain, (4, 5, 6)
  )

  # contract codomain tree
  #             -----------------------sym_data-------------------
  #             |              |                  |              |
  #       degen1*degen2   ndof_sec_codomain    degen3*degen4   ndof_sec_domain
  #
  T = promote_type(eltype(dense_block_config), Float64)
  sym_data::Array{T,4} = TensorAlgebra.contract(
    (1, 7, 2, 6),   # HERE WE SET INNER STRUCTURE FOR MATRIX BLOCKS
    data_1tree,
    (1, 2, 3, 5, 6),
    tree_codomain,
    (3, 5, 7),
    1 / sec_dim,  # normalization factor
  )
  sym_shape = (size(sym_data, 1) * size(sym_data, 2), size(sym_data, 3) * size(sym_data, 4))
  sym_block = reshape(sym_data, sym_shape)
  return sym_block
end

#################################  cast from dense array  ##################################
function FusionTensor(dense::AbstractArray, codomain_legs::Tuple, domain_legs::Tuple)
  bounds = Sectors.block_dimensions.((codomain_legs..., domain_legs...))
  blockarray = BlockArrays.PseudoBlockArray(dense, bounds...)
  return FusionTensor(blockarray, codomain_legs, domain_legs)
end

function FusionTensor(
  blockarray::BlockArrays.AbstractBlockArray, codomain_legs::Tuple, domain_legs::Tuple
)
  # input validation
  if length(codomain_legs) + length(domain_legs) != ndims(blockarray)  # compile time
    throw(DomainError("legs are incompatible with array ndims"))
  end
  if Sectors.quantum_dimension.((codomain_legs..., domain_legs...)) != size(blockarray)
    throw(DomainError("legs dimensions are incompatible with array"))
  end

  # initialize data_matrix
  data_mat = initialize_data_matrix(eltype(blockarray), codomain_legs, domain_legs)

  # find sectors
  col_sectors = GradedAxes.blocklabels(axes(data_mat, 2))
  dual_row_sectors = GradedAxes.blocklabels(GradedAxes.dual(axes(data_mat, 1)))
  allowed_sectors = intersect_sectors(dual_row_sectors, col_sectors)
  n_sectors = length(allowed_sectors)
  allowed_sectors_dims = Sectors.quantum_dimension.(allowed_sectors)
  col_shared_indices = findall(in(allowed_sectors), col_sectors)
  row_shared_indices = findall(in(allowed_sectors), dual_row_sectors)
  existing_blocks = BlockArrays.Block.(row_shared_indices, col_shared_indices)

  # split axes into irrep configuration blocks
  dual_codomain_legs = GradedAxes.dual.(codomain_legs)
  codomain_irreps = GradedAxes.blocklabels.(dual_codomain_legs)
  nondual_codomain_irreps = GradedAxes.blocklabels.(GradedAxes.nondual.(codomain_legs))
  domain_irreps = GradedAxes.blocklabels.(domain_legs)
  nondual_domain_irreps = GradedAxes.blocklabels.(GradedAxes.nondual.(domain_legs))
  domain_degens = GradedAxes.unlabel.(BlockArrays.blocklengths.(domain_legs))
  codomain_isdual = GradedAxes.isdual.(dual_codomain_legs)
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # cache computed trees
  codomain_trees = Dict{NTuple{length(codomain_legs),Int},Vector{Array{Float64,3}}}()

  # loop for each domain irrep configuration
  block_shifts_columns = zeros(Int, n_sectors)
  for iter_do in Iterators.product(eachindex.(domain_irreps)...)
    domain_irreps_config = getindex.(domain_irreps, iter_do)
    allowed_sectors_domain = intersect_sectors(domain_irreps_config, allowed_sectors)
    if !isempty(allowed_sectors_domain)
      trees_domain_config = prune_fusion_trees_compressed(
        getindex.(nondual_domain_irreps, iter_do), domain_isdual, allowed_sectors
      )

      # loop for each codomain irrep configuration
      block_shifts_rows = zeros(Int, n_sectors)
      for iter_co in Iterators.product(eachindex.(codomain_irreps)...)
        allowed_sectors_config = intersect_sectors(
          getindex.(codomain_irreps, iter_co), allowed_sectors_domain
        )
        if !isempty(allowed_sectors_config)
          trees_codomain_config = get_tree!(
            codomain_trees,
            iter_co,
            nondual_codomain_irreps,
            codomain_isdual,
            allowed_sectors,
          )
          dense_block_config = swap_dense_block(
            blockarray, codomain_legs, domain_legs, iter_co, iter_do
          )

          # loop for each symmetry sector allowed in this configuration
          for i_sec in findall(in(allowed_sectors_config), allowed_sectors)

            # contract fusion trees and reshape symmetric block as a matrix
            sym_block = contract_fusion_trees(
              dense_block_config,
              trees_codomain_config[i_sec],
              trees_domain_config[i_sec],
              allowed_sectors_dims[i_sec],
            )

            # find position and write matrix block
            r1 = block_shifts_rows[i_sec]
            r2 = r1 + size(dense_block_config, 1) * size(trees_codomain_config[i_sec], 3)
            c1 = block_shifts_columns[i_sec]
            c2 = c1 + size(dense_block_config, 2) * size(trees_domain_config[i_sec], 3)
            @views data_mat[existing_blocks[i_sec]][(r1 + 1):r2, (c1 + 1):c2] = sym_block
            block_shifts_rows[i_sec] = r2
          end
        end
      end

      domain_config_size = prod(getindex.(domain_degens, iter_do))
      block_shifts_columns += domain_config_size * size.(trees_domain_config, 3)
    end
  end

  return FusionTensor(data_mat, codomain_legs, domain_legs)
end

##################################  cast to dense array  ###################################
function add_sector_block!(
  dense_block_config::AbstractArray{<:Any,4},
  sym_block::AbstractMatrix,
  tree_codomain::AbstractArray{Float64,3},
  tree_domain::AbstractArray{Float64,3},
)
  codomain_ndof_config_sector = size(tree_codomain, 3)
  domain_ndof_config_sector = size(tree_domain, 3)
  #             ---------------------sym_block--------------------
  #             |                                                |
  #       degen1*degen2*ndof_sec_codomain    degen3*degen4*ndof_sec_domain
  #
  sym_data_shape = (
    size(sym_block, 1) ÷ codomain_ndof_config_sector,
    codomain_ndof_config_sector,
    size(sym_block, 2) ÷ domain_ndof_config_sector,
    domain_ndof_config_sector,
  )

  #             -----------------------sym_data-------------------
  #             |              |                  |              |
  #       degen1*degen2   ndof_sec_codomain    degen3*degen4   ndof_sec_domain
  #
  sym_data = reshape(sym_block, sym_data_shape)

  # contract codomain tree
  #             -------------------------data_1tree------------------------
  #             |               |               |              |          |
  #       degen1*degen2   degen3*degen4    ndof_sec_domain  dim1*dim2   sec_dim
  #
  data_1tree = TensorAlgebra.contract(
    (1, 2, 6, 3, 5), sym_data, (1, 7, 2, 6), tree_codomain, (3, 5, 7)
  )

  #        -----------------dense_block_config--------------
  #        |                 |               |             |
  #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
  #
  return TensorAlgebra.contract!(
    dense_block_config,
    (1, 2, 3, 4),
    data_1tree,
    (1, 2, 6, 3, 5),
    tree_domain,
    (4, 5, 6),
    1.0,
    1.0,
  )
end

function unswap_dense_block(
  dense_block_config::AbstractArray{<:Any,4},
  codomain_degens_config::Tuple,
  domain_degens_config::Tuple,
  codomain_dims_config::Tuple,
  domain_dims_config::Tuple,
)
  N = length(codomain_degens_config) + length(domain_degens_config)
  @assert length(codomain_degens_config) == length(codomain_dims_config)
  @assert length(domain_degens_config) == length(domain_dims_config)
  #        -----------------dense_block_config--------------
  #        |                 |               |             |
  #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
  #

  #        -------------------dense_block_permuted----------------
  #        |        |       |       |       |      |      |      |
  #       degen1  degen2  degen3  degen4   dim1   dim2   dim3   dim4
  #
  degen_dim_shape = (
    codomain_degens_config...,
    domain_degens_config...,
    codomain_dims_config...,
    domain_dims_config...,
  )
  dense_block_permuted = reshape(dense_block_config, degen_dim_shape)

  #        -------------dense_block_split_degen_dim------------
  #        |                 |                |               |
  #       / \               / \              / \             / \
  #      /   \             /   \            /   \           /   \
  #  degen1  dim1      degen2  dim2     degen3  dim3    degen4  dim4
  #
  inverse_perm_dense_data = ntuple(i -> fld1(i, 2) + (1 - i % 2) * N, 2 * N)
  dense_block_split_degen_dim = permutedims(dense_block_permuted, inverse_perm_dense_data)

  #
  #        ----------------------dense_block------------------
  #        |                  |                |             |
  #       degen1*dim1    degen2*dim2      degen3*dim3     degen4*dim4
  #
  dense_shape =
    size(dense_block_split_degen_dim)[begin:2:end] .*
    size(dense_block_split_degen_dim)[2:2:end]
  dense_block = reshape(dense_block_split_degen_dim, dense_shape)
  return dense_block
end

Base.Array(ft::FusionTensor) = Array(BlockSparseArrays.BlockSparseArray(ft))

function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  # initialize block array
  codomain_legs = GradedAxes.dual.(codomain_axes(ft))
  domain_legs = domain_axes(ft)
  bounds = Sectors.block_dimensions.((codomain_legs..., domain_legs...))
  blockarray = BlockSparseArrays.BlockSparseArray{eltype(ft)}(
    BlockArrays.blockedrange.(bounds)
  )

  col_sectors = GradedAxes.blocklabels(matrix_column_axis(ft))
  existing_blocks = BlockSparseArrays.block_stored_indices(data_matrix(ft))
  n_sectors = length(existing_blocks)
  existing_sectors = [col_sectors[it[2]] for it in eachindex(existing_blocks)]  # subset of allowed_sectors
  matrix_blocks = [data_matrix(ft)[it] for it in existing_blocks]

  # split axes into irrep configuration blocks
  codomain_irreps = GradedAxes.blocklabels.(codomain_legs)
  nondual_codomain_irreps = GradedAxes.blocklabels.(GradedAxes.nondual.(codomain_legs))
  codomain_irrep_dimensions = broadcast.(Sectors.quantum_dimension, codomain_irreps)
  domain_irreps = GradedAxes.blocklabels.(domain_legs)
  nondual_domain_irreps = GradedAxes.blocklabels.(GradedAxes.nondual.(domain_legs))
  domain_irrep_dimensions = broadcast.(Sectors.quantum_dimension, domain_irreps)
  codomain_degens = broadcast.(GradedAxes.unlabel, BlockArrays.blocklengths.(codomain_legs))
  domain_degens = broadcast.(GradedAxes.unlabel, BlockArrays.blocklengths.(domain_legs))
  codomain_isdual = GradedAxes.isdual.(codomain_legs)
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # cache computed trees
  codomain_trees = Dict{NTuple{length(codomain_legs),Int},Vector{Array{Float64,3}}}()

  # loop for each domain irrep configuration
  block_shifts_column = zeros(Int, n_sectors)
  for iter_do in Iterators.product(eachindex.(domain_irreps)...)
    domain_irreps_config = getindex.(domain_irreps, iter_do)
    existing_sectors_domain = intersect_sectors(domain_irreps_config, existing_sectors)
    if !isempty(existing_sectors_domain)
      trees_domain_config = prune_fusion_trees_compressed(
        getindex.(nondual_domain_irreps, iter_do), domain_isdual, existing_sectors
      )
      block_shifts_row = zeros(Int, n_sectors)
      domain_config_size = prod(getindex.(domain_degens, iter_do))
      domain_dims_config = getindex.(domain_irrep_dimensions, iter_do)

      # loop for each codomain irrep configuration
      for iter_co in Iterators.product(eachindex.(codomain_irreps)...)
        existing_sectors_config = intersect_sectors(
          getindex.(codomain_irreps, iter_co), existing_sectors_domain
        )
        if !isempty(existing_sectors_config)
          trees_codomain_config = get_tree!(
            codomain_trees,
            iter_co,
            nondual_codomain_irreps,
            codomain_isdual,
            existing_sectors,
          )
          codomain_config_size = prod(getindex.(codomain_degens, iter_co))
          codomain_dims_config = getindex.(codomain_irrep_dimensions, iter_co)
          #        -----------------dense_block_config--------------
          #        |                 |               |             |
          #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
          #
          dense_shape_mat = (
            codomain_config_size,
            domain_config_size,
            prod(codomain_dims_config),
            prod(domain_dims_config),
          )
          dense_block_config = zeros(eltype(ft), dense_shape_mat)

          # loop for each symmetry sector inside this configuration
          for i_sec in findall(in(existing_sectors_config), existing_sectors)
            c1 = block_shifts_column[i_sec]
            r1 = block_shifts_row[i_sec]
            r2 = r1 + size(trees_codomain_config[i_sec], 3) * codomain_config_size
            c2 = c1 + size(trees_domain_config[i_sec], 3) * domain_config_size
            sym_block = @views matrix_blocks[i_sec][(r1 + 1):r2, (c1 + 1):c2]

            add_sector_block!(
              dense_block_config,
              sym_block,
              trees_codomain_config[i_sec],
              trees_domain_config[i_sec],
            )

            block_shifts_row[i_sec] = r2
          end

          b = BlockArrays.Block(iter_co..., iter_do...)
          blockarray[b] = unswap_dense_block(
            dense_block_config,
            getindex.(codomain_degens, iter_co),
            getindex.(domain_degens, iter_do),
            codomain_dims_config,
            domain_dims_config,
          )
        end
      end
      block_shifts_column += domain_config_size * size.(trees_domain_config, 3)
    end
  end

  return blockarray
end
