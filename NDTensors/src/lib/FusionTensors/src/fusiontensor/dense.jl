# This file defines interface to cast from and to dense array

# TBD dual in codomain?
# TODO rewrite once matrix_row_axis is dual
# TODO remove BlockSparseArray block initialization once writing is fixed

##################################  utility tools  #########################################
function shape_split_degen_dims(legs, it)
  config_sectors = getindex.(GradedAxes.blocklabels.(legs), it)
  config_degens = GradedAxes.unlabel.(getindex.(GradedAxes.blocklengths.(legs), it))
  config_dims = Sectors.quantum_dimension.(config_sectors)
  shape = ntuple(
    i -> Bool(i % 2) ? config_degens[fld1(i, 2)] : config_dims[fld1(i, 2)], 2 * length(legs)
  )
  return shape
end

function fused_sectors(secs::NTuple{<:Any,<:Sectors.AbstractCategory})
  return GradedAxes.blocklabels(reduce(GradedAxes.fusion_product, secs))
end
fused_sectors(secs::Tuple{<:Sectors.AbstractCategory}) = only(secs)
fused_sectors(::Tuple{}) = Sectors.sector(())

function intersect_sectors(
  codomain_sectors::NTuple{<:Any,C}, domain_sectors::NTuple{<:Any,C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(fused_sectors(codomain_sectors), fused_sectors(domain_sectors))
end

function intersect_sectors(
  config_irreps::NTuple{<:Any,C}, allowed::Vector{C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(fused_sectors(config_irreps), allowed)
end

function intersect_sectors(
  allowed::Vector{C}, config_irreps::NTuple{<:Any,C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(allowed, fused_sectors(config_irreps))
end

function intersect_sectors(
  ::Sectors.CategoryProduct{Tuple{}}, allowed::Vector{<:Sectors.AbstractCategory}
)
  return intersect_sectors(Sectors.trivial(eltype(allowed)), allowed)
end

function intersect_sectors(
  allowed::Vector{<:Sectors.AbstractCategory}, ::Sectors.CategoryProduct{Tuple{}}
)
  return intersect_sectors(allowed, Sectors.trivial(eltype(allowed)))
end

function intersect_sectors(
  ::Sectors.CategoryProduct{Tuple{}}, ::Sectors.CategoryProduct{Tuple{}}
)
  return Sectors.sector(())
end

function intersect_sectors(
  codomain_sectors::Vector{C}, domain_sectors::Vector{C}
) where {C<:Sectors.AbstractCategory}
  return intersect(codomain_sectors, domain_sectors)
end

function intersect_sectors(
  sec::C, domain_sectors::Vector{C}
) where {C<:Sectors.AbstractCategory}
  return sec ∈ domain_sectors ? [sec] : Vector{C}()
end

function intersect_sectors(
  codomain_sectors::Vector{C}, sec::C
) where {C<:Sectors.AbstractCategory}
  return sec ∈ codomain_sectors ? [sec] : Vector{C}()
end

############################  cast from dense: zero leg cases  #############################
# no codomain leg
function FusionTensor(dense::AbstractArray, ::Tuple{}, domain_legs::Tuple)
  # add a dummy axis to compute data_matrix
  codomain_legs = (initialize_trivial_axis((), domain_legs),)
  dense_plus_1 = reshape(dense, (1, size(dense)...))
  ft_plus1 = FusionTensor(dense_plus_1, codomain_legs, domain_legs)
  data_mat = data_matrix(ft_plus1)
  # remove dummy axis
  return FusionTensor(data_mat, (), domain_legs)
end

# no domain leg
function FusionTensor(dense::AbstractArray, codomain_legs::Tuple, ::Tuple{})
  # add a dummy axis to compute data_matrix
  domain_legs = (initialize_trivial_axis(codomain_legs, ()),)
  dense_plus_1 = reshape(dense, (size(dense)..., 1))
  ft_plus1 = FusionTensor(dense_plus_1, codomain_legs, domain_legs)
  data_mat = data_matrix(ft_plus1)
  # remove dummy axis
  return FusionTensor(data_mat, codomain_legs, ())
end

# no leg
function FusionTensor(dense::AbstractArray, ::Tuple{}, ::Tuple{})
  data_mat = initialize_data_matrix(eltype(dense), (), ())
  data_mat[firstindex(data_mat)] = first(dense)
  return FusionTensor(data_mat, (), ())
end

#################################  cast from dense array  ##################################
function FusionTensor(dense::AbstractArray, codomain_legs::Tuple, domain_legs::Tuple)
  bounds = Sectors.block_boundaries.((codomain_legs..., domain_legs...))
  blockarray = BlockArrays.PseudoBlockArray(dense, bounds...)
  return FusionTensor(blockarray, codomain_legs, domain_legs)
end

function FusionTensor(
  blockarray::BlockArrays.AbstractBlockArray, codomain_legs::Tuple, domain_legs::Tuple
)
  # compile time check
  N_CO = length(codomain_legs)
  N_DO = length(domain_legs)
  N = ndims(blockarray)
  if N_CO + N_DO != N
    throw(DomainError("legs are incompatible with array ndims"))
  end

  # input validation
  if Sectors.quantum_dimension.((codomain_legs..., domain_legs...)) != size(blockarray)
    throw(DomainError("legs dimensions are incompatible with array"))
  end

  # initialize data_matrix
  data_mat = initialize_data_matrix(eltype(blockarray), codomain_legs, domain_legs)

  # find sectors
  # TODO ADAPT ONCE ROW_AXIS IS DUAL
  row_sectors, col_sectors = GradedAxes.blocklabels.(axes(data_mat))
  dual_row_sectors = GradedAxes.dual.(row_sectors)
  col_shared_indices = findall(in(dual_row_sectors), col_sectors)
  allowed_sectors = col_sectors[col_shared_indices]
  allowed_sectors_dims = Sectors.quantum_dimension.(allowed_sectors)
  # col_shared indices may have non-trivial order. TODO check with dual row
  row_shared_indices::Vector{Int} = findfirst.(.==(allowed_sectors), Ref(dual_row_sectors))
  n_sectors = length(allowed_sectors)

  existing_blocks = BlockArrays.Block.(row_shared_indices, col_shared_indices)
  # TBD can this be avoided? TODO remove debug NaN
  for b in existing_blocks
    data_mat[b] = NaN * similar(data_mat[b])
  end

  # split axes into irrep configuration blocks
  codomain_irrep_configurations = GradedAxes.blocklabels.(codomain_legs)
  domain_irrep_configurations = GradedAxes.blocklabels.(domain_legs)
  domain_degens = GradedAxes.unlabel.(BlockArrays.blocklengths.(domain_legs))
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # precompute codomain trees
  codomain_trees, allowed_codomain_configs = precompute_allowed_trees(
    codomain_irrep_configurations, .!GradedAxes.isdual.(codomain_legs), allowed_sectors
  )  # TBD: dual

  # prepare contraction
  perm_dense_data = (ntuple(i -> 2 * i - 1, N)..., ntuple(i -> 2 * i, N)...)

  # loop for each domain irrep configuration
  block_shifts_columns = zeros(Int, n_sectors)
  for iter_do in Iterators.product(eachindex.(domain_irrep_configurations)...)
    domain_irreps_config = getindex.(domain_irrep_configurations, iter_do)
    allowed_sectors_domain = intersect_sectors(domain_irreps_config, allowed_sectors)
    if !isempty(allowed_sectors_domain)
      domain_trees_config = prune_fusion_trees_compressed(
        domain_irreps_config, domain_isdual, allowed_sectors
      )

      # loop for each codomain irrep configuration
      block_shifts_rows = zeros(Int, n_sectors)
      for (i_co, iter_co) in enumerate(allowed_codomain_configs)
        codomain_irreps_config = getindex.(codomain_irrep_configurations, iter_co)
        allowed_sectors_config = intersect_sectors(
          codomain_irreps_config, allowed_sectors_domain
        )
        if !isempty(allowed_sectors_config)

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
          # in this form, we can apply fusion trees on both the domain and the codomain.

          # loop for each symmetry sector inside this configuration
          for sec in allowed_sectors_config
            i_sec::Int = findfirst(==(sec), allowed_sectors)  # cannot be nothing

            # contract domain tree
            #             -------------------------data_1tree----------------------
            #             |               |              |           |            |
            #       degen1*degen2   degen3*degen4    dim1*dim2    sec_dim    ndof_sec_domain
            #
            data_1tree::Array{eltype(data_mat),5} = TensorAlgebra.contract(
              (1, 2, 3, 5, 6),
              dense_block_config,
              (1, 2, 3, 4),
              domain_trees_config[i_sec],
              (4, 5, 6),
            )

            # contract codomain tree
            #             -----------------------sym_data-------------------
            #             |              |                  |              |
            #       degen1*degen2   ndof_sec_codomain    degen3*degen4   ndof_sec_domain
            #
            sym_data::Array{eltype(data_mat),4} = TensorAlgebra.contract(
              (1, 7, 2, 6),   # HERE WE SET INNER STRUCTURE FOR MATRIX BLOCKS
              data_1tree,
              (1, 2, 3, 5, 6),
              codomain_trees[i_sec, i_co],
              (3, 5, 7),
            )

            # reshape sym_data as a matrix and write matrix block
            r1 = block_shifts_rows[i_sec]
            r2 = r1 + shape_config[1] * size(codomain_trees[i_sec, i_co], 3)
            c1 = block_shifts_columns[i_sec]
            c2 = c1 + shape_config[2] * size(domain_trees_config[i_sec], 3)
            data_mat[existing_blocks[i_sec]][(r1 + 1):r2, (c1 + 1):c2] =
              reshape(sym_data, (r2 - r1, c2 - c1)) / allowed_sectors_dims[i_sec]
            block_shifts_rows[i_sec] = r2
          end
        end
      end

      domain_config_size = prod(getindex.(domain_degens, iter_do))
      block_shifts_columns += domain_config_size * size.(domain_trees_config, 3)
    end
  end

  return FusionTensor(data_mat, codomain_legs, domain_legs)
end

# constructor from dense array with norm check
function FusionTensor(
  dense::AbstractArray, codomain_legs::Tuple, domain_legs::Tuple, tol_check::Real
)
  ft = FusionTensor(dense, codomain_legs, domain_legs)

  # check that norm is the same in input and output
  dense_norm = LinearAlgebra.norm(dense)
  if abs(LinearAlgebra.norm(ft) - dense_norm) > tol_check * dense_norm
    throw(DomainError("Dense tensor norm is not preserved in FusionTensor cast"))
  end
  return ft
end

############################  cast to dense: zero leg cases  ###############################
# no codomain leg
function Base.Array(ft::FusionTensor{<:Any,N,Tuple{}}) where {N}
  domain_legs = domain_axes(ft)
  codomain_legs = (initialize_trivial_axis((), domain_legs),)
  ft_plus1 = FusionTensor(data_matrix(ft), codomain_legs, domain_legs)
  arr = Array(ft_plus1)
  return arr[1, ..]
end

# no domain leg
function Base.Array(ft::FusionTensor{<:Any,N,<:Any,Tuple{}}) where {N}
  codomain_legs = codomain_axes(ft)
  domain_legs = (initialize_trivial_axis(codomain_legs, ()),)
  ft_plus1 = FusionTensor(data_matrix(ft), codomain_legs, domain_legs)
  arr = Array(ft_plus1)
  return arr[.., 1]
end

# no leg
function Base.Array(ft::FusionTensor{<:Any,0,Tuple{},Tuple{}})
  return reshape([first(data_matrix(ft))], ())
end

##################################  cast to dense array  ###################################
Base.Array(ft::FusionTensor) = Array(BlockSparseArrays.BlockSparseArray(ft))

function BlockSparseArrays.BlockSparseArray(ft::FusionTensor)
  # initialize block array
  domain_legs = domain_axes(ft)
  codomain_legs = codomain_axes(ft)
  bounds = Sectors.block_boundaries.((codomain_legs..., domain_legs...))
  blockarray = BlockSparseArrays.BlockSparseArray{eltype(ft)}(
    BlockArrays.blockedrange.(bounds)
  )

  col_sectors = GradedAxes.blocklabels(matrix_column_axis(ft))
  existing_blocks = BlockSparseArrays.block_stored_indices(data_matrix(ft))
  n_sectors = length(existing_blocks)
  existing_sectors = [col_sectors[it[2]] for it in eachindex(existing_blocks)]  # subset of allowed_sectors
  matrix_blocks = [data_matrix(ft)[it] for it in existing_blocks]

  # split axes into irrep configuration blocks
  codomain_irrep_configurations = GradedAxes.blocklabels.(codomain_legs)
  codomain_irrep_dimensions =
    broadcast.(Sectors.quantum_dimension, codomain_irrep_configurations)
  domain_irrep_configurations = GradedAxes.blocklabels.(domain_legs)
  domain_irrep_dimensions =
    broadcast.(Sectors.quantum_dimension, domain_irrep_configurations)
  codomain_degens = GradedAxes.unlabel.(BlockArrays.blocklengths.(codomain_legs))
  domain_degens = GradedAxes.unlabel.(BlockArrays.blocklengths.(domain_legs))
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # precompute codomain trees
  codomain_trees, existing_codomain_configs = precompute_allowed_trees(
    codomain_irrep_configurations, .!GradedAxes.isdual.(codomain_legs), existing_sectors
  )  # TBD: dual

  # prepare contraction
  inverse_perm_dense_data = ntuple(i -> fld1(i, 2) + (1 - i % 2) * ndims(ft), 2 * ndims(ft))

  # loop for each domain irrep configuration
  block_shifts_column = zeros(Int, n_sectors)
  for iter_do in Iterators.product(eachindex.(domain_irrep_configurations)...)
    domain_irreps_config = getindex.(domain_irrep_configurations, iter_do)
    existing_sectors_domain = intersect_sectors(domain_irreps_config, existing_sectors)
    if !isempty(existing_sectors_domain)
      domain_trees_config = prune_fusion_trees_compressed(
        getindex.(domain_irrep_configurations, iter_do), domain_isdual, existing_sectors
      )
      block_shifts_row = zeros(Int, n_sectors)
      domain_config_size = prod(getindex.(domain_degens, iter_do))

      # loop for each codomain irrep configuration
      for (i_co, iter_co) in enumerate(existing_codomain_configs)
        codomain_irreps_config = getindex.(codomain_irrep_configurations, iter_co)
        existing_sectors_config = intersect_sectors(
          codomain_irreps_config, existing_sectors_domain
        )
        if !isempty(existing_sectors_config)
          codomain_config_size = prod(getindex.(codomain_degens, iter_co))
          dense_shape_mat = (
            codomain_config_size,
            domain_config_size,
            prod(getindex.(codomain_irrep_dimensions, iter_co)),
            prod(getindex.(domain_irrep_dimensions, iter_do)),
          )

          #        -----------------dense_block_config--------------
          #        |                 |               |             |
          #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
          #
          dense_block_config = zeros(eltype(ft), dense_shape_mat)

          # loop for each symmetry sector inside this configuration
          for sec in existing_sectors_config
            i_sec::Int = findfirst(==(sec), existing_sectors)

            c1 = block_shifts_column[i_sec]
            r1 = block_shifts_row[i_sec]
            domain_ndof_config_sector = size(domain_trees_config[i_sec], 3)
            codomain_ndof_config_sector = size(codomain_trees[i_sec, i_co], 3)
            r2 = r1 + codomain_ndof_config_sector * codomain_config_size
            c2 = c1 + domain_ndof_config_sector * domain_config_size

            #             -----------------------sym_data-------------------
            #             |              |                  |              |
            #       degen1*degen2   ndof_sec_codomain    degen3*degen4   ndof_sec_domain
            #
            sym_data_shape = (
              codomain_config_size,
              codomain_ndof_config_sector,
              domain_config_size,
              domain_ndof_config_sector,
            )
            sym_data = reshape(
              (@view matrix_blocks[i_sec][(r1 + 1):r2, (c1 + 1):c2]), sym_data_shape
            )

            # contract codomain tree
            #             -------------------------data_1tree------------------------
            #             |               |               |              |          |
            #       degen1*degen2   degen3*degen4    ndof_sec_domain  dim1*dim2   sec_dim
            #
            data_1tree::Array{eltype(ft),5} = TensorAlgebra.contract(
              (1, 2, 6, 3, 5),
              sym_data,
              (1, 7, 2, 6),
              codomain_trees[i_sec, i_co],
              (3, 5, 7),
            )

            #        -----------------dense_block_config--------------
            #        |                 |               |             |
            #       degen1*degen2   degen3*degen4    dim1*dim2    dim3*dim4
            #
            TensorAlgebra.contract!(
              dense_block_config,
              (1, 2, 3, 4),
              data_1tree,
              (1, 2, 6, 3, 5),
              domain_trees_config[i_sec],
              (4, 5, 6),
              1.0,
              1.0,
            )

            block_shifts_row[i_sec] = r2
          end
          degen_dim_shape = (
            getindex.(codomain_degens, iter_co)...,
            getindex.(domain_degens, iter_do)...,
            getindex.(codomain_irrep_dimensions, iter_co)...,
            getindex.(domain_irrep_dimensions, iter_do)...,
          )

          #        -------------------dense_block_permuted----------------
          #        |        |       |       |       |      |      |      |
          #       degen1  degen2  degen3  degen4   dim1   dim2   dim3   dim4
          #
          dense_block_permuted = reshape(dense_block_config, degen_dim_shape)

          #        --------------dense_block_split_degen_dim--------------
          #        |        |       |       |       |      |      |      |
          #       degen1  degen2  degen3  degen4   dim1   dim2   dim3   dim4
          #
          dense_block_split_degen_dim = permutedims(
            dense_block_permuted, inverse_perm_dense_data
          )

          #        ----------------------------------------------------
          #        |                 |                |               |
          #       / \               / \              / \             / \
          #      /   \             /   \            /   \           /   \
          #  degen1  dim1      degen2  dim2     degen3  dim3    degen4  dim4
          #
          b = BlockArrays.Block(iter_co..., iter_do...)
          blockarray[b] = reshape(dense_block_split_degen_dim, size(blockarray[b]))
        end
      end
      block_shifts_column += domain_config_size * size.(domain_trees_config, 3)
    end
  end

  return blockarray
end
