# This file defines interface to cast from and to dense array

function decompose_axis(leg::AbstractUnitRange)
  irrep_configuration = GradedAxes.blocklabels(leg)
  sector_dims = Sectors.quantum_dimension.(irrep_configuration)
  sector_degens = convert.(Int, (BlockArrays.blocklengths(leg)))
  block_boundaries = [0, cumsum(sector_degens .* sector_dims)...]
  shifts = range.(block_boundaries[begin:(end - 1)] .+ 1, block_boundaries[2:end])
  return irrep_configuration, sector_dims, sector_degens, shifts
end

function transpose_tuple_ntuple(t::Tuple{Vararg{<:NTuple{N,Any}}}) where {N}
  return ntuple(i -> ntuple(j -> t[j][i], length(t)), N)
end

decompose_axes(legs) = transpose_tuple_ntuple(decompose_axis.(legs))

# no codomain leg
function FusionTensor(::Tuple{}, domain_legs::Tuple, dense::AbstractArray)
  # add a dummy axis to compute data_matrix
  codomain_legs = (initialize_trivial_axis((), domain_legs),)
  dense_plus_1 = reshape(dense, (1, size(dense)...))
  ft_plus1 = FusionTensor(codomain_legs, domain_legs, dense_plus_1)
  data_mat = data_matrix(ft_plus1)
  # remove dummy axis
  return FusionTensor((), domain_legs, data_mat)
end

# no domain leg
function FusionTensor(codomain_legs::Tuple, ::Tuple{}, dense::AbstractArray)
  # add a dummy axis to compute data_matrix
  domain_legs = (initialize_trivial_axis(codomain_legs, ()),)
  dense_plus_1 = reshape(dense, (size(dense)..., 1))
  ft_plus1 = FusionTensor(codomain_legs, domain_legs, dense_plus_1)
  data_mat = data_matrix(ft_plus1)
  # remove dummy axis
  return FusionTensor(codomain_legs, (), data_mat)
end

# no leg
function FusionTensor(::Tuple{}, ::Tuple{}, dense::AbstractArray)
  data_mat = initialize_data_matrix(eltype(dense), (), ())
  data_mat[1, 1] = dense[1]
  return FusionTensor((), (), data_mat)
end

# constructor from dense array
# TBD add @inbounds?
# TBD dual in codomain?
# TBD inline internal degen in fusion_tree?
function FusionTensor(codomain_legs::Tuple, domain_legs::Tuple, dense::AbstractArray)

  # compile time check
  if length(codomain_legs) + length(domain_legs) != ndims(dense)
    throw(DomainError("legs are incompatible with array ndims"))
  end

  # input validation
  if Sectors.quantum_dimension.((codomain_legs..., domain_legs...)) != size(dense)
    throw(DomainError("legs dimensions are incompatible with dense array"))
  end

  # initialize data_matrix
  data_mat = initialize_data_matrix(eltype(dense), codomain_legs, domain_legs)
  row_sectors, col_sectors = GradedAxes.blocklabels.(axes(data_mat))
  row_shared_indices = findall(in(col_sectors), row_sectors)
  allowed_sectors = row_sectors[row_shared_indices]
  allowed_sectors_dims = Sectors.quantum_dimension.(allowed_sectors)
  col_shared_indices = findall(in(allowed_sectors), col_sectors)
  n_sectors = length(allowed_sectors)

  existing_blocks = BlockArrays.Block.(row_shared_indices, col_shared_indices)
  # TBD can this be avoided? TODO remove debug NaN
  for b in existing_blocks
    data_mat[b] = NaN * similar(data_mat[b])
  end

  # split axes into irrep configuration blocks
  codomain_irrep_configurations, codomain_dims, codomain_degens, codomain_shifts = decompose_axes(
    codomain_legs
  )
  domain_irrep_configurations, domain_dims, domain_degens, domain_shifts = decompose_axes(
    domain_legs
  )
  codomain_isdual = .!GradedAxes.isdual.(codomain_legs)  # TBD: dual
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # precompute codomain fusion trees and sort them by irrep
  # TODO anyway to change Vector{Vector} into Matrix{Vector{Array}}?
  #                      config  sector  ndof  tree
  codomain_fusion_trees = Vector{Vector{Vector{Vector{Float64}}}}()
  # predict shifts in data_matrix
  block_shifts_row = zeros(Int, n_sectors, 1)
  contribute_row_config = Vector{NTuple{length(codomain_legs),Int}}()
  init = initialize_trivial_axis(codomain_legs, domain_legs)
  for iter_co in Iterators.product(eachindex.(codomain_irrep_configurations)...)
    irreps_config = getindex.(codomain_irrep_configurations, iter_co)
    rep = reduce(GradedAxes.fusion_product, irreps_config; init=init)
    if !isempty(intersect(allowed_sectors, GradedAxes.blocklabels(rep)))
      trees, irreps = fusion_trees(irreps_config, codomain_isdual)
      kept_trees = [vec.(trees[findall(==(sec), irreps)]) for sec in allowed_sectors]
      internal_degens = length.(kept_trees)
      push!(contribute_row_config, iter_co)
      push!(codomain_fusion_trees, kept_trees)
      external_codomain_degen = getindex.(codomain_degens, iter_co)
      config_shift =
        block_shifts_row[:, end] + prod(external_codomain_degen) * internal_degens
      block_shifts_row = hcat(block_shifts_row, config_shift)
    end
  end

  # prepare contraction
  perm_dense_data = (
    ntuple(i -> 2 * i - 1, ndims(dense))..., ntuple(i -> 2 * i, ndims(dense))...
  )
  block_shifts_column = zeros(Int, n_sectors)

  # loop for each domain irrep configuration
  for iter_do in Iterators.product(eachindex.(domain_irrep_configurations)...)
    irreps_config = getindex.(domain_irrep_configurations, iter_do)
    rep = reduce(GradedAxes.fusion_product, irreps_config; init=init)
    if !isempty(intersect(allowed_sectors, GradedAxes.blocklabels(rep)))
      (domain_trees, domain_irreps) = fusion_trees(irreps_config, domain_isdual)
      domain_sector_degens = getindex.(domain_degens, iter_do)
      domain_sector_size = prod(domain_sector_degens)
      domain_sector_dims = getindex.(domain_dims, iter_do)
      domain_sector_dims_prod = prod(domain_sector_dims)
      doslices = getindex.(domain_shifts, iter_do)
      do_degen_dims = transpose_tuple_ntuple((domain_sector_degens, domain_sector_dims))
      do_shape = ntuple(i -> do_degen_dims[fld1(i, 2)][mod1(i, 2)], 2 * length(domain_legs))

      # keep only irreps that contribute
      pruned_domain_trees = [Vector{Matrix{Float64}}() for _ in 1:n_sectors]
      for i_sec in 1:n_sectors
        shape_dotree = (domain_sector_dims_prod, allowed_sectors_dims[i_sec])
        domain_sector_indices = findall(==(allowed_sectors[i_sec]), domain_irreps)
        pruned_domain_trees[i_sec] =
          (t -> reshape(t, shape_dotree)).(domain_trees[domain_sector_indices])
      end

      # loop for each codomain irrep configuration
      for (i_co, iter_co) in enumerate(contribute_row_config)
        codomain_trees = codomain_fusion_trees[i_co]
        codomain_sector_degens = getindex.(codomain_degens, iter_co)
        codomain_sector_size = prod(codomain_sector_degens)
        codomain_sector_dims = getindex.(codomain_dims, iter_co)
        codomain_sector_dims_prod = prod(codomain_sector_dims)
        coslices = getindex.(codomain_shifts, iter_co)
        co_degen_dims = transpose_tuple_ntuple((
          codomain_sector_degens, codomain_sector_dims
        ))
        co_shape = ntuple(
          i -> co_degen_dims[fld1(i, 2)][mod1(i, 2)], 2 * length(codomain_legs)
        )
        dense_shape = (co_shape..., do_shape...)
        sector_size = codomain_sector_size * domain_sector_size
        block_shape = (sector_size * codomain_sector_dims_prod, domain_sector_dims_prod)
        sym_shape = (codomain_sector_size, domain_sector_size)
        @show (coslices, doslices)
        dense_data_irrep_config = reshape(
          (@view dense[coslices..., doslices...]), dense_shape
        )
        dense_block = reshape(
          permutedims(dense_data_irrep_config, perm_dense_data), block_shape
        )

        # loop for each symmetry sector inside this configuration
        for i_sec in 1:n_sectors  # some sectors may be empty but at least one is allowed
          block = @view data_mat[existing_blocks[i_sec]]  # TBD @view needed?

          # write shapes explicitly for code clarity, could use :
          sec_dim = allowed_sectors_dims[i_sec]
          shape_1tree = (sector_size, codomain_sector_dims_prod * sec_dim)
          c1 = block_shifts_column[i_sec]

          # loop for each domain fusion tree    TBD inline in fusion tree?
          for dotree in pruned_domain_trees[i_sec]   # idicb
            data_1tree_mat = dense_block * dotree
            data_1tree = reshape(data_1tree_mat, shape_1tree)

            c2 = c1 + domain_sector_size
            r1 = block_shifts_row[i_sec, i_co]

            # loop for each codomain fusion tree
            for cotree in codomain_trees[i_sec]  # idirb
              sym_data = data_1tree * cotree
              r2 = r1 + codomain_sector_size
              # fill data_matrix
              block[(r1 + 1):r2, (c1 + 1):c2] = reshape(sym_data, sym_shape) / sec_dim
              r1 = r2
            end
            c1 = c2
          end
        end
      end
      block_shifts_column += domain_sector_size * length.(pruned_domain_trees)
    end
  end

  return FusionTensor(codomain_legs, domain_legs, data_mat)
end

# constructor from dense array with norm check
function FusionTensor(
  codomain_legs::Tuple, domain_legs::Tuple, dense::AbstractArray, tol_check::Real
)
  ft = FusionTensor(codomain_legs, domain_legs, dense)

  # check that norm is the same in input and output
  dense_norm = LinearAlgebra.norm(dense)
  if abs(LinearAlgebra.norm(ft) - dense_norm) > tol_check * dense_norm
    throw(DomainError("Dense tensor norm is not preserved in FusionTensor cast"))
  end
  return ft
end

# no codomain leg
function Base.Array(ft::FusionTensor{<:Any,N,Tuple{}}) where {N}
  domain_legs = domain_axes(ft)
  codomain_legs = (initialize_trivial_axis((), domain_legs),)
  ft_plus1 = FusionTensor(codomain_legs, domain_legs, data_matrix(ft))
  arr = Array(ft_plus1)
  return arr[1, ntuple(i -> :, N)...]
end

# no domain leg
function Base.Array(ft::FusionTensor{<:Any,N,<:Any,Tuple{}}) where {N}
  codomain_legs = codomain_axes(ft)
  domain_legs = (initialize_trivial_axis(codomain_legs, ()),)
  ft_plus1 = FusionTensor(codomain_legs, domain_legs, data_matrix(ft))
  arr = Array(ft_plus1)
  return arr[ntuple(i -> :, N)..., 1]
end

# no leg
function Base.Array(ft::FusionTensor{<:Any,0,Tuple{},Tuple{}})
  return reshape([data_matrix(ft)[1, 1]], ())
end

# cast to julia dense array with tensor size
function Base.Array(ft::FusionTensor)

  # initialize dense matrix
  dense = zeros(eltype(ft), size(ft))
  domain_legs = domain_axes(ft)
  codomain_legs = codomain_axes(ft)

  col_sectors = GradedAxes.blocklabels(matrix_column_axis(ft))
  existing_blocks = BlockSparseArrays.block_stored_indices(data_matrix(ft))
  n_sectors = length(existing_blocks)
  existing_sectors = [col_sectors[it[2]] for it in eachindex(existing_blocks)]   # subset of allowed_sectors
  matrix_blocks = [@view data_matrix(ft)[it] for it in existing_blocks]  # TBD @view?
  existing_sectors_dims = Sectors.quantum_dimension.(existing_sectors)

  # split axes into irrep configuration blocks
  codomain_irrep_configurations, codomain_dims, codomain_degens, codomain_shifts = decompose_axes(
    codomain_legs
  )
  domain_irrep_configurations, domain_dims, domain_degens, domain_shifts = decompose_axes(
    domain_legs
  )
  codomain_isdual = .!GradedAxes.isdual.(codomain_legs)  # TBD dual
  domain_isdual = GradedAxes.isdual.(domain_legs)

  # precompute codomain fusion trees and sort them by irrep
  # TODO anyway to change Vector{Vector} into Matrix{Vector{Array}}?
  #                      config  sector  ndof  tree
  codomain_fusion_trees = Vector{Vector{Vector{Matrix{Float64}}}}()
  # predict shifts in data_matrix
  contribute_row_config = Vector{NTuple{length(codomain_legs),Int}}()
  init = Sectors.trivial(eltype(col_sectors))

  for iter_co in Iterators.product(eachindex.(codomain_irrep_configurations)...)
    irreps_config = getindex.(codomain_irrep_configurations, iter_co)
    rep = reduce(GradedAxes.fusion_product, irreps_config; init=init)
    if !isempty(intersect(existing_sectors, GradedAxes.blocklabels(rep)))
      trees, irreps = fusion_trees(irreps_config, codomain_isdual)
      codomain_dims_prod = prod(getindex.(codomain_dims, iter_co))
      kept_trees = [
        reshape.(
          trees[findall(==(existing_sectors[i_sec]), irreps)],
          Ref((codomain_dims_prod, existing_sectors_dims[i_sec])),
        ) for i_sec in 1:n_sectors
      ]
      push!(contribute_row_config, iter_co)
      push!(codomain_fusion_trees, kept_trees)
    end
  end

  # prepare contraction
  perm_dense_data = ntuple(i -> fld1(i, 2) + (1 - i % 2) * ndims(ft), 2 * ndims(ft))
  block_shifts_column = zeros(Int, n_sectors)

  # TBD currently optimizes for contraction. Change to optimize for data moves?

  # loop for each domain irrep configuration
  for iter_do in Iterators.product(eachindex.(domain_irrep_configurations)...)
    irreps_config = getindex.(domain_irrep_configurations, iter_do)
    rep = reduce(GradedAxes.fusion_product, irreps_config; init=init)
    if !isempty(intersect(existing_sectors, GradedAxes.blocklabels(rep)))
      @show iter_do, irreps_config
      (domain_trees, domain_irreps) = fusion_trees(irreps_config, domain_isdual)
      domain_sector_degens = getindex.(domain_degens, iter_do)
      domain_sector_size = prod(domain_sector_degens)
      domain_sector_dims = getindex.(domain_dims, iter_do)
      domain_sector_dims_prod = prod(domain_sector_dims)
      doslices = getindex.(domain_shifts, iter_do)
      block_shifts_row = zeros(Int, n_sectors)

      # keep only irreps that contribute
      pruned_domain_trees = [Vector{Matrix{Float64}}() for _ in 1:n_sectors]
      for i_sec in 1:n_sectors
        shape_dotree = (domain_sector_dims_prod, existing_sectors_dims[i_sec])
        domain_sector_indices = findall(==(existing_sectors[i_sec]), domain_irreps)
        pruned_domain_trees[i_sec] =
          (t -> reshape(t, shape_dotree)).(domain_trees[domain_sector_indices])
      end

      # loop for each codomain irrep configuration
      for (i_co, iter_co) in enumerate(contribute_row_config)
        @show iter_co
        codomain_trees = codomain_fusion_trees[i_co]
        codomain_sector_degens = getindex.(codomain_degens, iter_co)
        codomain_sector_size = prod(codomain_sector_degens)
        codomain_sector_dims = getindex.(codomain_dims, iter_co)
        codomain_sector_dims_prod = prod(codomain_sector_dims)

        dense_shape = (
          codomain_sector_size * domain_sector_size * domain_sector_dims_prod,
          codomain_sector_dims_prod,
        )
        dense_block = zeros(eltype(ft), dense_shape)

        # loop for each symmetry sector inside this configuration
        for i_sec in 1:n_sectors  # some sectors may be empty but at least one is allowed
          @show i_sec

          sec_dim = existing_sectors_dims[i_sec]
          c1 = block_shifts_column[i_sec]
          r1 = block_shifts_row[i_sec]
          internal_degen_row_sector = length(codomain_trees[i_sec])
          internal_degen_col_sector = length(pruned_domain_trees[i_sec])
          r2 = r1 + internal_degen_row_sector * codomain_sector_size
          c2 = c1 + internal_degen_col_sector * domain_sector_size
          configuration_block_sector = reshape(
            matrix_blocks[i_sec][(r1 + 1):r2, (c1 + 1):c2],
            (
              internal_degen_row_sector,
              codomain_sector_size,
              internal_degen_col_sector,
              domain_sector_size,
            ),
          )

          # loop for each domain fusion tree
          for (idocb, dotree) in enumerate(pruned_domain_trees[i_sec])
            @show idocb
            data_1tree = vec(configuration_block_sector[:, :, idocb, :]) .* vec(dotree)'
            data_1tree_mat = reshape(
              data_1tree,
              (
                internal_degen_row_sector,
                codomain_sector_size * domain_sector_size * domain_sector_dims_prod,
                sec_dim,
              ),
            )

            # loop for each codomain fusion tree
            for (idorb, cotree) in enumerate(codomain_trees[i_sec])
              @show idorb
              dense_block += data_1tree_mat[idorb, :, :] * cotree'
            end
          end
          block_shifts_row[i_sec] = r2
        end
        coslices = getindex.(codomain_shifts, iter_co)
        degen_dim_shape = (
          codomain_sector_degens...,
          domain_sector_degens...,
          domain_sector_dims...,
          codomain_sector_dims...,
        )
        dense[coslices..., doslices...] = permutedims(
          reshape(dense_block, degen_dim_shape), perm_dense_data
        )
      end
      block_shifts_column += domain_sector_size * length.(pruned_domain_trees)
    end
  end

  return dense
end
