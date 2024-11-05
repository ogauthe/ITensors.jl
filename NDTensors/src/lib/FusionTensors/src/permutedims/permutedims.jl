# This file defines permutedims for a FusionTensor

# =================================  High level interface  =================================
# permutedims with 1 tuple of 2 separate tuples
function fusiontensor_permutedims(ft::FusionTensor, new_leg_indices::Tuple{NTuple,NTuple})
  return fusiontensor_permutedims(ft, new_leg_indices...)
end

# permutedims with 2 separate tuples
function fusiontensor_permutedims(
  ft::FusionTensor, new_domain_indices::Tuple, new_codomain_indices::Tuple
)
  biperm = TensorAlgebra.blockedperm(new_domain_indices, new_codomain_indices)
  return permutedims(ft, biperm)
end

function fusiontensor_permutedims(
  ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation{2}
)
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy. Also handle tricky 0-dim case.
  if ndims_domain(ft) == first(BlockArrays.blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  # TODO remove me
  return naive_permutedims(ft, biperm)

  permuted_data_matrix = permute_data_matrix(
    data_matrix(ft), domain_axes(ft), codomain_axes(ft), biperm
  )
  new_domain_legs, new_codomain_legs = TensorAlgebra.blockpermute(axes(ft), biperm)
  permuted = FusionTensor(permuted_data_matrix, new_domain_legs, new_codomain_legs)
  return permuted
end

function naive_permutedims(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  new_domain_legs, new_codomain_legs = TensorAlgebra.blockpermute(axes(ft), biperm)

  # stupid permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(biperm))
  permuted = FusionTensor(permuted_arr, new_domain_legs, new_codomain_legs)

  return permuted
end

# =================================  Low level interface  ==================================
function permute_data_matrix(
  old_data_matrix::Union{
    BlockSparseArrays.AbstractBlockSparseMatrix,
    LinearAlgebra.Adjoint{<:Number,<:BlockSparseArrays.AbstractBlockSparseMatrix},
  },
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  biperm::TensorAlgebra.BlockedPermutation{2},
)
  # TBD use FusedAxes as input?

  unitaries = compute_unitaries(  # TODO cache me
    old_domain_legs,
    old_codomain_legs,
    biperm,
  )

  # TODO cache FusedAxes
  old_domain_fused_axes = FusedAxes(old_domain_legs)
  old_codomain_fused_axes = FusedAxes(GradedAxes.dual.(old_codomain_legs))
  new_domain_legs, new_codomain_legs = TensorAlgebra.blockpermute(axes(ft), biperm)
  new_domain_fused_axes = FusedAxes(new_domain_legs)
  new_codomain_fused_axes = FusedAxes(GradedAxes.dual.(new_codomain_legs))
  new_data_matrix = initialize_data_matrix(
    eltype(old_data_matrix), new_domain_fused_axes, new_codomain_fused_axes
  )

  fill_data_matrix!(
    new_data_matrix,
    old_data_matrix,
    old_domain_fused_axes,
    old_codomain_fused_axes,
    new_domain_fused_axes,
    new_codomain_fused_axes,
    biperm,
    unitaries,
  )
  return new_data_matrix
end

# =====================================  Internals  ========================================
function add_structural_axes(biperm::TensorAlgebra.BlockedPermutation{2})
  block1 = biperm[BlockArrays.Block(1)]
  # TODO
  extended_perm = (1, block1..., length(block1), biperm[BlockArrays.Block(2)]...)
  return extended_perm
end

function fill_data_matrix!(
  new_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_data_matrix::Union{
    BlockSparseArrays.AbstractBlockSparseMatrix,
    LinearAlgebra.Adjoint{<:Number,<:BlockSparseArrays.AbstractBlockSparseMatrix},
  },
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  new_domain_fused_axes::FusedAxes,
  new_codomain_fused_axes::FusedAxes,
  biperm::TensorAlgebra.BlockedPermutation{2},
  unitaries::Dict,
)
  @assert ndims(old_domain_fused_axes) + ndims(old_codomain_fused_axes) == N
  @assert ndims(new_domain_fused_axes) + ndims(new_codomain_fused_axes) == N

  # TODO share functions with dense
  matrix_block_blocks = sort(
    collect(BlockSparseArrays.block_stored_indices(old_data_matrix))
  )
  old_existing_matrix_blocks = [view(old_data_matrix, b) for b in matrix_block_blocks]
  old_matrix_block_indices = reinterpret(Tuple{Int,Int}, matrix_block_blocks)
  #old_existing_sectors = GradedAxes.blocklabels(old_domain_fused_axes)[first.(
  #  old_matrix_block_indices
  #)]
  old_existing_outer_blocks = allowed_outer_blocks_sectors(
    old_domain_fused_axes, old_codomain_fused_axes, old_matrix_block_indices
  )

  extended_perm = add_structural_axes(biperm)

  # loop for each existing outer block TODO parallelize
  for (old_outer_block, old_outer_block_sectors) in old_existing_outer_blocks
    write_new_outer_block!(
      new_data_matrix,
      old_outer_block,
      old_outer_block_sectors,
      old_existing_matrix_blocks,
      unitaries[old_outer_block],
      extended_perm,
      old_domain_fused_axes,
      old_codomain_fused_axes,
      new_domain_fused_axes,
      new_codomain_fused_axes,
    )
  end
end

function write_new_outer_block!(
  new_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_outer_block::Tuple{Vararg{Int}},
  old_outer_block_sectors::Vector{<:SymmetrySectors.AbstractSector},
  old_existing_matrix_blocks::Vector{<:Matrix},
  unitary::BlockArrays.AbstractBlockMatrix,
  extended_perm::Tuple{Vararg{Int}},
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  new_domain_fused_axes::FusedAxes,
  new_codomain_fused_axes::FusedAxes,
)
  new_outer_array = permute_outer_block(
    old_existing_matrix_blocks,
    old_outer_block,
    old_outer_block_sectors,
    old_domain_fused_axes,
    old_codomain_fused_axes,
    extended_perm,
    unitary,
  )
  new_outer_block = map(i -> old_outer_block[i], flat_permutation)
  return write_new_outer_block!(
    new_data_matrix,
    new_outer_block_array,
    new_outer_block,
    new_domain_fused_axes,
    new_codomain_fused_axes,
  )
end

function permute_outer_block(
  old_existing_matrix_blocks::Vector{<:Matrix},
  old_outer_block::Tuple{Vararg{Int}},
  old_outer_block_sectors,
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  extended_perm::Tuple{Vararg{Int}},
  unitary::BlockArrays.AbstractBlockMatrix,
)
  old_domain_block = old_outer_block[begin:ndims(old_domain_fused_axes)]
  old_codomain_block = old_outer_block[(ndims(old_domain_fused_axes) + 1):end]
  old_domain_ext_mult = block_external_multiplicities(
    old_domain_fused_axes, old_domain_block
  )
  old_codomain_ext_mult = block_external_multiplicities(
    old_codomain_fused_axes, old_codomain_block
  )
  new_outer_shape = (
    size(unitary, 1), prod((old_domain_ext_mult..., old_codomain_ext_mult...))
  )
  new_outer_block_array = zeros(eltype(eltype(old_existing_matrix_blocks)), new_outer_shape)

  for old_sector in (old_outer_block_sectors)  # race condition: cannot parallelize
    i_sec = findfirst(==(old_sector), old_existing_sectors)  # TODO
    old_row_range = find_block_range(old_domain_fused_axes, old_domain_block, old_sector)
    old_col_range = find_block_range(
      old_codomain_fused_axes, old_codomain_block, old_sector
    )
    old_sym_block_matrix = view(
      old_existing_matrix_blocks[i_sec], old_row_range, old_col_range
    )  # TODO
    old_tensor_shape = (
      block_structural_multiplicity(old_domain_fused_axes, old_domain_block, old_sector),
      old_domain_ext_mult...,
      block_structural_multiplicity(
        old_codomain_fused_axes, old_codomain_block, old_sector
      ),
      old_codomain_ext_mult...,
    )
    old_sym_block_tensor = reshape(old_sym_block_matrix, old_tensor_shape)
    unitary_column = unitary[BlockArrays.Block(i_old_sector), :]  # take all new blocks at once
    new_outer_block_array += change_basis_block_sector(
      old_sym_block_tensor, unitary_column, extended_perm
    )
  end
  return new_outer_block_array
end

function write_new_outer_block!(
  new_data_matrix,
  new_outer_block_array,
  new_outer_block,
  new_domain_fused_axes,
  new_codomain_fused_axes,
)
  for new_sector in new_allowed_sectors  # not worth parallelize
    new_row_range = find_block_range(new_domain_fused_axes, new_domain_block, i_sec)
    new_col_range = find_block_range(new_codomain_fused_axes, new_codomain_block, i_sec)
    new_sym_block_sector = slice_new_sym_block(new_sym_block, new_sector)
    new_data_matrix[new_sector_index][new_row_range, new_col_range] = new_sym_block_sector
  end
end
