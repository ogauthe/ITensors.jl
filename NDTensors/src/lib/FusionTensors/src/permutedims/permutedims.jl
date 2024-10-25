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
    BlockSparseArrays.BlockSparseMatrix,
    LinearAlgebra.Adjoint{<:Number,<:BlockSparseArrays.BlockSparseMatrix},
  },
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  biperm::TensorAlgebra.BlockedPermutation{2},
)
  # TBD use FusedAxes as input?

  unitaries = compute_unitaries(  # TODO cache me
    old_domain_legs,
    old_codomain_legs,
    new_domain_legs,
    new_codomain_legs,
    flat_permutation,
  )

  new_domain_legs, new_codomain_legs = TensorAlgebra.blockpermute(axes(ft), biperm)
  new_data_matrix = initialize_data_matrix(
    eltype(old_data_matrix), new_domain_legs, new_codomain_legs
  )

  fill_data_matrix!(
    new_data_matrix,
    old_data_matrix,
    old_domain_legs,
    old_codomain_legs,
    new_domain_legs,
    new_codomain_legs,
    flat_permutation,
    unitaries,
  )
  return new_data_matrix
end

# =====================================  Internals  ========================================
function fill_data_matrix!(
  new_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_data_matrix::Union{
    BlockSparseArrays.BlockSparseMatrix,
    LinearAlgebra.Adjoint{<:Number,<:BlockSparseArrays.BlockSparseMatrix},
  },
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  flat_permutation::Tuple{Vararg{Int}},
  unitaries::Dict,
)
  #=
    for old_block in get_old_block_indices(structural_data)  # TODO PARALLELIZE ME
      isempty(old_block) && continue
      new_sym_block = initialize_new_sym_block(old_block)
      block_unitary = unitaries[old_block]

      for old_sector in existing_old_sectors  # race condition: cannot parallelize
        iso_block_sector = iso_block[old_sector, :]  # take all new blocks at once
        old_sym_block_sector = old_data_matrix[old_sector_index][r1:r2, c1:c2]  # TODO strided view
        new_sym_block += change_basis_block_sector(
          old_sym_block_sector, iso_block_sector, flat_permutation
        )
      end

      for new_sector in allowed_new_sectors  # not worth parallelize
        new_sym_block_sector = slice_new_sym_block(new_sym_block, new_sector)
        new_data_matrix[new_sector_index][r1:r2, c1:c2] = new_sym_block_sector
      end
    end
    =#

  existing_sectors, old_existing_blocks = find_existing_blocks(old_data_matrix)
  old_existing_matrix_blocks = [view(old_data_matrix, b) for b in old_existing_blocks]

  # TODO cache FusedAxes inside FusionTensor
  old_domain_fused_axes = FusedAxes(old_domain_legs)
  old_codomain_fused_axes = FusedAxes(old_codomain_legs)
  new_domain_fused_axes = FusedAxes(new_domain_legs)
  new_codomain_fused_axes = FusedAxes(new_codomain_legs)

  # loop for each codomain irrep configuration
  for old_iter_co in old_codomain_fused_axes
    #old_codomain_block_irreps = getindex.(old_codomain_irreps, old_iter_co)
    old_codomain_block_existing_sectors = intersect(codomain_block_irreps, existing_sectors)
    isempty(old_codomain_block_existing_sectors) && continue

    # loop for each domain irrep configuration
    for old_iter_do in old_domain_fused_axes
      old_block_existing_sectors = intersect(
        getindex.(old_domain_irreps, old_iter_do), old_codomain_block_existing_sectors
      )
      isempty(old_block_existing_sectors) && continue

      #domain_block_length = prod(getindex.(old_domain_degens, iter_do))

      old_iter = (old_iter_do..., old_iter_co...)
      new_iter = map(i -> old_iter[i], flat_permutation)
      new_iter_do = new_iter[begin:length(new_domain_legs)]
      new_iter_co = new_iter[begin:length(new_codomain_legs)]
      block_unitary = unitaries[old_iter]

      new_outer_block = zeros(eltype(old_data_matrix), blah_shape)

      # loop for each symmetry sector inside this configuration
      for i_sec in findall(in(block_existing_sectors), existing_sectors)
        old_row_range = find_block_range(
          old_domain_fused_axes, old_iter_do, GradedAxes.dual(existing_sectors[i_sec])
        )
        old_col_range = find_block_range(
          old_codomain_fused_axes, old_iter_co, existing_sectors[i_sec]
        )
        old_sym_block = view(existing_matrix_blocks[i_sec], old_row_range, old_col_range)
        blah_old = reshape(old_sym_block, blah_shape)
        new_outer_block += block_unitary[:, BlockArrays.Block(blah_index)] * blah_old

        add_sector_symmetric_block!(
          new_outer_block,
          old_sym_block,
          outer_block_degens,
          data_perm,
          old_row_sectors_struct_mult[i_sector],
          old_col_sectors_struct_mult[i_sector],
          old_row_ext_mult,
          old_col_ext_mult,
          old_row_sector_matrix_indices[i_sector],
          old_col_sector_matrix_indices[i_sector],
          unitary_row_sectors[i_sector],
        )
      end
    end
  end
end
