# This file defines permutedims for a FusionTensor

# =================================  High level interface  =================================
# permutedims with 1 tuple of 2 separate tuples
function Base.permutedims(ft::FusionTensor, new_leg_indices::Tuple{NTuple,NTuple})
  return permutedims(ft, new_leg_indices[1], new_leg_indices[2])
end

# permutedims with 2 separate tuples
function Base.permutedims(
  ft::FusionTensor, new_domain_indices::Tuple, new_codomain_indices::Tuple
)
  biperm = TensorAlgebra.blockedperm(new_domain_indices, new_codomain_indices)
  return permutedims(ft, biperm)
end

# 0-dim case
Base.permutedims(ft::FusionTensor{<:Any,0}, ::TensorAlgebra.BlockedPermutation{2,0}) = ft

function Base.permutedims(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy.
  if ndims_domain(ft) == first(BlockArrays.blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  # TODO remove me
  return naive_permutedims(ft, biperm)

  new_domain_legs, new_codomain_legs = TensorAlgebra.blockpermute(axes(ft), biperm)
  permuted_data_matrix = permute_data_matrix(
    data_matrix(ft),
    domain_axes(ft),
    codomain_axes(ft),
    new_domain_legs,
    new_codomain_legs,
    Tuple(biperm),
  )

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
  old_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  flat_permutation::Tuple{Vararg{Int}},
)
  @assert length(old_domain_legs) + length(old_codomain_legs) == length(flat_permutation)
  @assert length(new_domain_legs) + length(new_codomain_legs) == length(flat_permutation)

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
  )
  return new_data_matrix
end

# =====================================  Internals  ========================================
function fill_data_matrix!(
  new_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_domain_legs::Tuple{Vararg{AbstractUnitRange}},
  new_codomain_legs::Tuple{Vararg{AbstractUnitRange}},
  flat_permutation::Tuple{Vararg{Int}},
)
  structural_data = StructuralData(  # TODO cache me
    GradedAxes.blocklabels.(old_domain_legs),
    GradedAxes.blocklabels.(old_codomain_legs),
    GradedAxes.blocklabels.(new_domain_legs),
    GradedAxes.blocklabels.(new_codomain_legs),
    (GradedAxes.isdual.(old_domain_legs)..., GradedAxes.isdual(old_codomain_legs)...),
    flat_permutation,
  )

  for old_block in get_old_block_indices(structural_data)  # TODO PARALLELIZE ME
    new_sym_block = initialize_new_sym_block(old_block)
    iso_block = structural_data[old_block]

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
end
