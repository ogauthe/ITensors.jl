# This file defines permutedims for a FusionTensor

##################################  High level interface  ##################################
# permutedims with 1 tuple of 2 separate tuples
function Base.permutedims(ft::FusionTensor, new_leg_indices::Tuple{NTuple,NTuple})
  return permutedims(ft, new_leg_indices[1], new_leg_indices[2])
end

# permutedims with 2 separate tuples
function Base.permutedims(
  ft::FusionTensor, new_codomain_indices::Tuple, new_domain_indices::Tuple
)
  biperm = TensorAlgebra.blockedperm(new_codomain_indices, new_domain_indices)
  return permutedims(ft, biperm)
end

# 0-dim case
Base.permutedims(ft::FusionTensor{<:Any,0}, ::TensorAlgebra.BlockedPermutation{2,0}) = ft

function Base.permutedims(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation)
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy.
  if ndims_codomain(ft) == first(BlockArrays.blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  # TODO remove me
  permuted = naive_permutedims(ft, biperm)

  # TODO
  #structural_data = StructuralData(ft, biperm)  # TODO cache me
  #permuted_data_matrix = permute_data_matrix(ft, structural_data)

  #new_codomain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(1)])
  #new_domain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(2)])
  #permuted = FusionTensor(permuted_data_matrix, new_codomain_legs, new_domain_legs)
  return permuted
end

function naive_permutedims(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  new_codomain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(1)])
  new_domain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(2)])

  # stupid permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(biperm))
  permuted = FusionTensor(permuted_arr, new_codomain_legs, new_domain_legs)

  return permuted
end

##################################  Low level interface  ###################################
function permute_data_matrix(ft::FusionTensor, structural_data::StructuralData)
  biperm = get_biperm(structural_data)
  new_codomain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(1)])
  new_domain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(2)])
  new_data_matrix = initialize_data_matrix(eltype(ft), new_codomain_legs, new_domain_legs)

  fill_data_matrix!(
    new_data_matrix, data_matrix(ft), structural_data, codomain_axes(ft), domain_axes(ft)
  )

  return new_data_matrix
end

# =====================================  Internals  ========================================
function fill_data_matrix!(
  new_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  old_data_matrix::BlockSparseArrays.AbstractBlockSparseMatrix,
  structural_data::StructuralData,
  old_codomain_legs::Tuple,
  old_domain_legs::Tuple,
)
  @assert ndims(new_data_matrix) == ndims(old_data_matrix) == ndims(structural_data)

  for old_block in get_old_block_indices(structural_data)  # TODO PARALLELIZE ME
    new_sym_block = initialize_new_sym_block(old_block)
    iso_block = isometries[old_block]

    for old_sector in existing_old_sectors  # race condition: cannot parallelize
      iso_block_sector = iso_block[old_sector, :]
      old_sym_block_sector = old_data_matrix[old_sector_index][r1:r2, c1:c2]
      new_sym_block += change_basis_block_sector(old_sym_block_sector, iso_block_sector)
    end

    for new_sector in allowed_new_sectors  # not worth parallelize
      new_sym_block_sector = slice_new_sym_block(new_sym_block, new_sector)
      new_data_matrix[new_sector_index][r1:r2, c1:c2] = new_sym_block_sector
    end
  end
end
