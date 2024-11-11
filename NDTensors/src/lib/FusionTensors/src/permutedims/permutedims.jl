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
  biperm = blockedperm(new_domain_indices, new_codomain_indices)
  return permutedims(ft, biperm)
end

function fusiontensor_permutedims(ft::FusionTensor, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)

  # early return for identity operation. Do not copy. Also handle tricky 0-dim case.
  if ndims_domain(ft) == first(blocklengths(biperm))  # compile time
    if Tuple(biperm) == ntuple(identity, ndims(ft))
      return ft
    end
  end

  permuted_data_matrix = permute_data_matrix(
    data_matrix(ft), domain_axes(ft), codomain_axes(ft), biperm
  )
  new_domain_legs, new_codomain_legs = blockpermute(axes(ft), biperm)
  permuted = FusionTensor(permuted_data_matrix, new_domain_legs, new_codomain_legs)
  return permuted
end

function naive_permutedims(ft::FusionTensor, biperm::BlockedPermutation{2})
  @assert ndims(ft) == length(biperm)
  new_domain_legs, new_codomain_legs = blockpermute(axes(ft), biperm)

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
  old_domain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  old_codomain_legs::Tuple{Vararg{AbstractGradedUnitRange}},
  biperm::BlockedPermutation{2},
)
  # TBD use FusedAxes as input?

  unitaries = compute_unitaries(  # TODO cache me
    old_domain_legs,
    old_codomain_legs,
    biperm,
  )

  # TODO cache FusedAxes
  old_domain_fused_axes = FusedAxes(old_domain_legs)
  old_codomain_fused_axes = FusedAxes(dual.(old_codomain_legs))
  new_domain_legs, new_codomain_legs = blockpermute(
    (old_domain_legs..., old_codomain_legs...), biperm
  )
  new_domain_fused_axes = FusedAxes(new_domain_legs)
  new_codomain_fused_axes = FusedAxes(dual.(new_codomain_legs))
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
function add_structural_axes(
  biperm::BlockedPermutation{2}, ::Val{OldNDoAxes}
) where {OldNDoAxes}
  flat = Tuple(biperm)
  extended_perm = (
    OldNDoAxes + 1, length(biperm) + 2, (flat .+ (flat .>= OldNDoAxes + 1))...
  )
  return extended_perm
end

function fill_data_matrix!(
  new_data_matrix::AbstractBlockSparseMatrix,
  old_data_matrix::Union{
    AbstractBlockSparseMatrix,Adjoint{<:Number,<:AbstractBlockSparseMatrix}
  },
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  new_domain_fused_axes::FusedAxes,
  new_codomain_fused_axes::FusedAxes,
  biperm::BlockedPermutation{2},
  unitaries::Dict,
)
  @assert ndims(old_domain_fused_axes) + ndims(old_codomain_fused_axes) == length(biperm)
  @assert ndims(new_domain_fused_axes) + ndims(new_codomain_fused_axes) == length(biperm)

  old_matrix_block_indices = reinterpret(
    Tuple{Int,Int}, sort(collect(block_stored_indices(old_data_matrix)))
  )
  old_matrix_blocks = Dict(
    blocklabels(old_domain_fused_axes)[first(b)] => view(old_data_matrix, Block(b)) for
    b in old_matrix_block_indices
  )

  new_matrix_block_indices = intersect(new_domain_fused_axes, new_codomain_fused_axes)
  new_matrix_blocks = Dict(
    blocklabels(new_domain_fused_axes)[first(b)] => view!(new_data_matrix, Block(b)) for
    b in new_matrix_block_indices
  )

  old_existing_outer_blocks = allowed_outer_blocks_sectors(
    old_domain_fused_axes, old_codomain_fused_axes, old_matrix_block_indices
  )

  new_existing_outer_blocks = allowed_outer_blocks_sectors(
    new_domain_fused_axes, new_codomain_fused_axes, new_matrix_block_indices
  )

  extended_perm = add_structural_axes(biperm, Val(ndims(old_domain_fused_axes)))

  # loop for each existing outer block TODO parallelize
  for old_outer_block_sectors in old_existing_outer_blocks
    new_outer_block = map(i -> first(old_outer_block_sectors)[i], Tuple(biperm))
    new_outer_block_sectors = new_outer_block => new_existing_outer_blocks[new_outer_block]
    write_new_outer_block!(
      new_matrix_blocks,
      old_matrix_blocks,
      old_outer_block_sectors,
      new_outer_block_sectors,
      unitaries[first(old_outer_block_sectors)],
      extended_perm,
      old_domain_fused_axes,
      old_codomain_fused_axes,
      new_domain_fused_axes,
      new_codomain_fused_axes,
    )
  end
end

function write_new_outer_block!(
  new_matrix_blocks::Dict{S,<:AbstractMatrix},
  old_matrix_blocks::Dict{S,<:AbstractMatrix},
  old_outer_block_sectors::Pair{<:Tuple{Vararg{Int}},Vector{S}},
  new_outer_block_sectors::Pair{<:Tuple{Vararg{Int}},Vector{S}},
  unitary::AbstractBlockMatrix,
  extended_perm::Tuple{Vararg{Int}},
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  new_domain_fused_axes::FusedAxes,
  new_codomain_fused_axes::FusedAxes,
) where {S<:AbstractSector}
  new_outer_array = permute_outer_block(
    old_matrix_blocks,
    old_outer_block_sectors,
    old_domain_fused_axes,
    old_codomain_fused_axes,
    extended_perm,
    unitary,
  )
  return write_new_outer_block!(
    new_matrix_blocks,
    new_outer_array,
    new_outer_block_sectors,
    new_domain_fused_axes,
    new_codomain_fused_axes,
  )
end

function permute_outer_block(
  old_matrix_blocks::Dict{S,<:AbstractMatrix},
  old_outer_block_sectors::Pair{<:Tuple{Vararg{Int}},Vector{S}},
  old_domain_fused_axes::FusedAxes,
  old_codomain_fused_axes::FusedAxes,
  extended_perm::Tuple{Vararg{Int}},
  unitary::AbstractBlockMatrix,
) where {S<:AbstractSector}
  old_domain_block = first(old_outer_block_sectors)[begin:ndims(old_domain_fused_axes)]
  old_codomain_block = first(old_outer_block_sectors)[(ndims(old_domain_fused_axes) + 1):end]
  old_domain_ext_mult = block_external_multiplicities(
    old_domain_fused_axes, old_domain_block
  )
  old_codomain_ext_mult = block_external_multiplicities(
    old_codomain_fused_axes, old_codomain_block
  )
  return mapreduce(+, enumerate(last(old_outer_block_sectors))) do (i_sec, old_sector)
    old_row_range = find_block_range(old_domain_fused_axes, old_domain_block, old_sector)
    old_col_range = find_block_range(
      old_codomain_fused_axes, old_codomain_block, old_sector
    )
    old_sym_block_matrix = view(old_matrix_blocks[old_sector], old_row_range, old_col_range)
    old_tensor_shape = (
      old_domain_ext_mult...,
      block_structural_multiplicity(old_domain_fused_axes, old_domain_block, old_sector),
      old_codomain_ext_mult...,
      block_structural_multiplicity(
        old_codomain_fused_axes, old_codomain_block, old_sector
      ),
    )
    old_sym_block_tensor = reshape(old_sym_block_matrix, old_tensor_shape)
    unitary_column = unitary[:, Block(i_sec)]  # take all new blocks at once
    return change_basis_block_sector(old_sym_block_tensor, unitary_column, extended_perm)
  end
end

function change_basis_block_sector(
  old_sym_block_tensor::AbstractArray,
  unitary_column::AbstractBlockMatrix,
  extended_perm::Tuple{Vararg{Int}},
)
  old_permuted = permutedims(old_sym_block_tensor, extended_perm)
  new_shape = (size(unitary_column, 2), prod(size(old_permuted)[3:end]))
  reshaped = reshape(old_permuted, new_shape)
  new_outer_array = unitary_column * reshaped
  return new_outer_array
end

function write_new_outer_block!(
  new_matrix_blocks::Dict{S,<:AbstractMatrix},
  new_outer_array::AbstractMatrix,
  new_outer_block_sectors::Pair{<:Tuple{Vararg{Int}},Vector{S}},
  new_domain_fused_axes::FusedAxes,
  new_codomain_fused_axes::FusedAxes,
) where {S<:AbstractSector}
  new_domain_block = first(new_outer_block_sectors)[begin:ndims(new_domain_fused_axes)]
  new_codomain_block = first(new_outer_block_sectors)[(ndims(new_domain_fused_axes) + 1):end]
  for (i_sec, new_sector) in enumerate(last(new_outer_block_sectors))  # not worth parallelize
    new_row_range = find_block_range(new_domain_fused_axes, new_domain_block, new_sector)
    new_col_range = find_block_range(
      new_codomain_fused_axes, new_codomain_block, new_sector
    )
    new_matrix_blocks[new_sector][new_row_range, new_col_range] = view(
      new_outer_array, Block(i_sec)
    )
  end
end
