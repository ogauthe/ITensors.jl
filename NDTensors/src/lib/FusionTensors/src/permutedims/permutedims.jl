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
  #permuted_data_matrix = permute_data(ft, structural_data)

  #new_codomain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(1)])
  #new_domain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(2)])
  #permuted = FusionTensor(permuted_data_matrix, new_codomain_legs, new_domain_legs)
  return permuted
end

function naive_permutedims(ft::FusionTensor, biperm::TensorAlgebra.BlockedPermutation)
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
function permute_data(ft::FusionTensor, structural_data::StructuralData)
  new_codomain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(1)])
  new_domain_legs = getindex.(Ref(axes(ft)), biperm[BlockArrays.Block(2)])
  new_data_matrix = initialize_data_matrix(eltype(ft), new_codomain_legs, new_domain_legs)

  # TODO

  return new_data_matrix
end
