# This file defines permutedims for a FusionTensor

##################################  High level interface  ##################################
# permutedims with 1 tuple of 2 separate tuples
function Base.permutedims(ft::FusionTensor, new_axes::Tuple{NTuple,NTuple})
  return permutedims(ft, new_axes[1], new_axes[2])
end

# permutedims with 2 separate tuples
function Base.permutedims(
  ft::FusionTensor, new_codomain_axes::Tuple, new_domain_axes::Tuple
)
  perm = TensorAlgebra.blockedperm(new_codomain_axes, new_domain_axes)
  return permutedims(ft, perm)
end

# 0-dim case
Base.permutedims(ft::FusionTensor{<:Any,0}, ::TensorAlgebra.BlockedPermutation{2,0}) = ft

function Base.permutedims(
  ft::FusionTensor{<:Any,N}, perm::TensorAlgebra.BlockedPermutation{2,N}
) where {N}

  # early return for identity operation. Do not copy.
  if ndims_codomain(ft) == first(BlockArrays.blocklengths(perm))  # compile time
    if Tuple(perm) == ntuple(identity, N)
      return ft
    end
  end

  structural_data = StructuralData(
    perm,
    GradedAxes.blocklabels.(codomain_axes(ft)),
    GradedAxes.blocklabels.(domain_axes(ft)),
    GradedAxes.isdual.(axes(ft)),
  )
  permuted_data_matrix = permute_data(ft, structural_data)

  codomain_axes_out = getindex.(Ref(axes(ft)), perm[BlockArrays.Block(1)])
  domain_axes_out = getindex.(Ref(axes(ft)), perm[BlockArrays.Block(2)])
  out = FusionTensor(permuted_data_matrix, codomain_axes_out, domain_axes_out)
  return out
end

##################################  Low level interface  ###################################
function permute_data(ft::FusionTensor, structural_data::StructuralData)
  perm = permutation(structural_data)

  # TODO replace with correct implementation
  permuted_data_matrix = naive_permute_data(ft, perm)

  return permuted_data_matrix
end

function naive_permute_data(ft::FusionTensor, perm)
  codomain_axes_out = getindex.(Ref(axes(ft)), perm[BlockArrays.Block(1)])
  domain_axes_out = getindex.(Ref(axes(ft)), perm[BlockArrays.Block(2)])

  # stupid permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(perm))
  ftp = FusionTensor(permuted_arr, codomain_axes_out, domain_axes_out)

  return data_matrix(ftp)
end
