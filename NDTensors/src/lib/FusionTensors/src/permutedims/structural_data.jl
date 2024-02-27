using BlockArrays: blocks

#using NDTensors.FusionTensors: FusionTensor
using NDTensors.TensorAlgebra: BlockedPermutation, blockedperm

struct StructuralData{
  N,
  M,
  K,
  G<:GradedUnitRange,
  B,
  P<:BlockedPermutation{2,N,B} where {B<:Tuple{NTuple{M},NTuple{K}}},
}
  _permutation::P
end

# constructors
function StructuralData(
  axes_in::NTuple{N,G}, perm::P
) where {
  N,G<:GradedUnitRange,P<:BlockedPermutation{2,N,B}
} where {M,K,B<:Tuple{NTuple{M},NTuple{K}}}
  return StructuralData{N,M,K,G,B,P}(perm)
end

# getters
permutation(sd::StructuralData) = sd._permutation

# misc
Base.ndims(::StructuralData{N}) where {N} = N
n_codomain_axes(::StructuralData{N,M}) where {N,M} = M
n_domain_axes(::StructuralData{N,M,K}) where {N,M,K} = K
flatpermutation(sd::StructuralData) = Tuple((permutation(sd)))
