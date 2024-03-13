using BlockArrays: blocks

#using NDTensors.FusionTensors: FusionTensor
using NDTensors.TensorAlgebra: BlockedPermutation, blockedperm

struct StructuralData{
  N,
  NCoAxesIn,
  NDoAxesIn,
  NCoAxesOut,
  NDoAxesOut,
  G<:GradedUnitRange,
  B<:Tuple{NTuple{NCoAxesOut},NTuple{NDoAxesOut}},
  P<:BlockedPermutation{2,N,B},
}
  _permutation::P
end

# constructors
function StructuralData(
  codomain_axes_in::CoDomainAxes, domain_axes_in::DomainAxes, perm::P
) where {
  N,
  NCoAxesIn,
  NDoAxesIn,
  NCoAxesOut,
  NDoAxesOut,
  G<:GradedUnitRange,
  CoDomainAxes<:NTuple{NCoAxesIn,G},
  DomainAxes<:NTuple{NDoAxesIn,G},
  B<:Tuple{NTuple{NCoAxesOut},NTuple{NDoAxesOut}},
  P<:BlockedPermutation{2,N,B},
}
  # TODO impose constraint NCoAxesIn + NDoAxesIn = N
  # perm imposes it for Out
  return StructuralData{N,NCoAxesIn,NDoAxesIn,NCoAxesOut,NDoAxesOut,G,B,P}(perm)
end

# getters
permutation(sd::StructuralData) = sd._permutation
Base.ndims(::StructuralData{N}) where {N} = N
n_codomain_axes_in(::StructuralData{N,NCoAxesIn}) where {N,NCoAxesIn} = NCoAxesIn

function n_domain_axes_in(
  ::StructuralData{N,NCoAxesIn,NDoAxesIn}
) where {N,NCoAxesIn,NDoAxesIn}
  return NDoAxesIn
end

function n_codomain_axes_out(
  ::StructuralData{N,NCoAxesIn,NDoAxesIn,NCoAxesOut}
) where {N,NCoAxesIn,NDoAxesIn,NCoAxesOut}
  return NCoAxesOut
end

function n_domain_axes_out(
  ::StructuralData{N,NCoAxesIn,NDoAxesIn,NCoAxesOut,NDoAxesOut}
) where {N,NCoAxesIn,NDoAxesIn,NCoAxesOut,NDoAxesOut}
  return NDoAxesOut
end
