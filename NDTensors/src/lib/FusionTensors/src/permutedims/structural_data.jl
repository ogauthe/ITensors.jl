struct StructuralData{P,NCoAxesIn,NDoAxesIn,C}
  permutation::P

  # inner constructor to impose constraints on types
  function StructuralData(
    perm::TensorAlgebra.BlockedPermutation{2,N},
    codomain_sectors_in::NTuple{NCoAxesIn,Vector{C}},
    domain_sectors_in::NTuple{NDoAxesIn,Vector{C}},
    arrow_directions::NTuple{N,Bool},
  ) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
    if NCoAxesIn + NDoAxesIn != N
      return error("permutation incompatible with axes")
    end
    return new{typeof(perm),NCoAxesIn,NCoAxesIn,C}(perm)
  end
end

# getters
permutation(sd::StructuralData) = sd.permutation

function Base.ndims(::StructuralData{P,NCoAxesIn,NDoAxesIn}) where {P,NCoAxesIn,NDoAxesIn}
  return NCoAxesIn + NDoAxesIn
end
ndims_codomain_in(::StructuralData{P,NCoAxesIn}) where {P,NCoAxesIn} = NCoAxesIn
function ndims_domain_in(
  ::StructuralData{P,NCoAxesIn,NDoAxesIn}
) where {P,NCoAxesIn,NDoAxesIn}
  return NDoAxesIn
end

function ndims_codomain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[1]
end

function ndims_domain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[2]
end
