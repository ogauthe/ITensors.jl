struct StructuralData{NCoAxesIn,NDoAxesIn,P<:TensorAlgebra.BlockedPermutation}
  permutation::P
end

# constructors
function StructuralData(
  codomain_axes_in::Tuple, domain_axes_in::Tuple, perm::TensorAlgebra.BlockedPermutation
)
  # TODO impose constraint NCoAxesIn + NDoAxesIn = N
  # perm imposes it for Out
  return StructuralData{length(codomain_axes_in),length(domain_axes_in),typeof(perm)}(perm)
end

# getters
permutation(sd::StructuralData) = sd.permutation
function Base.ndims(::StructuralData{NCoAxesIn,NDoAxesIn}) where {NCoAxesIn,NDoAxesIn}
  return NCoAxesIn + NDoAxesIn
end
ndims_codomain_in(::StructuralData{NCoAxesIn}) where {NCoAxesIn} = NCoAxesIn

function ndims_codomain_out(sd::StructuralData)
  return length(permutation(sd)[1])
end

function ndims_domain_in(::StructuralData{NCoAxesIn,NDoAxesIn}) where {NCoAxesIn,NDoAxesIn}
  return NDoAxesIn
end

function ndimsodomain_out(sd::StructuralData)
  return length(permutation(sd)[2])
end
