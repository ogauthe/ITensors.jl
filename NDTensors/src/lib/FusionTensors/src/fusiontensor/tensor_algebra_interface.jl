# This file defines TensorAlgebra interface for a FusionTensor

# TBD define output_axes or directly allocate_output, or both?
"""
function TensorAlgebra.output_axes(
  ::typeof(TensorAlgebra.contract),
  biperm_dest::BlockedPermutation{2},
  a1::FusionTensor,
  biperm1::BlockedPermutation{2},
  a2::FusionTensor,
  biperm2::BlockedPermutation{2},
  α::Number=true,
)
  println(biperm_dest)
  println(biperm1)
  println(biperm2)
  axes_codomain, axes_contracted = TensorAlgebra.blockpermute(axes(a1), biperm1)
  axes_contracted2, axes_domain = TensorAlgebra.blockpermute(axes(a2), biperm2)
  @assert axes_contracted == axes_contracted2
  x = TensorAlgebra.genperm(
    (axes_codomain..., axes_domain...), TensorAlgebra.invperm(Tuple(biperm_dest))
  )
  println("111111  ", axes_codomain)
  println("222222  ", axes_domain)

  println(x, typeof(x))
  y1 = codomain_axes(a1)
  y2 = domain_axes(a2)
  return y1, y2
end
"""

# TBD how to deal with inner contraction = no ouput axis?
function TensorAlgebra.allocate_output(
  ::typeof(contract),
  biperm_dest::BlockedPermutation{2},
  a1::FusionTensor{T1,N},
  biperm1::BlockedPermutation{2,N},
  a2::FusionTensor{T2,M},
  biperm2::BlockedPermutation{2,M},
  α::Number=true,
) where {T1,T2,N,M}
  axes_dest = (
    (i -> axes(a1)[i]).(biperm1[Block(1)]), (i -> axes(a2)[i]).(biperm2[Block(2)])
  )
  return similar(a1, promote_type(eltype(a1), eltype(a2), typeof(α)), axes_dest)
end

# TBD do really I need to defined these as I cannot use them in contract! and has to redefine it?
# TensorAlgebra.fusedims(ft::FusionTensor, perm::BlockedPermutation) = permutedims(ft, perm)
# function TensorAlgebra.splitdims(ft1::FusionTensor, ft2::FusionTensor, blockedperm::BlockedPermutation)
# function TensorAlgebra.splitdims!(ft1::FusionTensor, ft2::FusionTensor, blockedperm::BlockedPermutation)

# I cannot use contract! from TensorAlgebra/src/contract/contract_matricize/contract.jl
# as it calls _mul!, which I should not overload.
# TBD I can also overload higher up and do not allow use of different algorithms
function TensorAlgebra.contract!(
  alg::TensorAlgebra.Algorithm{:matricize},
  # TBD replace with
  #  alg::TensorAlgebra.Algorithm"matricize",
  a_dest::FusionTensor,
  biperm_dest::BlockedPermutation,
  a1::FusionTensor,
  biperm1::BlockedPermutation,
  a2::FusionTensor,
  biperm2::BlockedPermutation,
  α::Number,
  β::Number,
)
  a1_perm = permutedims(a1, biperm1)
  a2_perm = permutedims(a2, biperm2)
  LinearAlgebra.mul!(a_dest, a1_perm, a2_perm, α, β)
  return a_dest
end
