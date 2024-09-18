# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD: Unitary format
#      * BlockMatrix?
#      * 4-dim BlockSparseArray?
#      * other?

# TBD: cache format
#       * global Dict of Dict{(N,C,OldNDo,OldNCo,NewNDo,NewNCo,OldArrows,flatperm), Dict}
#          + unitary Dict{NTuple{C<:AbstractCategory}, Unitary}

# TBD: inner structure of a matrix block
#       * (struct, ext) or its transpose

# TBD: cache of FusionTensor inner structure
#       * cache in FT (TensorKit choice)
#       * cache in StructuralData  (froSTspin choice)
#       * no cache

# Current implementation:
# * Unitary = BlockMatrix
# * no unitary cache
# * inner structure = (struct, ext)
# * no cache of internal structure
#

struct StructuralData{N,C,OldNDo,OldNCo,NewNDo,NewNCo,Unitaries}
  old_domain_labels::NTuple{OldNDo,Vector{C}}
  old_codomain_labels::Tuple{OldNCo,Vector{C}}
  new_domain_labels::Tuple{NewNDo,Vector{C}}
  new_codomain_labels::Tuple{NewNCo,Vector{C}}
  old_arrows::NTuple{N,Bool}
  flat_permutation::NTuple{N,Int}
  unitaries::Unitaries

  function StructuralData(
    old_domain_labels::Tuple{Vararg{Vector{C}}},
    old_codomain_labels::Tuple{Vararg{Vector{C}}},
    new_domain_labels::Tuple{Vararg{Vector{C}}},
    new_codomain_labels::Tuple{Vararg{Vector{C}}},
    old_arrows::NTuple{N,Bool},
    flat_permutation::NTuple{N,Int},
    unitaries,
  ) where {N,C<:Sectors.AbstractCategory}
    @assert length(old_domain_labels) + length(old_codomain_labels) == N
    @assert length(new_domain_labels) + length(new_codomain_labels_codomain_labels) == N
    @assert N > 0

    return new{
      N,
      C,
      length(old_domain_labels),
      length(old_codomain_labels),
      length(new_domain_labels),
      length(new_codomain_labels),
      eltype(unitaries),
    }(
      old_domain_labels,
      old_codomain_labels,
      new_domain_labels,
      new_codomain_labels,
      old_arrows,
      flat_permutation,
      unitaries,
    )
  end
end

function StructuralData(
  old_domain_labels::Tuple{Vararg{Vector{C}}},
  old_codomain_labels::Tuple{Vararg{Vector{C}}},
  new_domain_labels::Tuple{Vararg{Vector{C}}},
  new_codomain_labels::Tuple{Vararg{Vector{C}}},
  old_arrows::NTuple{N,Bool},
  flat_permutation::NTuple{N,Int},
) where {N,C<:Sectors.AbstractCategory}
  unitaries = compute_unitaries_CG(
    old_domain_labels,
    old_codomain_labels,
    new_domain_labels,
    new_codomain_labels,
    old_arrows,
    flat_permutation,
  )
  return StructuralData(
    old_domain_labels,
    old_codomain_labels,
    new_domain_labels,
    new_codomain_labels,
    old_arrows,
    flat_permutation,
    unitaries,
  )
end
