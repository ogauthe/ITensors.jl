#
# Ising category
#
# (same fusion rules as su2{2})
#

using HalfIntegers: Half, twice
using ..GradedAxes: GradedAxes

struct Ising <: AbstractCategory
  l::Half{Int}
end

# TODO: Use `Val` dispatch here?
function Ising(s::AbstractString)
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return Ising((a - 1)//2)
  end
  return error("Unrecognized input \"$s\" to Ising constructor")
end

SymmetryStyle(::Type{Ising}) = NotAbelianStyle()

GradedAxes.dual(i::Ising) = i

category_label(i::Ising) = i.l

trivial(::Type{Ising}) = Ising(0)

quantum_dimension(::NotAbelianStyle, i::Ising) = (category_label(i) == 1//2) ? √2 : 1.0

# Fusion rules identical to su2₂
function label_fusion_rule(::Type{Ising}, l1, l2)
  degen, suk_sectors = label_fusion_rule(su2{2}, l1, l2)
  sectors = Ising.(category_label.(suk_sectors))
  return degen, sectors
end

# TODO: Use `Val` dispatch here?
label_to_str(i::Ising) = ("1", "σ", "ψ")[twice(category_label(i)) + 1]

function Base.show(io::IO, f::Ising)
  return print(io, "Ising(", label_to_str(f), ")")
end
