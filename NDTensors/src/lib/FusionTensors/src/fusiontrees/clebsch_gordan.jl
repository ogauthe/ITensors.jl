# This file defines Clebsch-Gordan tensors
# one tensor is defined from 3 simple objects s1, s2 and s3
# and contains the coefficients fusing s1 ⊗ s2 -> s3

using WignerSymbols: clebschgordan

# TBD how to deal with fusion ring inner degeneracies
# always return a rank-4 tensor?
# TBD move this function into Sectors?
function clebsch_gordan_tensor(c1::C, c2::C, c3::C) where {C<:Sectors.AbstractCategory}
  return clebsch_gordan_tensor(Sectors.SymmetryStyle(c1), c1, c2, c3)
end

function clebsch_gordan_tensor(s1::C, s2::C, s3::C) where {C<:Sectors.CategoryProduct}
  c1 = Tuple(Sectors.categories(s1))
  c2 = Tuple(Sectors.categories(s2))
  c3 = Tuple(Sectors.categories(s3))
  cats_cg = clebsch_gordan_tensor.(c1, c2, c3)
  cgt = reduce(_tensor_kron, cats_cg)
  return cgt
end

function clebsch_gordan_tensor(::Sectors.AbelianGroup, s1, s2, s3)
  return s1 ⊗ s2 == s3 ? ones((1, 1, 1)) : zeros((1, 1, 1))
end

function clebsch_gordan_tensor(
  ::Sectors.NonAbelianGroup, s1::Sectors.SU{2}, s2::Sectors.SU{2}, s3::Sectors.SU{2}
)
  d1 = Sectors.quantum_dimension(s1)
  d2 = Sectors.quantum_dimension(s2)
  d3 = Sectors.quantum_dimension(s3)
  j1 = HalfIntegers.half(d1 - 1)
  j2 = HalfIntegers.half(d2 - 1)
  j3 = HalfIntegers.half(d3 - 1)
  cgtensor = Array{Float64,3}(undef, (d1, d2, d3))
  for i in 1:d1
    m1 = j1 - i + 1
    for j in 1:d2
      m2 = j2 - j + 1
      for k in 1:d3
        m3 = j3 - k + 1
        cgtensor[i, j, k] = clebschgordan(j1, m1, j2, m2, j3, m3)
      end
    end
  end
  return cgtensor
end

# LinearAlgebra.kron does not allow input with ndims=3
function _tensor_kron(a, b)
  sha = ntuple(i -> Bool(i % 2) ? size(a, i ÷ 2 + 1) : 1, 2 * ndims(a))
  shb = ntuple(i -> Bool(i % 2) ? 1 : size(b, i ÷ 2), 2 * ndims(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end
