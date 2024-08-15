# This file defines Clebsch-Gordan tensors
# one tensor is defined from 3 simple objects s1, s2 and s3
# and contains the coefficients fusing s1 ⊗ s2 -> s3

function symbol_1j(s::Sectors.AbstractCategory)
  cgt = clebsch_gordan_tensor(s, GradedAxes.dual(s), Sectors.trivial(s), 1)
  return sqrt(Sectors.quantum_dimension(s)) * cgt[:, :, 1]
end

function clebsch_gordan_tensor(
  s1::Sectors.AbstractCategory,
  s2::Sectors.AbstractCategory,
  s3::Sectors.AbstractCategory,
  arrow1::Bool,
  arrow2::Bool,
  inner_mult_index::Int,
)
  cgt = clebsch_gordan_tensor(s1, s2, s3, inner_mult_index)
  if arrow1
    flip1 = symbol_1j(s1)
    cgt = TensorAlgebra.contract((1, 2, 3), flip1, (4, 1), cgt, (4, 2, 3))
  end
  if arrow2
    flip2 = symbol_1j(s2)
    cgt = TensorAlgebra.contract((1, 2, 3), flip2, (4, 2), cgt, (1, 4, 3))
  end
  return cgt
end

function clebsch_gordan_tensor(
  s1::Sectors.SU{2}, s2::Sectors.SU{2}, s3::Sectors.SU{2}, ::Int
)
  return clebsch_gordan_tensor(s1, s2, s3)  # no inner multiplicity
end

function clebsch_gordan_tensor(s1::Sectors.SU{2}, s2::Sectors.SU{2}, s3::Sectors.SU{2})
  d1 = Sectors.quantum_dimension(s1)
  d2 = Sectors.quantum_dimension(s2)
  d3 = Sectors.quantum_dimension(s3)
  j1 = HalfIntegers.half(d1 - 1)
  j2 = HalfIntegers.half(d2 - 1)
  j3 = HalfIntegers.half(d3 - 1)
  cgtensor = Array{Float64,3}(undef, (d1, d2, d3))
  for (i, j, k) in Iterators.product(1:d1, 1:d2, 1:d3)
    m1 = j1 - i + 1
    m2 = j2 - j + 1
    m3 = j3 - k + 1
    cgtensor[i, j, k] = WignerSymbols.clebschgordan(j1, m1, j2, m2, j3, m3)
  end
  return cgtensor
end

function clebsch_gordan_tensor(
  s1::Sectors.SU{3}, s2::Sectors.SU{3}, s3::Sectors.SU{3}, inner_mult_index::Int
)
  d1 = Sectors.quantum_dimension(s1)
  d2 = Sectors.quantum_dimension(s2)
  d3 = Sectors.quantum_dimension(s3)
  cgtensor = zeros(d1, d2, d3)
  # dummy
  return cgtensor
end
