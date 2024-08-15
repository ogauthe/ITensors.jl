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

function clebsch_gordan_tensor(s1::Sectors.O2, s2::Sectors.O2, s3::Sectors.O2, ::Int)
  return clebsch_gordan_tensor(s1, s2, s3)  # no inner multiplicity
end

function clebsch_gordan_tensor(s1::Sectors.O2, s2::Sectors.O2, s3::Sectors.O2)
  d1 = Sectors.quantum_dimension(s1)
  d2 = Sectors.quantum_dimension(s2)
  d3 = Sectors.quantum_dimension(s3)
  cgt = zeros((d1, d2, d3))
  s3 ∉ GradedAxes.blocklabels(GradedAxes.fusion_product(s1, s2)) && return cgt

  # adapted from TensorKit
  l1 = Sectors.category_label(s1)
  l2 = Sectors.category_label(s2)
  l3 = Sectors.category_label(s3)
  if l3 <= 0  # 0even or 0odd
    if l1 <= 0 && l2 <= 0
      cgt[1, 1, 1, 1] = 1.0
    else
      if Sectors.istrivial(s3)
        cgt[1, 2, 1, 1] = 1.0 / sqrt(2)
        cgt[2, 1, 1, 1] = 1.0 / sqrt(2)
      else
        cgt[1, 2, 1, 1] = 1.0 / sqrt(2)
        cgt[2, 1, 1, 1] = -1.0 / sqrt(2)
      end
    end
  elseif l1 <= 0  # 0even or 0odd
    cgt[1, 1, 1, 1] = 1.0
    cgt[1, 2, 2, 1] = s1 == Sectors.zero_odd(Sectors.O2) ? -1.0 : 1.0
  elseif l2 == 0
    cgt[1, 1, 1, 1] = 1.0
    cgt[2, 1, 2, 1] = s2 == Sectors.zero_odd(Sectors.O2) ? -1.0 : 1.0
  elseif l3 == l1 + l2
    cgt[1, 1, 1, 1] = 1.0
    cgt[2, 2, 2, 1] = 1.0
  elseif l3 == l1 - l2
    cgt[1, 2, 1, 1] = 1.0
    cgt[2, 1, 2, 1] = 1.0
  elseif l3 == l2 - l1
    cgt[2, 1, 1, 1] = 1.0
    cgt[1, 2, 2, 1] = 1.0
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
