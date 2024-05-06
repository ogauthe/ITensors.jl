# This file defines Clebsch-Gordan tensors
# one tensor is defined from 3 simple objects s1, s2 and s3
# and contains the coefficients fusing s1 ⊗ s2 -> s3

using WignerSymbols: clebschgordan

clebsch_gordan_tensor(s1, s2, s3) = clebsch_gordan_tensor(s1, s2, s3, false, false, 1)

function clebsch_gordan_tensor(
  s1::Sectors.SU{2},
  s2::Sectors.SU{2},
  s3::Sectors.SU{2},
  isdual1::Bool,
  isdual2::Bool,
  ::Int,
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
  if isdual1
    diag1 = reshape((-1) .^ collect(((d1 % 2) + 1):(d1 % 2 + d1)), (d1, 1, 1))
    cgtensor = diag1 .* reverse(cgtensor; dims=1)
  end
  if isdual2
    diag2 = reshape((-1) .^ collect(((d2 % 2) + 1):(d2 % 2 + d2)), (1, d2, 1))
    cgtensor = diag2 .* reverse(cgtensor; dims=2)
  end

  return cgtensor
end

function clebsch_gordan_tensor(
  s1::Sectors.SU{3},
  s2::Sectors.SU{3},
  s3::Sectors.SU{3},
  isdual1::Bool,
  isdual2::Bool,
  inner_degen_index::Int,
)
  d1 = Sectors.quantum_dimension(s1)
  d2 = Sectors.quantum_dimension(s2)
  d3 = Sectors.quantum_dimension(s3)
  cgtensor = zeros(d1, d2, d3)
  # dummy
  return cgtensor
end
