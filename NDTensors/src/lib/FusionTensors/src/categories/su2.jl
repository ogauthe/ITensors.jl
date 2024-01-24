# This file defines Clebsch-Gordan tensor for Category SU2

using WignerSymbols
using RationalRoots
using NDTensors.Sectors: SU2, label, dimension

"""
Compute Clebsch-Gordan tensor merging s1 ⊗ s2 on s3
"""
function clebschgordantensor(s1::SU2, s2::SU2, s3::SU2)
  s1l = label(s1)
  s2l = label(s2)
  s3l = label(s3)
  shape = (dimension(s1), dimension(s2), dimension(s3))
  cgtensor = Array{RationalRoots.RationalRoot{BigInt},3}(undef, shape)
  for i in 1:dimension(s1)
    j1 = -s1l + i - 1
    for j in 1:dimension(s2)
      j2 = -s2l + j - 1
      for k in 1:dimension(s3)
        j3 = -s3l + k - 1
        cgtensor[i, j, k] = clebschgordan(s1l, j1, s2l, j2, s3l, j3)
      end
    end
  end
  return cgtensor
end
