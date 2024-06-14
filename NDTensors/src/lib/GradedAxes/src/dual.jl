function dual end

using NDTensors.LabelledNumbers:
  LabelledStyle, IsLabelled, NotLabelled, label, labelled, unlabel
label_dual(x) = label_dual(LabelledStyle(x), x)
label_dual(::NotLabelled, x) = x
label_dual(::IsLabelled, x) = labelled(unlabel(x), dual(label(x)))

# TBD rename deepdual? yet another name?
label_dual(g::AbstractGradedUnitRange) = gradedrange(label_dual.(blocklengths(g)))
