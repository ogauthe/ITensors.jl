# This file defines SymmetryStyle, a trait to distinguish abelian groups, non-abelian groups
# and non-group fusion categories.

using ..LabelledNumbers: LabelledInteger, label

abstract type SymmetryStyle end

struct AbelianGroup <: SymmetryStyle end
struct NonAbelianGroup <: SymmetryStyle end
struct NonGroupCategory <: SymmetryStyle end
struct EmptyCategoryStyle <: SymmetryStyle end  # CategoryProduct with zero category inside

combine_styles(::AbelianGroup, ::AbelianGroup) = AbelianGroup()
combine_styles(::AbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::AbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonAbelianGroup, ::AbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonAbelianGroup) = NonAbelianGroup()
combine_styles(::NonAbelianGroup, ::NonGroupCategory) = NonGroupCategory()
combine_styles(::NonGroupCategory, ::SymmetryStyle) = NonGroupCategory()
combine_styles(::NonGroupCategory, ::EmptyCategoryStyle) = NonGroupCategory()
combine_styles(::EmptyCategoryStyle, s::SymmetryStyle) = s
combine_styles(s::SymmetryStyle, ::EmptyCategoryStyle) = s
combine_styles(::EmptyCategoryStyle, ::EmptyCategoryStyle) = EmptyCategoryStyle()

SymmetryStyle(l::LabelledInteger) = SymmetryStyle(label(l))

# crash for empty g. Currently impossible to construct.
SymmetryStyle(g::AbstractUnitRange) = SymmetryStyle(first(g))
