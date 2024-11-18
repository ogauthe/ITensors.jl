# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

#
# A fusion tree fuses N sectors sec1, secN  onto one sector fused_sec. A given set of
# sectors and arrow directions (as defined by a given outer block) contains several fusion
# trees that typically fuse to several sectors (in the abelian group case, there is only one)
# irrep in the fusion ring and each of them corresponds to a single "thin" fusion tree with
#
#
#
#             /
#          sec123
#           /\
#          /  \
#       sec12  \
#        /\     \
#       /  \     \
#     sec1 sec2  sec3
#
#
#
#
# convention: irreps are already dualed if needed, arrows do not affect them. They only
# affect the basis on which the tree projects for self-dual irreps.
#
#
# The interface uses AbstractGradedUnitRanges as input for interface simplicity
# however only blocklabels are used and blocklengths are never read.

using NDTensors.LabelledNumbers: LabelledNumbers  # TBD avoid depending on internals?

struct FusionTree{S,N,M}
  base_sectors::NTuple{N,S}
  base_arrows::NTuple{N,Bool}
  fused_sector::S
  level_sectors::NTuple{M,S}  # M = N-1
  level_outer_multiplicities::NTuple{M,Int}  # M = N-1

  # TBD could have level_sectors with length N-2
  # currently first(level_sectors) == first(base_sectors)
  # redundant but allows for simpler, generic grow_tree code

  function FusionTree(
    base_sectors, base_arrows, fused_sector, level_sectors, level_outer_multiplicities
  )
    N = length(base_sectors)
    @assert length(level_sectors) == max(0, N - 1)
    @assert length(level_outer_multiplicities) == max(0, N - 1)
    return new{typeof(fused_sector),length(base_sectors),length(level_sectors)}(
      base_sectors, base_arrows, fused_sector, level_sectors, level_outer_multiplicities
    )
  end
end

# getters
base_arrows(f::FusionTree) = f.base_arrows
base_sectors(f::FusionTree) = f.base_sectors
fused_sector(f::FusionTree) = f.fused_sector
level_sectors(f::FusionTree) = f.level_sectors
level_outer_multiplicities(f::FusionTree) = f.level_outer_multiplicities

# interface
Base.ndims(::FusionTree{<:Any,N}) where {N} = N
Base.isless(f1::FusionTree, f2::FusionTree) = isless(to_tuple(f1), to_tuple(f2))

function to_tuple(f::FusionTree)
  return (
    base_sectors(f)...,
    base_arrows(f)...,
    fused_sector(f),
    level_sectors(f)...,
    level_outer_multiplicities(f)...,
  )
end

LabelledNumbers.label_type(::FusionTree{S}) where {S} = S  # TBD use different function name?

function build_trees(legs::Vararg{AbstractGradedUnitRange{LA}}) where {LA}
  # TBD when to impose LA to be the same for every leg?
  tree_arrows = isdual.(legs)
  sectors = blocklabels.(legs)
  return mapreduce(vcat, CartesianIndices(blocklength.(legs))) do it
    block_sectors = getindex.(sectors, Tuple(it))
    return build_trees(block_sectors, tree_arrows)
  end
end

# TBD is this necessary / useful?
function SymmetrySectors.:×(f1::FusionTree, f2::FusionTree)
  @assert base_arrows(f1) == base_arrows(f2)
  product_base_sectors = .×(base_sectors(f1), base_sectors(f2))
  product_fused_sector = fused_sector(f1) × fused_sector(f2)
  product_level_sectors = .×(level_sectors(f1), level_sectors(f2))
  product_level_outer_multiplicities =
    outer_multiplicity_kron.(
      base_sectors(f1)[2:end],
      level_sectors(f1),
      (level_sectors(f1)[2:end]..., fused_sector(f1)),
      level_outer_multiplicities(f1),
      level_outer_multiplicities(f2),
    )
  return FusionTree(
    product_base_sectors,
    base_arrows(f1),
    product_fused_sector,
    product_level_sectors,
    product_level_outer_multiplicities,
  )
end

function outer_multiplicity_kron(
  sec1, sec2, fused, outer_multiplicity1, outer_multiplicity2
)
  full_space = fusion_product(sec1, sec2)
  nsymbol = blocklengths(full_space)[findfirst(==(fused), blocklabels(full_space))]
  linear_inds = LinearIndices((nsymbol, outer_multiplicity2))
  return linear_inds[outer_multiplicity1, outer_multiplicity2]
end

# zero leg: need S to get sector type information
function FusionTree{S}(::Tuple{}, ::Tuple{}) where {S}
  return FusionTree((), (), trivial(S), (), ())
end

function FusionTree{S}(base_sectors::NTuple{N,S}, base_arrows::NTuple{N,Bool}) where {N,S}
  return build_trees(base_sectors, base_arrows)
end

# one leg
FusionTree(sect::AbstractSector, arrow::Bool) = FusionTree((sect,), (arrow,), sect, (), ())

# =====================================  Internals  ========================================
function grow_tree(
  parent_tree::FusionTree,
  level_sector::AbstractSector,
  level_arrow::Bool,
  child_fused_sector,
  outer_mult,
)
  child_base_sectors = (base_sectors(parent_tree)..., level_sector)
  child_base_arrows = (base_arrows(parent_tree)..., level_arrow)
  child_level_sectors = (level_sectors(parent_tree)..., fused_sector(parent_tree))
  child_outer_mul = (level_outer_multiplicities(parent_tree)..., outer_mult)
  return FusionTree(
    child_base_sectors,
    child_base_arrows,
    child_fused_sector,
    child_level_sectors,
    child_outer_mul,
  )
end

function grow_tree(parent_tree::FusionTree, level_sector::AbstractSector, level_arrow::Bool)
  new_space = fusion_product(fused_sector(parent_tree), level_sector)
  return mapreduce(vcat, zip(blocklabels(new_space), blocklengths(new_space))) do (la, n)
    return [
      grow_tree(parent_tree, level_sector, level_arrow, la, outer_mult) for
      outer_mult in 1:n
    ]
  end
end

function build_trees(old_trees::Vector, sectors_to_fuse::Tuple, arrows_to_fuse::Tuple)
  next_level_trees = mapreduce(vcat, old_trees) do tree
    return grow_tree(tree, first(sectors_to_fuse), first(arrows_to_fuse))
  end
  return build_trees(next_level_trees, sectors_to_fuse[2:end], arrows_to_fuse[2:end])
end

function build_trees(trees::Vector, ::Tuple{}, ::Tuple{})
  return trees
end

function build_trees(
  sectors_to_fuse::NTuple{N,S}, arrows_to_fuse::NTuple{N,Bool}
) where {N,S<:AbstractSector}
  trees = [FusionTree(first(sectors_to_fuse), first(arrows_to_fuse))]
  return build_trees(trees, sectors_to_fuse[2:end], arrows_to_fuse[2:end])
end
