# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

#
# A fusion tree fuses N irreps with quantum dimensions dim1, ..., dimk onto one
# irrep sec with quantum dimension dim_sec. There may be several path that fuse to this
# irrep in the fusion ring and each of them corresponds to a single "thin" fusion tree with
# one degree of freedom.
# We take the struct_mult_sec trees that fuse on sector sec and merge all of these into one
# "thick" fusion tree containing all degrees of freedom for sector sec.
# The result is a N+2 dimension fusion tree with "unfused" size
# (dim1, dim2, ..., dimN, dim_sec, struct_mult_sec)
#
#      dim_sec  struct_mult_sec
#           \  /
#            \/
#            /
#           /
#          /\
#         /  \
#        /\   \
#       /  \   \
#     dim1 dim2 dim3
#
#
# It is convenient to "fuse" the tree by merging together the dimension legs to yield a
# 3-dim tensor with size (dim1*dim2*...*dimN, dim_sec, struct_mult_sec)
#
#             ---------------------------
#             |             |           |
#       dim1*dim2*dim3   dim_sec    struct_mult_sec
#
#
# convention: the trees are not normalized, i.e they do not define a projector on a given
# sector but carry a scaling factor sqrt(dim_sec)
#
# convention: irreps are already dualed if needed, arrows do not affect them. They only
# affect the basis on which the tree acts for self-dual irreps.
#

# ===================================  Utility tools  ======================================
function braid_tuples(t1::Tuple{Vararg{<:Any,N}}, t2::Tuple{Vararg{<:Any,N}}) where {N}
  t12 = (t1, t2)
  nested = ntuple(i -> getindex.(t12, i), N)
  return TensorAlgebra.flatten_tuples(nested)
end

function f_to_c_perm(iterable_product)
  tstrides = (reverse(cumprod(length.(iterable_product)[begin:(end - 1)]))..., 1)
  return map(
    (t,) -> sum((t .- 1) .* tstrides) + 1,
    Iterators.flatten((Iterators.product(eachindex.(reverse(iterable_product))...),),),
  )
end

# compute Kronecker product of fusion trees
# more efficient with recursive construction
trees_kron(a, b, c...) = trees_kron(trees_kron(a, b), c...)

function trees_kron(a, b)
  return map(
    ((t1, t2),) -> _tensor_kron(t1, t2), Iterators.flatten((Iterators.product(a, b),),)
  )
end

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a::AbstractArray{<:Any,N}, b::AbstractArray{<:Any,N}) where {N}
  t1 = ntuple(_ -> 1, N)
  sha = braid_tuples(size(a), t1)
  shb = braid_tuples(t1, size(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end

# ================================  High level interface  ==================================
function merge_tree_leaves(a::AbstractArray)
  shape_3leg = (prod(size(a)[begin:(end - 2)]), size(a, ndims(a) - 1), size(a, ndims(a)))
  return reshape(a, shape_3leg)
end

function unmerge_tree_leaves(
  tree::AbstractArray{<:Real,3}, irreps::NTuple{<:Any,<:Sectors.AbstractCategory}
)
  irreps_shape = Sectors.quantum_dimension.(irreps)
  return unmerge_tree_leaves(tree, irreps_shape)
end

function unmerge_tree_leaves(tree::AbstractArray{<:Real,3}, irreps_shape::NTuple{<:Any,Int})
  new_shape = (irreps_shape..., size(tree, 2), size(tree, 3))
  return reshape(tree, new_shape)
end

function get_tree!(
  cache::Dict{NTuple{N,Int},<:Vector{<:Array{<:Real,3}}},
  it::NTuple{N,Int},
  irreps_vectors::NTuple{N,Vector{C}},
  tree_arrows::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(cache, it) do
    compute_pruned_leavesmerged_fusion_trees(
      getindex.(irreps_vectors, it), tree_arrows, allowed_sectors
    )
  end
end

function get_tree!(
  cache::Dict{NTuple{N,Int},<:Vector{<:Array{<:Real}}},
  it::NTuple{N,Int},
  irreps_vectors::NTuple{N,Vector{C}},
  tree_arrows::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(cache, it) do
    compute_pruned_fusion_trees(getindex.(irreps_vectors, it), tree_arrows, allowed_sectors)
  end
end

function compute_pruned_leavesmerged_fusion_trees(
  irreps::NTuple{N,C}, tree_arrows::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  return merge_tree_leaves.(
    compute_pruned_fusion_trees(irreps, tree_arrows, target_sectors)
  )
end

function compute_pruned_fusion_trees(
  ::Tuple{}, ::Tuple{}, target_sectors::Vector{<:Sectors.AbstractCategory}
)
  @assert issorted(target_sectors, lt=!isless, rev=true)  # strict
  trees_sector = [zeros((Sectors.quantum_dimension(sec), 0)) for sec in target_sectors]
  i0 = findfirst(==(Sectors.trivial(eltype(target_sectors))), target_sectors)
  if !isnothing(i0)
    trees_sector[i0] = ones((1, 1))
  end
  return trees_sector
end

function compute_pruned_fusion_trees(
  irreps::NTuple{N,C}, tree_arrows::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  @assert issorted(target_sectors, lt=!isless, rev=true)  # strict
  irreps_dims = Sectors.quantum_dimension.(irreps)
  trees, tree_irreps = fusion_trees(irreps, tree_arrows)
  trees_sector = [
    zeros((irreps_dims..., Sectors.quantum_dimension(sec), 0)) for sec in target_sectors
  ]
  i_sec, j = 1, 1
  while i_sec <= lastindex(target_sectors) && j <= lastindex(tree_irreps)
    if target_sectors[i_sec] < tree_irreps[j]
      i_sec += 1
    elseif tree_irreps[j] < target_sectors[i_sec]
      j += 1
    else
      trees_sector[i_sec] = trees[j]
      i_sec += 1
      j += 1
    end
  end
  return trees_sector
end

# ================================  Low level interface  ===================================
function fusion_trees(::Tuple{}, ::Tuple{})
  return [ones((1, 1))], [Sectors.sector()]
end

function fusion_trees(
  ::NTuple{N,Sectors.CategoryProduct{Tuple{}}}, ::NTuple{N,Bool}
) where {N}
  return [ones(ntuple(_ -> 1, N + 2))], [Sectors.sector()]
end

function fusion_trees(
  irreps::NTuple{N,<:Sectors.CategoryProduct}, tree_arrows::NTuple{N,Bool}
) where {N}
  # for CategoryProduct, either compute tree(kron( CG tensor for each category))
  # or kron( tree(CG tensor 1 category) for each category).
  # second option allows for easy handling of Abelian groups and should be more efficient
  category_irreps = Sectors.categories.(irreps)
  n_cat = length(first(category_irreps))

  # construct fusion tree for each category
  transposed_cats = ntuple(c -> getindex.(category_irreps, c), n_cat)
  category_trees_irreps = fusion_trees.(transposed_cats, Ref(tree_arrows))

  # reconstruct sector for each product tree
  tree_irreps = map(
    cats -> Sectors.sector(eltype(category_irreps), cats),
    Iterators.flatten((Iterators.product((getindex.(category_trees_irreps, 2))...),),),
  )

  # compute Kronecker product of fusion trees
  trees = trees_kron(getindex.(category_trees_irreps, 1)...)

  # sort irreps. Each category is sorted, permutation is obtained by reversing loop order
  perm = f_to_c_perm(getindex.(category_trees_irreps, 2))
  tree_irreps = getindex.(Ref(tree_irreps), perm)
  trees = getindex.(Ref(trees), perm)

  return trees, tree_irreps
end

function fusion_trees(
  irreps::NTuple{N,<:Sectors.AbstractCategory}, tree_arrows::NTuple{N,Bool}
) where {N}
  return fusion_trees(Sectors.SymmetryStyle(first(irreps)), irreps, tree_arrows)
end

# =====================================  Internals  ========================================

# fusion tree for an Abelian group is trivial
# it does not depend on arrow directions
function fusion_trees(::Sectors.AbelianGroup, irreps::Tuple, ::Tuple)
  irrep_prod = reduce(⊗, irreps)
  return [ones(ntuple(_ -> 1, length(irreps) + 2))], [irrep_prod]
end

function build_trees(
  old_tree::Matrix,
  old_irrep::Sectors.AbstractCategory,
  level_irrep::Sectors.AbstractCategory,
  level_arrow::Bool,
  inner_multiplicity::Integer,
  sec::Sectors.AbstractCategory,
)
  sector_trees = Vector{typeof(old_tree)}()
  for inner_mult_index in 1:inner_multiplicity
    cgt_inner_mult = clebsch_gordan_tensor(
      old_irrep, level_irrep, sec, false, level_arrow, inner_mult_index
    )
    dim_old_irrep, dim_level_irrep, dim_sec = size(cgt_inner_mult)
    tree = old_tree * reshape(cgt_inner_mult, (dim_old_irrep, dim_level_irrep * dim_sec))
    new_tree = reshape(tree, (size(old_tree, 1) * dim_level_irrep, dim_sec))
    push!(sector_trees, new_tree)
  end
  return sector_trees
end

function build_trees(
  old_tree::Matrix,
  old_irrep::Sectors.AbstractCategory,
  level_irrep::Sectors.AbstractCategory,
  level_arrow::Bool,
)
  new_trees = Vector{typeof(old_tree)}()
  new_irreps = Vector{typeof(old_irrep)}()
  rep = GradedAxes.fusion_product(old_irrep, level_irrep)
  for (inner_multiplicity, sec) in
      zip(BlockArrays.blocklengths(rep), GradedAxes.blocklabels(rep))
    sector_trees = build_trees(
      old_tree, old_irrep, level_irrep, level_arrow, inner_multiplicity, sec
    )
    append!(new_trees, sector_trees)
    append!(new_irreps, repeat([sec], inner_multiplicity))
  end
  return new_trees, new_irreps
end

function build_trees(
  trees::Vector, irreps::Vector, level_irrep::Sectors.AbstractCategory, level_arrow::Bool
)
  next_level_trees = typeof(trees)()
  next_level_irreps = typeof(irreps)()
  for (old_tree, old_irrep) in zip(trees, irreps)
    new_trees, new_irreps = build_trees(old_tree, old_irrep, level_irrep, level_arrow)
    append!(next_level_trees, new_trees)
    append!(next_level_irreps, new_irreps)
  end
  return next_level_trees, next_level_irreps
end

function build_trees(trees::Vector, tree_irreps::Vector, irreps::Tuple, tree_arrows::Tuple)
  next_level_trees, next_level_irreps = build_trees(
    trees, tree_irreps, first(irreps), first(tree_arrows)
  )
  return build_trees(next_level_trees, next_level_irreps, irreps[2:end], tree_arrows[2:end])
end

function build_trees(trees::Vector, tree_irreps::Vector, ::Tuple{}, ::Tuple{})
  return trees, tree_irreps
end

function compute_thin_trees(irreps::Tuple, tree_arrows::Tuple)
  # init from trivial, NOT from first(irreps) to get first arrow correct
  thin_trees = [ones((1, 1))]
  tree_irreps = [Sectors.trivial(first(irreps))]
  return build_trees(thin_trees, tree_irreps, irreps, tree_arrows)
end

function cat_thin_trees(
  thin_trees::Vector, uncat_tree_irreps::Vector, sector_irrep::Sectors.AbstractCategory
)
  indices_irrep = findall(==(sector_irrep), uncat_tree_irreps)
  thin_trees_irrep = getindex.(Ref(thin_trees), indices_irrep)
  thick_shape = (size(first(thin_trees_irrep))..., length(indices_irrep))
  return reshape(reduce(hcat, thin_trees_irrep), thick_shape)
end

function cat_thin_trees(thin_trees::Vector, uncat_tree_irreps::Vector)
  # cat trees fusing on the same irrep
  tree_irreps = sort(unique(uncat_tree_irreps))
  thick_trees = map(
    irrep -> cat_thin_trees(thin_trees, uncat_tree_irreps, irrep), tree_irreps
  )
  return thick_trees, tree_irreps
end

# arrow direction is needed to define CG tensor for Lie groups
function fusion_trees(::Sectors.NonAbelianGroup, irreps::Tuple, tree_arrows::Tuple)
  # compute "thin" trees: 1 tree = fuses on ONE irrep
  thin_trees, uncat_tree_irreps = compute_thin_trees(irreps, tree_arrows)

  # cat thin trees into "thick" trees
  thick_mergedleaves_trees, tree_irreps = cat_thin_trees(thin_trees, uncat_tree_irreps)

  irrep_dims = Sectors.quantum_dimension.(irreps)
  thick_trees = map(tree -> unmerge_tree_leaves(tree, irrep_dims), thick_mergedleaves_trees)
  return thick_trees, tree_irreps
end
