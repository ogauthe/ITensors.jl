# This file defines fusion trees for any abelian or non-abelian group

# TBD
# compatibility with TensorKit conventions?

#
# A fusion tree fuses N irreps with quantum dimensions dim1, ..., dimk onto one
# irrep sec with quantum dimension dim_sec. There may be several path that fuse to this
# irrep in the fusion ring and each of them corresponds to a single "thin" fusion tree with
# one degree of freedom.
# We take the ndof_sec trees that fuse on sector sec and merge all of these into one "thick"
# fusion tree containing all degrees of freedom for sector sec.
# The result is a N+2 dimension fusion tree with "uncompressed" size
# (dim1, dim2, ..., dimN, dim_sec, ndof_sec)
#
#      dim_sec  ndof_sec
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
# It is convenient to compress the tree by merging together the dimension legs to yield a
# 3-dim tensor with size (dim1*dim2*...*dimN, dim_sec, ndof_sec)
#
#             ---------------------------
#             |             |           |
#       dim1*dim2*dim3   dim_sec    ndof_sec
#
#
# convention: the trees are not normalized, i.e they do not define a projector on a given
# sector but carry a scaling factor sqrt(dim_sec)
#

# ===================================  Utility tools  ======================================
# TODO move tuple operations elsewhere
flatten_nested_tuple(t::Tuple) = merge_tuples(t...)
function merge_tuples(t1, t2, t3...)
  return merge_tuples(merge_tuples(t1, t2), t3...)
end
merge_tuples(t1::Tuple, t2::Tuple) = (t1..., t2...)
merge_tuples(t1::Tuple) = t1
merge_tuples() = ()

function braid_tuples(t1::Tuple{Vararg{<:Any,N}}, t2::Tuple{Vararg{<:Any,N}}) where {N}
  t12 = (t1, t2)
  nested = ntuple(i -> getindex.(t12, i), N)
  return flatten_nested_tuple(nested)
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
function compress_tree(a::AbstractArray)
  shape_3leg = (prod(size(a)[begin:(end - 2)]), size(a, ndims(a) - 1), size(a, ndims(a)))
  return reshape(a, shape_3leg)
end

function decompress_tree(
  tree::AbstractArray{<:Any,3}, irreps::NTuple{<:Any,<:Sectors.AbstractCategory}
)
  irreps_shape = Sectors.quantum_dimension.(irreps)
  return decompress_tree(tree, irreps_shape)
end

function decompress_tree(tree::AbstractArray{<:Any,3}, irreps_shape::NTuple{<:Any,Int})
  new_shape = (irreps_shape..., size(tree, 2), size(tree, 3))
  return reshape(tree, new_shape)
end

function get_tree!(
  dic::Dict{NTuple{N,Int},Vector{Array{Float64,3}}},
  it::NTuple{N,Int},
  nondual_irreps_vectors::NTuple{N,Vector{C}},
  irreps_isdual::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(dic, it) do
    prune_fusion_trees_compressed(
      getindex.(nondual_irreps_vectors, it), irreps_isdual, allowed_sectors
    )
  end
end

function get_tree!(
  dic::Dict{NTuple{N,Int},Vector{<:Array{Float64}}},
  it::NTuple{N,Int},
  nondual_irreps_vectors::NTuple{N,Vector{C}},
  irreps_isdual::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(dic, it) do
    prune_fusion_trees(
      getindex.(nondual_irreps_vectors, it), irreps_isdual, allowed_sectors
    )
  end
end

function prune_fusion_trees_compressed(
  nondual_irreps::NTuple{N,C}, irreps_isdual::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  return compress_tree.(prune_fusion_trees(nondual_irreps, irreps_isdual, target_sectors))
end

function prune_fusion_trees(
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

function prune_fusion_trees(
  nondual_irreps::NTuple{N,C}, irreps_isdual::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  @assert issorted(target_sectors, lt=!isless, rev=true)  # strict
  irreps_dims = Sectors.quantum_dimension.(nondual_irreps)
  trees, tree_irreps = fusion_trees(nondual_irreps, irreps_isdual)
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
  nondual_irreps::NTuple{N,<:Sectors.CategoryProduct}, irreps_isdual::NTuple{N,Bool}
) where {N}
  # for CategoryProduct, either compute tree(kron( CG tensor for each category))
  # or kron( tree(CG tensor 1 category) for each category).
  # second option allows for easy handling of Abelian groups and should be more efficient
  category_irreps = Sectors.categories.(nondual_irreps)
  n_cat = length(first(category_irreps))

  # construct fusion tree for each category
  transposed_cats = ntuple(c -> getindex.(category_irreps, c), n_cat)
  category_trees_irreps = fusion_trees.(transposed_cats, Ref(irreps_isdual))

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
  nondual_irreps::NTuple{N,<:Sectors.AbstractCategory}, irreps_isdual::NTuple{N,Bool}
) where {N}
  irreps = ntuple(
    i -> irreps_isdual[i] ? GradedAxes.dual(nondual_irreps[i]) : nondual_irreps[i], N
  )
  return fusion_trees(Sectors.SymmetryStyle(first(irreps)), irreps, irreps_isdual)
end

# =====================================  Internals  ========================================

# fusion tree for an Abelian group is trivial
# it does not depend on irreps_isdual once irreps themselves are dualed according to it
function fusion_trees(::Sectors.AbelianGroup, irreps::Tuple, ::Tuple)
  irrep_prod = reduce(⊗, irreps)
  return [ones(ntuple(_ -> 1, length(irreps) + 2))], [irrep_prod]
end

function build_trees(
  old_tree::Matrix,
  old_irrep::Sectors.AbstractCategory,
  level_irrep::Sectors.AbstractCategory,
  level_isdual::Bool,
  ndof_sec::Int,
  sec::Sectors.AbstractCategory,
)
  sector_trees = Vector{typeof(old_tree)}()
  for inner_multiplicity in 1:ndof_sec
    cgt_inner_mult = clebsch_gordan_tensor(
      old_irrep, level_irrep, sec, false, level_isdual, inner_multiplicity
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
  level_isdual::Bool,
)
  new_trees = Vector{typeof(old_tree)}()
  new_irreps = Vector{typeof(old_irrep)}()
  rep = GradedAxes.fusion_product(old_irrep, level_irrep)
  for (ndof_sec, sec) in
      zip(GradedAxes.unlabel.(BlockArrays.blocklengths(rep)), GradedAxes.blocklabels(rep))
    sector_trees = build_trees(
      old_tree, old_irrep, level_irrep, level_isdual, ndof_sec, sec
    )
    append!(new_trees, sector_trees)
    append!(new_irreps, repeat([sec], ndof_sec))
  end
  return new_trees, new_irreps
end

function build_trees(
  trees::Vector, irreps::Vector, level_irrep::Sectors.AbstractCategory, level_isdual::Bool
)
  next_level_trees = typeof(trees)()
  next_level_irreps = typeof(irreps)()
  for (old_tree, old_irrep) in zip(trees, irreps)
    new_trees, new_irreps = build_trees(old_tree, old_irrep, level_irrep, level_isdual)
    append!(next_level_trees, new_trees)
    append!(next_level_irreps, new_irreps)
  end
  return next_level_trees, next_level_irreps
end

function build_trees(
  trees::Vector, tree_irreps::Vector, irreps::Tuple, irreps_isdual::Tuple
)
  next_level_trees, next_level_irreps = build_trees(
    trees, tree_irreps, first(irreps), first(irreps_isdual)
  )
  return build_trees(
    next_level_trees, next_level_irreps, irreps[2:end], irreps_isdual[2:end]
  )
end

function build_trees(trees::Vector, tree_irreps::Vector, ::Tuple{}, ::Tuple{})
  return trees, tree_irreps
end

function compute_thin_trees(irreps::Tuple, irreps_isdual::Tuple)
  # init from trivial, NOT from first(irreps) to get isdual correct
  compressed_thin_trees = [ones((1, 1))]
  unmerged_tree_irreps = [Sectors.trivial(first(irreps))]
  return build_trees(compressed_thin_trees, unmerged_tree_irreps, irreps, irreps_isdual)
end

function merge_trees_irrep(
  thin_trees::Vector, unmerged_tree_irreps::Vector, irrep::Sectors.AbstractCategory
)
  indices_irrep = findall(==(irrep), unmerged_tree_irreps)
  thin_trees_irrep = getindex.(Ref(thin_trees), indices_irrep)
  thick_shape = (size(first(thin_trees_irrep))..., length(indices_irrep))
  return reshape(reduce(hcat, thin_trees_irrep), thick_shape)
end

function merge_trees_irrep(thin_trees::Vector, unmerged_tree_irreps::Vector)
  # merge trees fusing on the same irrep (simpler + avoids moving data at each tree level)
  tree_irreps = sort(unique(unmerged_tree_irreps))
  thick_trees = map(
    irrep -> merge_trees_irrep(thin_trees, unmerged_tree_irreps, irrep), tree_irreps
  )
  return thick_trees, tree_irreps
end

# isdual information is still needed to define CG tensor
function fusion_trees(::Sectors.NonAbelianGroup, irreps::Tuple, irreps_isdual::Tuple)
  # compute trees as 1 tree = 1 degree of freedom
  thin_compressed_trees, unmerged_tree_irreps = compute_thin_trees(irreps, irreps_isdual)
  thick_compressed_trees, tree_irreps = merge_trees_irrep(
    thin_compressed_trees, unmerged_tree_irreps
  )

  irrep_dims = Sectors.quantum_dimension.(irreps)
  thick_trees = map(tree -> decompress_tree(tree, irrep_dims), thick_compressed_trees)
  return thick_trees, tree_irreps
end
