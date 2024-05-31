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

###################################  utility tools  ########################################
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
# more efficient with iterative construction
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

reconstruct_sector(T::Type, cats::Tuple) = Sectors.sector(T(cats))  # recover NamedTuple

function compress_tree(a::AbstractArray)
  shape_3leg = (prod(size(a)[begin:(end - 2)]), size(a, ndims(a) - 1), size(a, ndims(a)))
  return reshape(a, shape_3leg)
end

function decompress_tree(
  tree::AbstractArray{<:Any,3}, irreps::NTuple{<:Any,<:Sectors.AbstractCategory}
)
  new_shape = (Sectors.quantum_dimension.(irreps)..., size(tree, 2), size(tree, 3))
  return reshape(tree, new_shape)
end

#################################   High level interface  ##################################
function get_tree!(
  dic::Dict{NTuple{N,Int},Vector{Array{Float64,3}}},
  it::NTuple{N,Int},
  sectors_all::NTuple{N,Vector{C}},
  isdual::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(dic, it) do
    prune_fusion_trees_compressed(getindex.(sectors_all, it), isdual, allowed_sectors)
  end
end

function get_tree!(
  dic::Dict{NTuple{N,Int},Vector{<:Array{Float64}}},
  it::NTuple{N,Int},
  sectors_all::NTuple{N,Vector{C}},
  isdual::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  get!(dic, it) do
    prune_fusion_trees(getindex.(sectors_all, it), isdual, allowed_sectors)
  end
end

function prune_fusion_trees_compressed(
  irreps_config::NTuple{N,C}, irreps_isdual::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  return compress_tree.(prune_fusion_trees(irreps_config, irreps_isdual, target_sectors))
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
  irreps_config::NTuple{N,C}, irreps_isdual::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  @assert issorted(target_sectors, lt=!isless, rev=true)  # strict
  irreps_dims = Sectors.quantum_dimension.(irreps_config)
  trees, tree_irreps = fusion_trees(irreps_config, irreps_isdual)
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

##################################  low level interface  ###################################
function fusion_trees(::Tuple{}, ::Tuple{})
  return [ones((1, 1))], [Sectors.sector()]
end

function fusion_trees(
  irreps::NTuple{N,<:Sectors.CategoryProduct}, isdual::NTuple{N,Bool}
) where {N}
  # for CategoryProduct, either compute tree(kron( CG tensor for each category))
  # or kron( tree(CG tensor 1 category) for each category).
  # second option allows for easy handling of Abelian groups and should be more efficient
  category_irreps = Sectors.categories.(irreps)
  n_cat = length(first(category_irreps))

  # construct fusion tree for each category
  transposed_cats = ntuple(c -> getindex.(category_irreps, c), n_cat)
  category_trees_irreps = fusion_trees.(transposed_cats, Ref(isdual))

  # reconstruct sector for each product tree
  tree_irreps = map(
    cats -> reconstruct_sector(eltype(category_irreps), cats),  # recover NamedTuple key
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
  irreps::NTuple{N,<:Sectors.AbstractCategory}, isdual::NTuple{N,Bool}
) where {N}
  irreps_arrow = ntuple(i -> isdual[i] ? GradedAxes.dual(irreps[i]) : irreps[i], N)
  return fusion_trees(Sectors.SymmetryStyle(first(irreps)), irreps_arrow, isdual)
end

#######################################  Internals  ########################################

# fusion tree for an Abelian group is trivial
# it does not depend on dual once irreps themselves are dualed
function fusion_trees(::Sectors.AbelianGroup, irreps_arrow::Tuple, ::Tuple)
  irrep_prod = reduce(⊗, irreps_arrow)
  return [ones(ntuple(_ -> 1, length(irreps_arrow) + 2))], [irrep_prod]
end

function add_cg_layer(
  trees::Vector,
  tree_irreps::Vector,
  layer_irrep::Sectors.AbstractCategory,
  isdual_layer::Bool,
)
  new_irreps = typeof(tree_irreps)()
  new_trees = typeof(trees)()
  for (old_tree, old_irrep) in zip(trees, tree_irreps)
    sh_cg_dof = (size(old_tree, 2), :)
    rep = GradedAxes.fusion_product(old_irrep, layer_irrep)
    for (ndof_sec, sec) in zip(BlockArrays.blocklengths(rep), GradedAxes.blocklabels(rep))
      shnt = (:, Sectors.quantum_dimension(sec))
      for inner_multiplicity in 1:ndof_sec
        cgt_layer_dof = clebsch_gordan_tensor(
          old_irrep, layer_irrep, sec, false, isdual_layer, inner_multiplicity
        )
        nt = old_tree * reshape(cgt_layer_dof, sh_cg_dof)
        nt = reshape(nt, shnt)
        push!(new_trees, nt)
        push!(new_irreps, sec)
      end
    end
  end
  return new_trees, new_irreps
end

function compute_thin_trees(irreps_arrow::Tuple, isdual::Tuple)
  compressed_thin_trees = [ones((1, 1))]
  unmerged_tree_irreps = [Sectors.trivial(first(irreps_arrow))]
  for (irrep, b) in zip(irreps_arrow, isdual)
    compressed_thin_trees, unmerged_tree_irreps = add_cg_layer(
      compressed_thin_trees, unmerged_tree_irreps, irrep, b
    )
  end
  return compressed_thin_trees, unmerged_tree_irreps
end

function merge_trees_irrep(
  thin_trees::Vector, unmerged_tree_irreps::Vector, irrep::Sectors.AbstractCategory
)
  indices_irrep = findall(==(irrep), unmerged_tree_irreps)
  thin_trees_irrep = getindex.(Ref(thin_trees), indices_irrep)
  thick_shape = (size(first(thin_trees_irrep))..., length(indices_irrep))
  return reshape(hcat(thin_trees_irrep...), thick_shape)
end

# isdual information is still needed to define CG tensor
function fusion_trees(::Sectors.NonAbelianGroup, irreps_arrow::Tuple, isdual::Tuple)
  # compute trees as 1 tree = 1 degree of freedom
  thin_compressed_trees, unmerged_tree_irreps = compute_thin_trees(irreps_arrow, isdual)

  # merge trees fusing on the same irrep (simpler + avoids moving data at each layer)
  tree_irreps = sort(unique(unmerged_tree_irreps))
  thick_compressed_trees =
    merge_trees_irrep.(Ref(thin_compressed_trees), Ref(unmerged_tree_irreps), tree_irreps)
  thick_trees = decompress_tree.(thick_compressed_trees, Ref(irreps_arrow))
  return thick_trees, tree_irreps
end
