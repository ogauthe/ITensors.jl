# This file defines fusion trees for any abelian or non-abelian group

# TBD
# better way to contract tensors?
# compatibility with TensorKit conventions

#
# A fusion tree fuses k irreps with quantum dimensions dim1, ..., dimk onto one
# irrep P with quantum dimension dimP. There may be several path that produce
# this irrep in the fusion ring and each of them correspond to a single "simple" fusion
# tree with one degree of freedom.
# We take the ndof_sec trees that fuse on sector sec and merge all of these into one thick
# fusion tree containing all degrees of freedom for irrep sec.
# The result is a k+2 dimension fusion tree with size (dim1, dim2, ..., dimk, dim_sec, ndof_sec)
#
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
#     dim1 dim1 dim3
#
# It is convenient to merge together the dimension legs to yield a 3-dim tensor
#             ---------------------------
#             |             |           |
#       dim1*dim2*dim3   dim_sec    ndof_sec
#

###################################  utility tools  ########################################
# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a, b)
  sha = ntuple(i -> Bool(i % 2) ? size(a, i ÷ 2 + 1) : 1, 2 * ndims(a))
  shb = ntuple(i -> Bool(i % 2) ? 1 : size(b, i ÷ 2), 2 * ndims(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end

recover_key(::Type{<:Tuple}, cats::Tuple) = Sectors.sector(cats)

function recover_key(::Type{<:NamedTuple{Keys}}, cats::Tuple) where {Keys}
  return Sectors.sector(ntuple(c -> Keys[c] => cats[c], length(cats))...)
end

function reshape_3legs(a::AbstractArray)
  shape_3leg = (prod(size(a)[begin:(end - 2)]), size(a, ndims(a) - 1), size(a, ndims(a)))
  return reshape(a, shape_3leg)
end

#################################   High level interface  ##################################
function precompute_allowed_trees(
  irrep_configurations::NTuple{N,Vector{C}},
  irreps_isdual::NTuple{N,Bool},
  allowed_sectors::Vector{C},
) where {N,C<:Sectors.AbstractCategory}
  n_sectors = length(allowed_sectors)
  trees = Matrix{Array{Float64,3}}(undef, (n_sectors, 0))
  allowed_configs = Vector{NTuple{N,Int}}()
  for it in Iterators.product(eachindex.(irrep_configurations)...)
    irreps_config = getindex.(irrep_configurations, it)
    if !isempty(intersect_sectors(irreps_config, allowed_sectors))
      trees_config_sector = prune_fusion_trees_compressed(
        irreps_config, irreps_isdual, allowed_sectors
      )
      trees = hcat(trees, trees_config_sector)
      push!(allowed_configs, it)
    end
  end
  return trees, allowed_configs
end

function prune_fusion_trees_compressed(irreps_config, irreps_isdual, target_sectors)
  return reshape_3legs.(prune_fusion_trees(irreps_config, irreps_isdual, target_sectors))
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
  return [ones((1, 1))], [Sectors.sector(())]
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
  category_tree_input = ntuple(c -> (ntuple(i -> category_irreps[i][c], N)), n_cat)
  category_trees_irreps = ntuple(c -> fusion_trees(category_tree_input[c], isdual), n_cat)

  # compute kronecker product of fusion trees
  # more efficient with iterative construction instead of Iterators.product
  trees = first(first(category_trees_irreps))
  for c in 2:n_cat  # C order loop
    trees = collect(
      _tensor_kron(t, tc) for t in trees for tc in first(category_trees_irreps[c])
    )
  end

  # recover irreps for each tree
  # do not use previous loop as category type would change at each step
  tree_irreps = [  # keep sorted irreps with C-order loop
    recover_key(eltype(category_irreps), reverse(it)) for it in Iterators.flatten((
      Iterators.product((reverse(ntuple(c -> category_trees_irreps[c][2], n_cat)))...),
    ),)
  ]

  return trees, tree_irreps
end

function fusion_trees(
  irreps::NTuple{N,<:Sectors.AbstractCategory}, isdual::NTuple{N,Bool}
) where {N}
  irreps_arrow = ntuple(i -> isdual[i] ? GradedAxes.dual(irreps[i]) : irreps[i], N)
  return fusion_trees(Sectors.SymmetryStyle(first(irreps)), irreps_arrow, isdual)
end

# fusion tree for an Abelian group is trivial
# it does not depend on dual
function fusion_trees(::Sectors.AbelianGroup, irreps_arrow, ::Tuple)
  irrep_prod = reduce(⊗, irreps_arrow)
  return [ones(ntuple(_ -> 1, length(irreps_arrow) + 2))], [irrep_prod]
end

# isdual information is still needed to define CG tensor
function fusion_trees(::Sectors.NonAbelianGroup, irreps_arrow::NTuple{N}, isdual) where {N}
  # for a non-abelian group, a given set of irreps leads to several different sectors
  # with many degrees of freedom
  # 3 possible conventions:        | exemple for fusion tree SU2(1/2)^3
  # - 1 tree = 1 set of irreps     | 1 tree with shape (2,2,2,8)
  # - 1 tree = 1 output sector     | 2 tree with shapes (2,2,2,2,2), (2,2,2,1,4)
  # - 1 tree = 1 degree of freedom | 3 trees with shapes (2,2,2,2), (2,2,2,2), (2,2,2,4)

  tree_matrices = [ones((1, 1))]
  unmerged_tree_irreps = [Sectors.trivial(eltype(irreps_arrow))]

  # compute trees as 1 tree = 1 degree of freedom
  for (i, irrep) in enumerate(irreps_arrow)
    tree_matrices, unmerged_tree_irreps = add_cg_layer(
      tree_matrices, unmerged_tree_irreps, irrep, isdual[i]
    )
  end

  # merge trees fusing on the same irrep at the very end
  # simpler + avoids moving data at each step
  # convention: the trees are not normalized, i.e they do not define isometries but
  # carry a scaling factor tree' * tree = dim(irrep) * I
  shape_input = Sectors.quantum_dimension.(irreps_arrow)
  tree_irreps = sort(unique(unmerged_tree_irreps))
  trees = Vector{Array{Float64,N + 2}}()
  for irrep in tree_irreps
    sector_indices = findall(==(irrep), unmerged_tree_irreps)
    shape_1tree = (shape_input..., Sectors.quantum_dimension(irrep))
    shape_thick_tree = (shape_1tree..., length(sector_indices))
    sector_tree = Array{Float64,N + 2}(undef, shape_thick_tree)
    for (i, tree) in enumerate(tree_matrices[sector_indices])
      sector_tree[.., i] = reshape(tree, shape_1tree)
    end
    push!(trees, sector_tree)
  end
  return trees, tree_irreps
end

function add_cg_layer(trees, tree_irreps, layer_irrep, isdual_layer)
  new_irreps = Vector{typeof(layer_irrep)}()
  new_trees = Vector{Array{Float64,2}}()
  for (old_tree, old_irrep) in zip(trees, tree_irreps)
    sh_cg_dof = (size(old_tree, 2), :)
    rep = GradedAxes.fusion_product(old_irrep, layer_irrep)
    for (ndof_sec, sec) in zip(BlockArrays.blocklengths(rep), GradedAxes.blocklabels(rep))
      dim_sec = Sectors.quantum_dimension(sec)
      shnt = (:, dim_sec)
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
