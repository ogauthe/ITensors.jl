# This file defines fusion trees for any abelian or non-abelian group

# TBD
# better way to contract tensors?
# compatibility with TensorKit conventions

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a, b)
  sha = ntuple(i -> Bool(i % 2) ? size(a, i ÷ 2 + 1) : 1, 2 * ndims(a))
  shb = ntuple(i -> Bool(i % 2) ? 1 : size(b, i ÷ 2), 2 * ndims(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
end

function prune_fusion_trees(
  irreps_config::NTuple{N,C}, irreps_isdual::NTuple{N,Bool}, target_sectors::Vector{C}
) where {N,C<:Sectors.AbstractCategory}
  # A fusion tree fuses k irreps with quantum dimensions dim1, ..., dimk onto one
  # irrep P with quantum dimension dimP. There may be several path that produce
  # this irrep in the fusion ring and each of them correspond to a single "simple" fusion
  # tree with one degree of freedom.
  # We take the ndofP trees that fuse on irrep P and merge all of these into one thick
  # fusion tree containing all degrees of freedom for irrep P.
  # The result is a k+2 dimension fusion tree with size (dim1, dim2, ..., dimk, dimP, nof_P)
  #
  #
  #      dimP  ndofP
  #         \  /
  #          \/
  #          /
  #         /
  #        /\
  #       /  \
  #      /\   \
  #     /  \   \
  #   dim1 dim1 dim3
  #
  # It is convenient to merge together the dimension legs to yield a 3-dim tensor
  #
  #  dim1*dim2*dim3-------------dimP
  #                     |
  #                   ndofP

  @assert issorted(target_sectors)
  target_sectors_dims = Sectors.quantum_dimension.(target_sectors)
  irreps_dims_prod = prod(Sectors.quantum_dimension.(irreps_config))
  n_sectors = length(target_sectors)
  rep = reduce(GradedAxes.fusion_product, irreps_config; init=Sectors.trivial(C))
  trees, tree_irreps = fusion_trees(irreps_config, irreps_isdual)
  trees_sector = [
    zeros((irreps_dims_prod, target_sectors_dims[i_sec], 0)) for i_sec in 1:n_sectors
  ]
  i_sec, j = 1, 1
  while i_sec <= n_sectors && j <= lastindex(tree_irreps)
    if target_sectors[i_sec] < tree_irreps[j]
      i_sec += 1
    elseif tree_irreps[j] < target_sectors[i_sec]
      j += 1
    else
      shape_3legs = (
        irreps_dims_prod,
        target_sectors_dims[i_sec],
        GradedAxes.unlabel.(GradedAxes.blocklengths(rep)[j]),
      )
      trees_sector[i_sec] = reshape(trees[j], shape_3legs)
      i_sec += 1
      j += 1
    end
  end
  return trees_sector
end

function fusion_trees(::Tuple{}, ::Tuple{})
  return [ones((1,))], [Sectors.sector(())]
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

recover_key(::Type{<:Tuple}, cats::Tuple) = Sectors.sector(cats)
function recover_key(::Type{<:NamedTuple{Keys}}, cats::Tuple) where {Keys}
  return Sectors.sector(ntuple(c -> Keys[c] => cats[c], length(cats))...)
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

function add_cg_layer(trees, tree_irreps, layer_irrep, isdual)
  new_irreps = Vector{typeof(layer_irrep)}()
  new_trees = Vector{Array{Float64,2}}()
  for i in eachindex(tree_irreps)
    dim_i = Sectors.quantum_dimension(tree_irreps[i])
    shcg = (dim_i, :)
    rep = GradedAxes.fusion_product(tree_irreps[i], layer_irrep)
    for (degen, s) in zip(BlockArrays.blocklengths(rep), GradedAxes.blocklabels(rep))
      dim_s = Sectors.quantum_dimension(s)
      shnt = (:, dim_s)
      for internal_mult in 1:degen
        cgt = clebsch_gordan_tensor(
          tree_irreps[i], layer_irrep, s, false, isdual, internal_mult
        )
        nt = trees[i] * reshape(cgt, shcg)
        nt = reshape(nt, shnt)
        push!(new_trees, nt)
        push!(new_irreps, s)
      end
    end
  end
  return new_trees, new_irreps
end
