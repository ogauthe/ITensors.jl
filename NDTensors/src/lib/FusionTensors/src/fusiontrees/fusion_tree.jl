# This file defines fusion trees for any abelian or non-abelian group

# TBD
# better way to contract tensors?
# prune trees?
# change tree structure / number of dof?
# compatibility with TensorKit conventions

# LinearAlgebra.kron does not allow input for ndims>2
function _tensor_kron(a, b)
  sha = ntuple(i -> Bool(i % 2) ? size(a, i ÷ 2 + 1) : 1, 2 * ndims(a))
  shb = ntuple(i -> Bool(i % 2) ? 1 : size(b, i ÷ 2), 2 * ndims(b))
  c = reshape(a, sha) .* reshape(b, shb)
  return reshape(c, size(a) .* size(b))
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
  trees = first(last(category_trees_irreps))
  for c in reverse(1:(n_cat - 1))  # F order loop
    trees = collect(
      _tensor_kron(t, tc) for t in trees for tc in first(category_trees_irreps[c])
    )
  end

  # recover irreps for each tree
  # do not use previous loop as category type would change at each step
  tree_irreps = [
    recover_key(eltype(category_irreps), it) for it in Iterators.flatten((
      Iterators.product(ntuple(c -> category_trees_irreps[c][2], n_cat)...),
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
  return [ones(ntuple(_ -> 1, length(irreps_arrow) + 1))], [irrep_prod]
end

# isdual information is still needed to define CG tensor
function fusion_trees(::Sectors.NonAbelianGroup, irreps_arrow, isdual)
  # for a non-abelian group, a given set of irreps leads to several different sectors
  # with many degrees of freedom
  # 3 possible conventions:        | exemple for fusion tree SU2(1/2)^3
  # - 1 tree = 1 set of irreps     | 1 tree with shape (2,2,2,8)
  # - 1 tree = 1 output sector     | 2 tree with shapes (2,2,2,2,2), (2,2,2,1,4)
  # - 1 tree = 1 degree of freedom | 3 trees with shapes (2,2,2,2), (2,2,2,2), (2,2,2,4)

  # here choose 1 tree = 1 degree of freedom

  tree_matrices = [ones((1, 1))]
  unsorted_tree_irreps = [Sectors.trivial(eltype(irreps_arrow))]

  for (i, irrep) in enumerate(irreps_arrow)
    tree_matrices, unsorted_tree_irreps = add_cg_layer(
      tree_matrices, unsorted_tree_irreps, irrep, isdual[i]
    )
  end

  sh0 = Sectors.quantum_dimension.(irreps_arrow)
  so = sortperm(unsorted_tree_irreps; alg=MergeSort)  # impose deterministic sort
  tree_irreps = Vector{eltype(irreps_arrow)}()
  trees = Vector{Array{Float64,length(irreps_arrow) + 1}}()
  for k in so
    push!(tree_irreps, unsorted_tree_irreps[k])
    push!(trees, reshape(tree_matrices[k], (sh0..., size(tree_matrices[k], 2))))
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
