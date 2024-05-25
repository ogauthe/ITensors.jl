# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

struct StructuralData{P,NCoAxesIn,NDoAxesIn,C}
  permutation::P

  # inner constructor to impose constraints on types
  function StructuralData(
    perm::TensorAlgebra.BlockedPermutation{2,N},
    sectors_codomain_in::NTuple{NCoAxesIn,Vector{C}},
    sectors_domain_in::NTuple{NDoAxesIn,Vector{C}},
    arrow_directions_in::NTuple{N,Bool},
    #isometries::Matrix{Mat},  # or Dict(int)
  ) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
    if NCoAxesIn + NDoAxesIn != N
      return error("permutation incompatible with axes")
    end
    return new{typeof(perm),NCoAxesIn,NCoAxesIn,C}(perm)
  end
end

# getters
permutation(sd::StructuralData) = sd.permutation

function Base.ndims(::StructuralData{<:Any,NCoAxesIn,NDoAxesIn}) where {NCoAxesIn,NDoAxesIn}
  return NCoAxesIn + NDoAxesIn
end
ndims_codomain_in(::StructuralData{<:Any,NCoAxesIn}) where {NCoAxesIn} = NCoAxesIn
ndims_domain_in(::StructuralData{<:Any,<:Any,NDoAxesIn}) where {NDoAxesIn} = NDoAxesIn

function ndims_codomain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[1]
end

function ndims_domain_out(sd::StructuralData)
  return BlockArrays.blocklengths(permutation(sd))[2]
end

###################################  utility tools  ########################################
function sectors_to_reducible(sectors_vec::Vector{<:Sectors.AbstractCategory}, isdual::Bool)
  g = GradedAxes.gradedrange(collect(sec => 1 for sec in sectors_vec))
  return isdual ? GradedAxes.label_dual(g) : g
end

function fused_sectors(
  sectors_vec::NTuple{N,Vector{<:Sectors.AbstractCategory}},
  arrow_directions::NTuple{N,Bool},
) where {N}
  reducible = sectors_to_reducible.(sectors_vec, arrow_directions)
  return GradedAxes.blocklabels(reduce(GradedAxes.fusion_product, reducible))
end

function intersect_sectors(
  sectors_codomain::NTuple{NCoAxes,Vector{C}},
  sectors_domain::NTuple{NDoAxes,Vector{C}},
  arrow_directions::NTuple{N,Bool},
) where {NCoAxes,NDoAxes,N,C<:Sectors.AbstractCategory}
  @assert NCoAxes + NDoAxes == N
  codomain_fused_sectors = fused_sectors(sectors_codomain, arrow_directions[begin:NCoAxes])
  domain_fused_sectors = fused_sectors(sectors_domain, arrow_directions[(NCoAxes + 1):end])
  return intersect_sectors(codomain_fused_sectors, domain_fused_sectors)
end

function intersect_sectors(
  ::Tuple{},
  sectors_domain::NTuple{N,Vector{<:Sectors.AbstractCategory}},
  arrow_directions::NTuple{N,Bool},
) where {N}
  domain_fused_sectors = fused_sectors(sectors_domain, arrow_directions)
  return intersect_sectors(
    Sectors.trivial(eltype(domain_fused_sectors)), domain_fused_sectors
  )
end

function intersect_sectors(
  sectors_codomain::NTuple{N,Vector{<:Sectors.AbstractCategory}},
  ::Tuple{},
  arrow_directions::NTuple{N,Bool},
) where {N}
  codomain_fused_sectors = fused_sectors(sectors_codomain, arrow_directions)
  return intersect_sectors(
    codomain_fused_sectors, Sectors.trivial(eltype(codomain_fused_sectors))
  )
end

function get_tree!(dic, it, sectors_all, isdual, allowed_sectors)
  get!(dic, it) do
    prune_fusion_trees(getindex.(sectors_all, it), isdual, allowed_sectors)
  end
end

########################  Constructor from Clebsch-Gordan trees ############################
function compute_isometries_CG(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  sectors_codomain_in::NTuple{NCoAxesIn,Vector{C}},
  sectors_domain_in::NTuple{NDoAxesIn,Vector{C}},
  arrow_directions_in::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
  allowed_sectors_in = intersect_sectors(
    sectors_codomain_in, sectors_domain_in, arrow_directions_in
  )
  sectors_in = (sectors_codomain_in..., sectors_domain_in...)
  sectors_codomain_out = getindex.(Ref(sectors_in), perm[BlockArrays.Block(1)])
  sectors_domain_out = getindex.(Ref(sectors_in), perm[BlockArrays.Block(2)])
  arrow_directions_out = getindex.(Ref(arrow_directions_in), Tuple(perm))
  isdual_codomain_in = arrow_directions_in[begin:NCoAxesIn]
  isdual_domain_in = arrow_directions_in[(NCoAxesIn + 1):end]
  isdual_codomain_out = getindex.(Ref(arrow_directions_in), perm[BlockArrays.Block(1)])
  isdual_domain_out = getindex.(Ref(arrow_directions_in), perm[BlockArrays.Block(2)])
  allowed_sectors_out = intersect_sectors(
    sectors_codomain_out, sectors_domain_out, arrow_directions_out
  )

  isometries = Vector{
    BlockArrays.BlockMatrix{  # TBD use BlockSparseArray 4-dim?
      Float64,
      Matrix{Matrix{Float64}},
      Tuple{
        BlockArrays.BlockedUnitRange{Vector{Int64}},
        BlockArrays.BlockedUnitRange{Vector{Int64}},
      },
    },
  }()
  # avoid precomputing all trees, but cache computed ones
  trees_codomain_in = Dict{NTuple{NCoAxesIn,Int},Vector{Array{Float64,NCoAxesIn + 2}}}()
  trees_domain_in = Dict{NTuple{NDoAxesIn,Int},Vector{Array{Float64,NDoAxesIn + 2}}}()
  trees_codomain_out = Dict{NTuple{NCoAxesOut,Int},Vector{Array{Float64,NCoAxesOut + 2}}}()
  trees_domain_out = Dict{NTuple{NDoAxesOut,Int},Vector{Array{Float64,NDoAxesOut + 2}}}()
  for it in Iterators.product(eachindex.(sectors_in)...)
    sectors_codomain_in_config = getindex.(sectors_codomain_in, it[begin:NCoAxesIn])
    sectors_domain_in_config = getindex.(sectors_domain_in, it[(NCoAxesIn + 1):end])
    if !isempty(intersect_sectors(sectors_codomain_in_config, sectors_domain_in_config))
      trees_codomain_in_config = get_tree!(
        trees_codomain_in,
        it[begin:NCoAxesIn],
        sectors_codomain_in,
        isdual_codomain_in,
        allowed_sectors_in,
      )
      trees_domain_in_config = get_tree!(
        trees_domain_in,
        it[(NCoAxesIn + 1):end],
        sectors_domain_in,
        isdual_domain_in,
        allowed_sectors_in,
      )
      trees_codomain_out_config = get_tree!(
        trees_codomain_out,
        getindex.(Ref(it), perm[BlockArrays.Block(1)]),
        sectors_codomain_out,
        isdual_codomain_out,
        allowed_sectors_out,
      )
      trees_domain_out_config = get_tree!(
        trees_domain_out,
        getindex.(Ref(it), perm[BlockArrays.Block(2)]),
        sectors_domain_out,
        isdual_domain_out,
        allowed_sectors_out,
      )

      isometry = overlap_cg_trees(
        trees_codomain_in_config,
        trees_domain_in_config,
        trees_codomain_out_config,
        trees_domain_out_config,
        perm,
      )
      push!(isometries, isometry)
    end
  end
  return isometries
end

function overlap_cg_trees(
  trees_config_codomain_in,
  trees_config_domain_in,
  trees_config_codomain_out,
  trees_config_domain_out,
  perm,
)
  # compile time
  NCoAxesIn = ndims(eltype(trees_config_codomain_in)) - 2
  NDoAxesIn = ndims(eltype(trees_config_domain_in)) - 2
  NCoAxesOut = ndims(eltype(trees_config_codomain_out)) - 2
  NDoAxesOut = ndims(eltype(trees_config_domain_out)) - 2
  N = length(perm)
  @assert NCoAxesIn + NDoAxesIn == N
  @assert NCoAxesOut + NDoAxesOut == N
  @assert N > 0

  n_sectors_in = length(trees_config_domain_in)
  n_sectors_out = length(trees_config_domain_out)

  block_rows =
    size.(trees_config_codomain_in, NCoAxesIn + 2) .*
    size.(trees_config_domain_in, NDoAxesIn + 2)
  block_cols =
    size.(trees_config_codomain_out, NCoAxesOut + 2) .*
    size.(trees_config_domain_out, NDoAxesOut + 2)

  # blocksize = (n_sectors_in, n_sectors_out)
  # blocksizes = (ndof_config_sectors_in, ndof_config_sectors_out)
  isometry = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  config_dims_prod =
    prod(size(first(trees_config_codomain_in))[begin:(end - 2)]) *
    prod(size(first(trees_config_domain_in))[begin:(end - 2)])
  projectors_out = [ones((config_dims_prod, 0)) for _ in 1:n_sectors_out]
  labels_out = ntuple(identity, N + 2)
  labels_co_out = (ntuple(identity, NCoAxesOut)..., N + 3, N + 1)
  labels_do_out = (ntuple(i -> i + NCoAxesOut, NDoAxesOut)..., N + 3, N + 2)
  for j in 1:n_sectors_out
    cotree = trees_config_codomain_out[j]
    dotree = trees_config_domain_out[j]
    if length(cotree) > 0 && length(dotree) > 0
      proj_out_tensor::Array{Float64,N + 2} = TensorAlgebra.contract(
        labels_out, cotree, labels_co_out, dotree, labels_do_out
      )
      projectors_out[j] = reshape(proj_out_tensor, (config_dims_prod, :))
    end
  end

  labels_in = (N + 1, N + 2, Tuple(perm)...)
  labels_co_in = (ntuple(identity, NCoAxesIn)..., N + 3, N + 1)
  labels_do_in = (ntuple(i -> i + NCoAxesIn, NDoAxesIn)..., N + 3, N + 2)
  for i in 1:n_sectors_in
    cotree = trees_config_codomain_in[i]
    dotree = trees_config_domain_in[i]
    if length(cotree) > 0 && length(dotree) > 0
      proj_in_tensor::Array{Float64,N + 2} = TensorAlgebra.contract(
        labels_in, cotree, labels_co_in, dotree, labels_do_in
      )
      proj_in_mat = reshape(proj_in_tensor, (:, config_dims_prod))
      for j in 1:n_sectors_out
        unitary = proj_in_mat * projectors_out[j]
        isometry[BlockArrays.Block(i, j)] =
          unitary / size(trees_config_codomain_out[j], NCoAxesOut + 1)
      end
    end
  end

  return isometry
end

###################################  Constructor from 6j  ##################################
function compute_isometries_6j(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  codomain_sectors_in::NTuple{NCoAxesIn,Vector{C}},
  domain_sectors_in::NTuple{NDoAxesIn,Vector{C}},
  arrow_directions::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
end
