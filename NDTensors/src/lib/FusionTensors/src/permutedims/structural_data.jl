# This file defines StructuralData to be used in permutedims
# StructuralData only depends on Fusion Category, symmetry sectors and permutation
# it does not depend on tensor coefficients or degeneracies

# TBD
# * dual in codomain
# * BlockArray?

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

########################  Constructor from Clebsch-Gordan trees ############################
function contract_projectors(
  trees_codomain_config::Vector{<:Array{Float64}},
  trees_domain_config::Vector{<:Array{Float64}},
  labels_dest::Tuple{Vararg{Int}},
)
  NCoAxes = ndims(eltype(trees_codomain_config)) - 2
  NDoAxes = ndims(eltype(trees_domain_config)) - 2
  N = NDoAxes + NCoAxes
  @assert length(labels_dest) == N + 2

  dims_prod = prod((
    size(first(trees_codomain_config))[begin:NCoAxes]...,
    size(first(trees_domain_config))[begin:NDoAxes]...,
  ))
  projectors = [zeros((dims_prod, 0)) for _ in trees_codomain_config]
  labels_codomain = (ntuple(identity, NCoAxes)..., N + 3, N + 1)
  labels_domain = (ntuple(i -> i + NCoAxes, NDoAxes)..., N + 3, N + 2)
  for (i, (cotree, dotree)) in enumerate(zip(trees_codomain_config, trees_domain_config))
    if length(cotree) > 0 && length(dotree) > 0  # some trees are empty

      #          ----------------dim_sec---------
      #          |                              |
      #          |   ndof_codomain_sec          |  ndof_domain_sec
      #           \  /                           \  /
      #            \/                             \/
      #            /                              /
      #           /                              /
      #          /\                             /\
      #         /  \                           /  \
      #        /\   \                         /\   \
      #       /  \   \                       /  \   \
      #     dim1 dim1 dim3                 dim4 dim5 dim6
      p = TensorAlgebra.contract(
        labels_dest, cotree, labels_codomain, dotree, labels_domain
      )
      projectors[i] = reshape(p, (dims_prod, :))
    end
  end
  return projectors
end

function overlap_cg_trees(
  trees_codomain_in_config::Vector{<:Array{Float64}},
  trees_domain_in_config::Vector{<:Array{Float64}},
  trees_codomain_out_config::Vector{<:Array{Float64}},
  trees_domain_out_config::Vector{<:Array{Float64}},
  perm::TensorAlgebra.BlockedPermutation{2},
)
  # compile time
  NCoAxesIn = ndims(eltype(trees_codomain_in_config)) - 2
  NDoAxesIn = ndims(eltype(trees_domain_in_config)) - 2
  NCoAxesOut = ndims(eltype(trees_codomain_out_config)) - 2
  NDoAxesOut = ndims(eltype(trees_domain_out_config)) - 2
  N = length(perm)
  @assert NCoAxesIn + NDoAxesIn == N
  @assert NCoAxesOut + NDoAxesOut == N
  @assert N > 0

  # initialize output as a BlockArray with
  # blocksize: (n_sectors_out, n_sectors_in)
  # blocksizes: (ndof_config_sectors_out, ndof_config_sectors_in)
  block_rows =
    size.(trees_codomain_out_config, NCoAxesOut + 2) .*
    size.(trees_domain_out_config, NDoAxesOut + 2)
  block_cols =
    size.(trees_codomain_in_config, NCoAxesIn + 2) .*
    size.(trees_domain_in_config, NDoAxesIn + 2)
  isometry = BlockArrays.BlockArray{Float64}(undef, block_rows, block_cols)

  projectors_out = FusionTensors.contract_projectors(
    trees_codomain_out_config, trees_domain_out_config, ntuple(identity, N + 2)
  )
  projectors_in = FusionTensors.contract_projectors(
    trees_codomain_in_config, trees_domain_in_config, (Tuple(perm)..., N + 1, N + 2)
  )

  for (j, proj_in) in enumerate(projectors_in)
    for (i, proj_out) in enumerate(projectors_out)
      isometry[BlockArrays.Block(i, j)] =
        (proj_out'proj_in) / size(trees_codomain_out_config[i], NCoAxesOut + 1)
    end
  end

  return isometry
end

function compute_isometries_CG(
  perm::TensorAlgebra.BlockedPermutation{2,N},
  sectors_codomain_in::NTuple{NCoAxesIn,Vector{C}},
  sectors_domain_in::NTuple{NDoAxesIn,Vector{C}},
  arrow_directions_in::NTuple{N,Bool},
) where {N,NCoAxesIn,NDoAxesIn,C<:Sectors.AbstractCategory}

  # compile time
  @assert N > 0
  @assert NCoAxesIn + NDoAxesIn == N
  NCoAxesOut, NDoAxesOut = BlockArrays.blocklengths(perm)

  # define axes, isdual and allowed sectors
  allowed_sectors_in = intersect_sectors(
    sectors_codomain_in, sectors_domain_in, arrow_directions_in
  )
  sectors_in = (sectors_codomain_in..., sectors_domain_in...)
  sectors_codomain_out = getindex.(Ref(sectors_in), perm[BlockArrays.Block(1)])
  sectors_domain_out = getindex.(Ref(sectors_in), perm[BlockArrays.Block(2)])
  arrow_directions_out = getindex.(Ref(arrow_directions_in), Tuple(perm))
  isdual_codomain_in = .!arrow_directions_in[begin:NCoAxesIn]  # TBD
  isdual_domain_in = arrow_directions_in[(NCoAxesIn + 1):end]
  isdual_codomain_out = .!getindex.(Ref(arrow_directions_in), perm[BlockArrays.Block(1)])  # TBD
  isdual_domain_out = getindex.(Ref(arrow_directions_in), perm[BlockArrays.Block(2)])
  allowed_sectors_out = intersect_sectors(
    sectors_codomain_out, sectors_domain_out, arrow_directions_out
  )

  # initialize output
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

  # cache computed Clebsch-Gordan trees
  trees_codomain_in = Dict{NTuple{NCoAxesIn,Int},Vector{Array{Float64,NCoAxesIn + 2}}}()
  trees_domain_in = Dict{NTuple{NDoAxesIn,Int},Vector{Array{Float64,NDoAxesIn + 2}}}()
  trees_codomain_out = Dict{NTuple{NCoAxesOut,Int},Vector{Array{Float64,NCoAxesOut + 2}}}()
  trees_domain_out = Dict{NTuple{NDoAxesOut,Int},Vector{Array{Float64,NDoAxesOut + 2}}}()

  # loop over all sector configuration
  for it in Iterators.product(eachindex.(sectors_in)...)
    if !isempty(
      intersect_sectors(
        getindex.(sectors_codomain_in, it[begin:NCoAxesIn]),
        getindex.(sectors_domain_in, it[(NCoAxesIn + 1):end]),
      ),
    )
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
