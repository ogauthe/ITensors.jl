# This files implemets intersect_sectors to find allowed sectors from domain and codomain

# TBD move to SymmetrySectors? define as gradedrange(::Vector{<:SymmetrySectors.AbstractSector})?
function sectors_to_reducible(sectors_vec::Vector{<:SymmetrySectors.AbstractSector})
  return GradedAxes.gradedrange(sectors_vec .=> 1)
end

#####################################  fused_sectors  ######################################
function fused_sectors(
  sectors_vec::NTuple{<:Any,<:Vector{<:SymmetrySectors.AbstractSector}}
)
  reducible = sectors_to_reducible.(sectors_vec)
  return GradedAxes.blocklabels(GradedAxes.fusion_product(reducible...))
end

function fused_sectors(sector_tuple::NTuple{<:Any,<:SymmetrySectors.AbstractSector})
  return GradedAxes.blocklabels(GradedAxes.fusion_product(sector_tuple...))
end

fused_sectors(sector_tuple::Tuple{<:SymmetrySectors.AbstractSector}) = only(sector_tuple)

fused_sectors(::Tuple{}) = SymmetrySectors.TrivialSector()

###################################  intersect_sectors  ####################################
function intersect_sectors(
  sectors_domain::NTuple{<:Any,Vector{S}}, sectors_codomain::NTuple{<:Any,Vector{S}}
) where {S<:SymmetrySectors.AbstractSector}
  domain_fused_sectors = fused_sectors(sectors_domain)
  codomain_fused_sectors = fused_sectors(sectors_codomain)
  return intersect_sectors(domain_fused_sectors, codomain_fused_sectors)
end

function intersect_sectors(
  ::Tuple{}, sectors_codomain::NTuple{<:Any,<:Vector{<:SymmetrySectors.AbstractSector}}
)
  codomain_fused_sectors = fused_sectors(sectors_codomain)
  return intersect_sectors(
    SymmetrySectors.trivial(eltype(codomain_fused_sectors)), codomain_fused_sectors
  )
end

function intersect_sectors(
  sectors_domain::NTuple{<:Any,<:Vector{<:SymmetrySectors.AbstractSector}}, ::Tuple{}
)
  domain_fused_sectors = fused_sectors(sectors_domain)
  return intersect_sectors(
    domain_fused_sectors, SymmetrySectors.trivial(eltype(domain_fused_sectors))
  )
end

function intersect_sectors(
  domain_sectors::NTuple{<:Any,S}, codomain_sectors::NTuple{<:Any,S}
) where {S<:SymmetrySectors.AbstractSector}
  return intersect_sectors(fused_sectors(domain_sectors), codomain_sectors)
end

function intersect_sectors(
  allowed::Vector{S}, sector_tuple::NTuple{<:Any,S}
) where {S<:SymmetrySectors.AbstractSector}
  return intersect_sectors(sector_tuple, allowed)
end

function intersect_sectors(
  s::SymmetrySectors.AbstractSector,
  sector_tuple::NTuple{<:Any,<:SymmetrySectors.AbstractSector},
)
  return intersect_sectors(s, fused_sectors(sector_tuple))
end

function intersect_sectors(
  sector_tuple::NTuple{<:Any,S}, allowed::Vector{S}
) where {S<:SymmetrySectors.AbstractSector}
  return intersect_sectors(fused_sectors(sector_tuple), allowed)
end

function intersect_sectors(other, s::SymmetrySectors.AbstractSector)
  return intersect_sectors(s, other)
end

function intersect_sectors(
  ::SymmetrySectors.TrivialSector, allowed::Vector{SymmetrySectors.TrivialSector()}
)
  return [SymmetrySectors.TrivialSector()]
end

function intersect_sectors(
  ::SymmetrySectors.TrivialSector, allowed::Vector{<:SymmetrySectors.AbstractSector}
)
  return intersect_sectors(
    SymmetrySectors.trivial(eltype(allowed)), SymmetrySectors.TrivialSector()
  )
end

function intersect_sectors(
  ::SymmetrySectors.TrivialSector, s::SymmetrySectors.AbstractSector
)
  return intersect_sectors(SymmetrySectors.trivial(s), s)
end

function intersect_sectors(
  s::SymmetrySectors.AbstractSector, ::SymmetrySectors.TrivialSector
)
  return intersect_sectors(SymmetrySectors.trivial(s), s)
end

function intersect_sectors(::SymmetrySectors.TrivialSector, ::SymmetrySectors.TrivialSector)
  return [SymmetrySectors.TrivialSector()]
end

function intersect_sectors(sec1::S, sec2::S) where {S<:SymmetrySectors.AbstractSector}
  return sec1 == sec2 ? [sec1] : Vector{S}()
end

function intersect_sectors(
  sec::S, allowed::Vector{S}
) where {S<:SymmetrySectors.AbstractSector}
  return sec ∈ allowed ? [sec] : Vector{S}()
end

function intersect_sectors(
  domain_allowed::Vector{S}, codomain_allowed::Vector{S}
) where {S<:SymmetrySectors.AbstractSector}
  return intersect(domain_allowed, codomain_allowed)
end
