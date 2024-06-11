# This files implemets intersect_sectors to find allowed sectors from codomain and domain

# TBD move to Sectors? define as gradedrange(::Vector{<:Sectors.AbstractCategory})?
function sectors_to_reducible(sectors_vec::Vector{<:Sectors.AbstractCategory})
  return GradedAxes.gradedrange(sectors_vec .=> 1)
end

#####################################  fused_sectors  ######################################
function fused_sectors(sectors_vec::NTuple{<:Any,<:Vector{<:Sectors.AbstractCategory}})
  reducible = sectors_to_reducible.(sectors_vec)
  return GradedAxes.blocklabels(GradedAxes.fusion_product(reducible...))
end

function fused_sectors(sector_tuple::NTuple{<:Any,<:Sectors.AbstractCategory})
  return GradedAxes.blocklabels(GradedAxes.fusion_product(sector_tuple...))
end

fused_sectors(sector_tuple::Tuple{<:Sectors.AbstractCategory}) = only(sector_tuple)

fused_sectors(::Tuple{}) = Sectors.sector()

###################################  intersect_sectors  ####################################
function intersect_sectors(
  sectors_codomain::NTuple{<:Any,Vector{C}}, sectors_domain::NTuple{<:Any,Vector{C}}
) where {C<:Sectors.AbstractCategory}
  codomain_fused_sectors = fused_sectors(sectors_codomain)
  domain_fused_sectors = fused_sectors(sectors_domain)
  return intersect_sectors(codomain_fused_sectors, domain_fused_sectors)
end

function intersect_sectors(
  ::Tuple{}, sectors_domain::NTuple{<:Any,<:Vector{<:Sectors.AbstractCategory}}
)
  domain_fused_sectors = fused_sectors(sectors_domain)
  return intersect_sectors(
    Sectors.trivial(eltype(domain_fused_sectors)), domain_fused_sectors
  )
end

function intersect_sectors(
  sectors_codomain::NTuple{<:Any,<:Vector{<:Sectors.AbstractCategory}}, ::Tuple{}
)
  codomain_fused_sectors = fused_sectors(sectors_codomain)
  return intersect_sectors(
    codomain_fused_sectors, Sectors.trivial(eltype(codomain_fused_sectors))
  )
end

function intersect_sectors(
  codomain_sectors::NTuple{<:Any,C}, domain_sectors::NTuple{<:Any,C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(fused_sectors(codomain_sectors), domain_sectors)
end

function intersect_sectors(
  allowed::Vector{C}, sector_tuple::NTuple{<:Any,C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(sector_tuple, allowed)
end

function intersect_sectors(
  c::Sectors.AbstractCategory, sector_tuple::NTuple{<:Any,<:Sectors.AbstractCategory}
)
  return intersect_sectors(c, fused_sectors(sector_tuple))
end

function intersect_sectors(
  sector_tuple::NTuple{<:Any,C}, allowed::Vector{C}
) where {C<:Sectors.AbstractCategory}
  return intersect_sectors(fused_sectors(sector_tuple), allowed)
end

function intersect_sectors(other, c::Sectors.AbstractCategory)
  return intersect_sectors(c, other)
end

function intersect_sectors(
  ::Sectors.CategoryProduct{Tuple{}}, allowed::Vector{<:Sectors.AbstractCategory}
)
  return intersect_sectors(Sectors.trivial(eltype(allowed)), allowed)
end

function intersect_sectors(::Sectors.CategoryProduct{Tuple{}}, c::Sectors.AbstractCategory)
  return intersect_sectors(Sectors.trivial(c), c)
end

function intersect_sectors(c::Sectors.AbstractCategory, ::Sectors.CategoryProduct{Tuple{}})
  return intersect_sectors(Sectors.trivial(c), c)
end

function intersect_sectors(
  ::Sectors.CategoryProduct{Tuple{}}, ::Sectors.CategoryProduct{Tuple{}}
)
  return [Sectors.sector()]
end

function intersect_sectors(sec1::C, sec2::C) where {C<:Sectors.AbstractCategory}
  return sec1 == sec2 ? [sec1] : Vector{C}()
end

function intersect_sectors(sec::C, allowed::Vector{C}) where {C<:Sectors.AbstractCategory}
  return sec ∈ allowed ? [sec] : Vector{C}()
end

function intersect_sectors(
  codomain_allowed::Vector{C}, domain_allowed::Vector{C}
) where {C<:Sectors.AbstractCategory}
  return intersect(codomain_allowed, domain_allowed)
end
