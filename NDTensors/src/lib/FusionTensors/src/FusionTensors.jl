module FusionTensors

export FusionTensors,
  n_codomain_legs,
  n_domain_legs,
  matrix_size,

  # TODO does order matter?
  include(
  "fusiontensor.jl"
)
include("dense.jl")
include("fusiontensorbase.jl")
include("fusiontensorblocks.jl")
include("linalg.jl")
include("permutedims.jl")
end
