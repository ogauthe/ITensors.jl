module FusionTensors

using LinearAlgebra: LinearAlgebra

using BlockArrays: BlockArrays
using EllipsisNotation: var".."
using HalfIntegers: HalfIntegers

import NDTensors.BlockSparseArrays
import NDTensors.GradedAxes
import NDTensors.Sectors: Sectors, ⊗
import NDTensors.TensorAlgebra

include("fusiontrees/clebsch_gordan.jl")
include("fusiontrees/fusion_tree.jl")

include("fusiontensor/fusiontensor.jl")
include("fusiontensor/base_interface.jl")
include("fusiontensor/dense.jl")
include("fusiontensor/linear_algebra_interface.jl")
include("fusiontensor/tensor_algebra_interface.jl")
include("permutedims/structural_data.jl")
include("permutedims/permutedims.jl")
end
