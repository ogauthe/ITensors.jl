module FusionTensors

using LinearAlgebra: LinearAlgebra

using BlockArrays: BlockArrays
using HalfIntegers: HalfIntegers
using WignerSymbols: WignerSymbols

using NDTensors.BlockSparseArrays: BlockSparseArrays
using NDTensors.GradedAxes: GradedAxes
using NDTensors.SymmetrySectors: SymmetrySectors, ⊗
using NDTensors.TensorAlgebra: TensorAlgebra

include("fusiontrees/clebsch_gordan.jl")
include("fusiontrees/fusion_tree.jl")

include("fusiontensor/fusedaxes.jl")
include("fusiontensor/fusiontensor.jl")
include("fusiontensor/base_interface.jl")
include("fusiontensor/dense.jl")
include("fusiontensor/linear_algebra_interface.jl")
include("fusiontensor/tensor_algebra_interface.jl")
include("permutedims/unitaries.jl")
include("permutedims/permutedims.jl")
end
