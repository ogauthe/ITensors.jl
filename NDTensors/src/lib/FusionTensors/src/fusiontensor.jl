# This file defines struct FusionTensor and constructors

using ITensors: @debug_check
using NDTensors.BlockSparseArrays: BlockSparseArray

struct FusionTensor{
  T<:Number,N,Axes<:Tuple{N,AbstractUnitRange{Int}},Arr<:BlockSparseArray{T,2}
} <: AbstractArray{T,N}
  # TBD more type stable with only N fixed or with NRL and NCL as type parameters?
  # can also define N_ROW_LEG as type parameter
  # with N fixed and n_row_legs dynamic, permutedims, dagger and co preserve type
  # but tensor contraction output type is not knwon at compile time
  axes::Axes
  n_row_legs::Int
  matrix::Arr
end

# getters
matrix(ft::FusionTensor) = ft.matrix
axes(ft::FusionTensor) = ft.axes
n_row_legs(ft::FusionTensor) = ft.n_row_legs

# misc
domain_axes(ft::FusionTensor) = axes(ft)[begin:n_row_legs(ft)]
codomain_axes(ft::FusionTensor) = axes(ft)[n_row_legs(t):end]
n_column_legs(ft::FusionTensor) = ndims(ft) - n_row_legs(ft)
matrix_size(ft::FusionTensor) = size(matrix(ft))
row_axis(ft::FusionTensor) = axes(matrix(ft))[1]
column_axis(ft::FusionTensor) = axes(matrix(ft))[2]

# constructors
function FusionTensor(codomain_axes, domain_axes, matrix)
  @debug_check length(codomain_axes) > 0
  @debug_check length(domain_axes) > 0
  @debug_check prod(length.(codomain_axes)) == size(matrix, 1)
  @debug_check prod(length.(domain_axes)) == size(matrix, 2)
  axes = (codomain_axes..., domain_axes...)
  n_row_legs = length(axes)
  return FusionTensor(axes, n_row_legs, matrix)
end

# swap row and column axes, transpose matrix blocks, dual any axis. No basis change.
function dagger(ft::FusionTensor)  # TBD change name? TBD move to permutedims?
  return FusionTensor(
    dual.(domain_axes(ft)), dual.(codomain_axes(ft)), transpose(matrix(ft))
  )
end
