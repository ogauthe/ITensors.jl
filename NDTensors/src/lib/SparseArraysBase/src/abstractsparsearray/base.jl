using ..SparseArraysBase: SparseArraysBase

# Base
function Base.:(==)(a1::AnyAbstractSparseArray, a2::AnyAbstractSparseArray)
  return SparseArraysBase.sparse_isequal(a1, a2)
end

function Base.reshape(a::AnyAbstractSparseArray, dims::Tuple{Vararg{Int}})
  return SparseArraysBase.sparse_reshape(a, dims)
end

function Base.zero(a::AnyAbstractSparseArray)
  return SparseArraysBase.sparse_zero(a)
end

function Base.one(a::AnyAbstractSparseArray)
  return SparseArraysBase.sparse_one(a)
end