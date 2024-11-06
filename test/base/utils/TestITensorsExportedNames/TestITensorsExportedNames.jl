module TestITensorsExportedNames
#=
# List constructed with (along with cleaning up
# macro symbols):
using DelimitedFiles: writedlm
using ITensors: ITensors
open("itensors_exported_names.jl", "w") do io
  writedlm(io, repr.(names(ITensors)) .* ",")
end
=#
const ITENSORS_EXPORTED_NAMES = [
  Symbol("@disable_warn_order"),
  Symbol("@reset_warn_order"),
  Symbol("@set_warn_order"),
  Symbol("@ts_str"),
  :Apply,
  :Block,
  :ITensor,
  :ITensors,
  :Index,
  :IndexSet,
  :IndexVal,
  :LinearAlgebra,
  :NDTensors,
  :Order,
  :QN,
  :Spectrum,
  :TagSet,
  :addblock!,
  :addtags,
  :addtags!,
  :allhastags,
  :anyhastags,
  :apply,
  :argsdict,
  :array,
  :axpy!,
  :blockdim,
  :blockoffsets,
  :checkflux,
  :combinedind,
  :combiner,
  :commonind,
  :commonindex,
  :commoninds,
  :complex!,
  :contract,
  :convert_eltype,
  :convert_leaf_eltype,
  :dag,
  :delta,
  :dense,
  :denseblocks,
  :diag,
  :diagITensor,
  :diag_itensor,
  :diagitensor,
  :dim,
  :dims,
  :dir,
  :directsum,
  :disable_tblis!,
  :disable_warn_order!,
  :dot,
  :eachindval,
  :eachnzblock,
  :eachval,
  :eigen,
  :eigs,
  :emptyITensor,
  :enable_tblis!,
  :entropy,
  :factorize,
  :filterinds,
  :findindex,
  :findinds,
  :firstind,
  :firstintersect,
  :firstsetdiff,
  :flux,
  :fparity,
  :getfirst,
  :getindex,
  :hadamard_product,
  :hascommoninds,
  :hasid,
  :hasind,
  :hasinds,
  :hasplev,
  :hasqns,
  :hassameinds,
  :hastags,
  :id,
  :ind,
  :index_id_rng,
  :inds,
  :inner,
  :insertblock!,
  :isactive,
  :isfermionic,
  :ishermitian,
  :isindequal,
  :itensor,
  :linkindex,
  :lq,
  :mapprime,
  :mapprime!,
  :matmul,
  :matrix,
  :maxdim,
  :mindim,
  :modulus,
  :mul!,
  :nblocks,
  :nnz,
  :nnzblocks,
  :noncommonind,
  :noncommoninds,
  :noprime,
  :noprime!,
  :norm,
  :normalize,
  :normalize!,
  :not,
  :nullspace,
  :nzblock,
  :nzblocks,
  :onehot,
  :order,
  :permute,
  :plev,
  :polar,
  :pop,
  :popfirst,
  :prime,
  :prime!,
  :product,
  :push,
  :pushfirst,
  :ql,
  :qn,
  :qr,
  :randn!,
  :randomITensor,
  :random_itensor,
  :readcpp,
  :removeqn,
  :removeqns,
  :removetags,
  :removetags!,
  :replaceind,
  :replaceind!,
  :replaceindex!,
  :replaceinds,
  :replaceinds!,
  :replaceprime,
  :replacetags,
  :replacetags!,
  :reset_warn_order!,
  :rmul!,
  :rq,
  :scalar,
  :scale!,
  :set_warn_order!,
  :setdir,
  :setelt,
  :setindex,
  :setprime,
  :setprime!,
  :setspace,
  :settags,
  :settags!,
  :sim,
  :siteindex,
  :space,
  :splitblocks,
  :storage,
  :store,
  :svd,
  :swapind,
  :swapinds,
  :swapinds!,
  :swapprime,
  :swapprime!,
  :swaptags,
  :swaptags!,
  :tags,
  :tr,
  :transpose,
  :truncerror,
  :unionind,
  :unioninds,
  :uniqueind,
  :uniqueindex,
  :uniqueinds,
  :use_debug_checks,
  :val,
  :vector,
  :δ,
  :⊕,
  :⊙,
]
end
