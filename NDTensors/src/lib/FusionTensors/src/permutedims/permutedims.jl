using BlockArrays: Block

using NDTensors.FusionTensors:
  FusionTensor,
  StructuralData,
  n_codomain_axes,
  n_domain_axes,
  data_matrix,
  n_codomain_axes_in
using NDTensors.TensorAlgebra: BlockedPermutation, blockedperm

function Base.permutedims(
  ft::FusionTensor{T,N}, new_codomain_axes, new_domain_axes
) where {T,N}
  # designed to crash if length(new_codomain_axes) + length(new_domain_axes) != N
  perm::BlockedPermutation{2,N} = blockedperm(new_codomain_axes, new_domain_axes)
  return permutedims(ft, perm)
end

function Base.permutedims(
  ft::FusionTensor{T,N,NCoAxesIn}, perm::BlockedPermutation{2,N,B}
) where {T,N,NCoAxesIn,NCoAxesOut,B<:Tuple{NTuple{NCoAxesOut},NTuple}}

  # early return for identity operation. Do not copy.
  # TODO compile separetly, only for case M==J?
  if NCoAxesIn == NCoAxesOut && Tuple(perm) == ntuple(i -> i, N)
    return ft
  end

  structural_data = StructuralData(codomain_axes(ft), domain_axes(ft), perm)
  permuted_data_matrix = _permute_data(ft, structural_data)

  codomain_axes_out = (i -> axes(ft)[i]).(perm[Block(1)])
  domain_axes_out = (i -> axes(ft)[i]).(perm[Block(2)])
  out = FusionTensor(codomain_axes_out, domain_axes_out, permuted_data_matrix)
  return out
end

function _permute_data(
  ft::FusionTensor{T,N,NCoAxesIn,NDoAxesIn,G},
  structural_data::StructuralData{N,NCoAxesIn,NDoAxesIn,NCoAxesOut,NDoAxesOut,G},
) where {T,N,NCoAxesIn,NDoAxesIn,NCoAxesOut,NDoAxesOut,G}
  perm = permutation(structural_data)
  codomain_axes_out = (i -> axes(ft)[i]).(perm[Block(1)])
  domain_axes_out = (i -> axes(ft)[i]).(perm[Block(2)])

  # stupid permute: cast to dense, permutedims, cast to FusionTensor
  arr = Array(ft)
  permuted_arr = permutedims(arr, Tuple(perm))

  ftp = FusionTensor(codomain_axes_out, domain_axes_out, permuted_arr)
  permuted_data_matrix = data_matrix(ftp)
  return permuted_data_matrix
end

"""
struct StructuralData
  elementary_label_indices::Matrix{Int}
  # each of these could be a vector of GradedAxes

  # this allows to localize data chunks inside tensor
  internal_axes_input_rows::Vector{GradedUnitRange}
  internal_axes_input_cols::Vector{GradedUnitRange}
  internal_axes_output_rows::Vector{GradedUnitRange}
  internal_axes_output_cols::Vector{GradedUnitRange}

  # this is the hard stuff
  # BlockSparseArray contains its axes
  # CG can always be real. Impose Float64?
  # use 2 (I/O) or 4 (IR,IC/OR,OC) axes?
  isometries::Vector{BlockSparseArray{Float64}}
end

"" "
# other possibility, with integers
struct StructuralData
    elementary_label_indices::Matrix{Int}
    # each of these could be a vector of GradedAxes
    internal_multiplicities_input_rows::Matrix{Int}
    internal_multiplicities_input_columns::Matrix{Int}
    internal_multiplicities_output_rows::Matrix{Int}
    internal_multiplicities_output_columns::Matrix{Int}

    # this is needed if the other stuff are integer
    block_irreps_out::Vector{C}

    # this is the hard stuff
    isometry_in_blocks::Matrix{Matrix{Float64}}  # or Dict(int)
    # or Dict{Tuple{Int64, Int64}, Matrix}
end
"" "

function permutedims(  # args != Base.permutedims. Change name?
  t::NonAbelianTensor,
  permutation::Vector{Int},
  n_codomain_out::Vector{Int},
)
  if sort(permutation) != collect(1::t.ndims)
    throw(DomainError(permutation, "invalid permutation"))
  end

  in_axes = [t.codomain_axes, t.domain_axes]
  out_domain_axes = in_axes[permutation[begin:n_codomain_out]]
  out_codomain_axes = in_axes[permutation[n_codomain_out:end]]

  # TODO cache me
  structural_data = compute_structural_data(
    t.axes, permutation, t.n_codomain_axes, n_codomain_out
  )

  out_row_axis, out_col_axis, out_blocks = transpose_data(
    t, structural_data, permutation, n_codomain_out
  )

  out = FusionTensor(
    out_codomain_axes, out_domain_axes, out_row_axis, out_col_axis, out_blocks
  )

  # norm should be the same
  @debug_check abs(norm(out) - norm(t)) < 1e-13 * norm(t)
  return out
end

function compute_structural_data(
  graded_axes::Vector{GradedUnitRange},
  permutation::Vector{Integer},
  n_codomain_in::Integer,
  n_codomain_out::Integer,
)
  @debug_check length(graded_axes) == length(permutation)
  ndim = length(graded_axes)
  @debug_check 0 < n_codomain_in < ndim
  @debug_check 0 < n_codomain_out < ndim

  # this function is purely structual: it reads the labels insides axes, but not the external
  # multiplicities
  n_ele = .*(blocklengths.(graded_axes))

  unitaries = []
  ele_indices = []

  for i_ele in 1:n_ele
    if block_is_allowed(i_ele)
      u = compute_cg_overlap(i_ele)
      # TODO cache u
      push!(unitaries, u)
      push!(ele_indices, i_ele)
    end
  end
  structural_data = (ele_indices, unitaries)
  return structural_data
end

function compute_cg_overlap(args)
  # this can be done with 6j
  # for now contract Clebsch-Gordan trees
  cg_codomain_in, cg_domain_in, cg_codomain_out, cg_domain_out, permutation = args
  return contract_cg_trees(
    cg_codomain_in, cg_domain_in, cg_codomain_out, cg_domain_out, permutation
  )
end

function contract_cg_trees(
  cg_codomain_in, cg_domain_in, cg_codomain_out, cg_domain_out, permutation
)
  return isometry
end

function transpose_data(
  t::NonAbelianTensor{T,C},
  structural_data,
  permutation::Vector{Integer},
  n_codomain_out::Int,
)
  (ele_indices, unitaries) = structural_data
  out_blocks = [zeros{T}(size) for size in block_sizes]

  # recover external multiplicities
  emir, slices_ir = compute_slices(external_mul_ir, internal_mul_ir)
  emic, slices_ic = compute_slices(external_mul_ic, internal_mul_ic)
  emor, slices_or = compute_slices(external_mul_or, internal_mul_or)
  emoc, slices_oc = compute_slices(external_mul_oc, internal_mul_oc)

  # the most external loop is always over the set of elementary labels
  # here "ele_indices" is coming from structural_data
  # it contains only the elementary_label_sets allowed by the symmetry (forbidden ones are already gone)
  for i_ele in ele_indices    #TODO PARALLELIZE ME

    # there are a 4 possibilites for the inner loops:
    # * constructing a block containing all non-abelian degrees of freedom
    #   - pro: apply full unitaty in one go (matmul)
    #   - con: data locality is lost => different matrix blocks can be far away
    #   - con: dealing with allowed yet missing matrix blocks is a mess
    #   - con: constructing the block requires complex indices handling

    # * constructing a block containing the degrees of freedom belonging to the same INPUT matrix block
    #   - pro: apply unitary over all columns in one go (column wise matmul)
    #   - pro: missing matrix blocks are easily avoided
    #   - pro: data locality preserved: all coeff from a block come from the same matrix block
    #   - pro: largest contiguous data chunks
    #   - con: constructing the block requires complex indices handling
    #   - con: need to dispatch output blocks

    # * constructing a block containing the degrees of freedom belonging to the same matrix block IN and OUT
    #   - pro: block matmul
    #   - pro: missing matrix blocks are easily avoided
    #   - pro: data locality preserved: all coeff from a block come from the same matrix block
    #   - pro: large data chunks, always contiguous
    #   - con: constructing the block requires complex indices handling

    # * constructing a block containing just 1 degree of freedom
    #   - pro: data locality preserved: all coeff from a block come from the same matrix block
    #   - pro: missing matrix blocks are easily avoided
    #   - pro: simplest index handling
    #   - con: need to loop over all unitary coeff (element wise matmul)
    #   - con: small data chunks = external multiplicities

    i_ir, i_ic, i_or, i_oc = ele_indices[i_ele]
    ele_sh = Array{Int64}(undef, t.ndim)
    ele_sh[1:nrr_in] .= external_degen_ir[i_ir]
    ele_sh[(nrr_in + 1):end] .= external_degen_ic[i_ic]
    emir_ele = emir[i_ir]
    emic_ele = emic[i_ic]
    emor_ele = emor[i_or]
    emoc_ele = emoc[i_oc]
    u = unitaries[i_ele]

    # loop over input degrees of freedom
    for bin in 1:blocklength(u.axes[1])
      if bin in t.matrix_row_axis  # deal with allowed yet missing blocks

        # find ranges for this block
        range_ir = emir[i_ir] * internal_axes_input_rows[i_ir][Block(bin)]
        range_ic = emic[i_ic] * internal_axes_input_cols[i_ic][Block(bin)]

        # recover (non-contiguous) data from matrix blocks
        in_data = matrix_block[bin][range_ir, range_ic]
        in_data = reshape(in_data, ele_sh)

        # swap axes according to permutation
        in_data = permutedims(in_data, permutation)

        # change basis by applying unitary

        # loop over input degrees of freedom
        for bout in 1:blocklength(u.axes[2])
          # coefficient wise matmul
          ub = u[Block(bin), Block(bout)]
          matrix_block_out = findblock()
          for imorb in size(ub, 1)
            for imocb in size(ub, 2)
              block_out_index = get_block_index(i_ele, dof_out)
              range_or = get_slice_or(i_ele, dof_out)
              range_oc = get_slice_oc(i_ele, dof_out)
              out_blocks[bout][range_or, range_oc] += ub[imorb, imocb] * in_data
            end
          end
        end
      end
    end
  end
  return out_blocks
end
"""
