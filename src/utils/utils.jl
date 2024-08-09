"""
    batch(group::Vector, bsize::Int; [provide_offset::Bool = false])

Create batches of data from `group`.

Each batch is a subvector of `group` with length equal to `bsize`. If `provide_offset` is true, each batch will be a tuple containing both the offset and the subvector, otherwise only the subvector will be returned.

# Arguments

  - `group::Vector`: The input vector from which to create batches.
  - `bsize::Int`: The size of each batch.
  - `provide_offset::Bool = false`: Whether to include the offset in the output batches. Defaults to `false`.

# Returns

A vector of tuples, where each tuple contains an offset and a subvector, or just a vector containing subvectors, depending on the value of `provide_offset`.
"""
function batch(group::Vector, bsize::Int; provide_offset::Bool = false)
    vtype = provide_offset ?
            Vector{Tuple{Int, typeof(group)}} :
            Vector{typeof(group)}
    batches = vtype()
    offset = 1
    while offset <= length(group)
        if provide_offset
            push!(batches, (offset, group[offset:min(length(group), offset + bsize - 1)]))
        else
            push!(batches, group[offset:min(length(group), offset + bsize - 1)])
        end
        offset += bsize
    end
    batches
end
