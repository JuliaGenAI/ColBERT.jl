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


"""
    _sort_by_length(
        integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)

Sort sentences by number of attended tokens, if the number of sentences is larger than `bsize`.

# Arguments

  - `integer_ids`: The token IDs of documents to be sorted.
  - `integer_mask`: The attention masks of the documents to be sorted (attention masks are just bits).
  - `bsize`: The size of batches to be considered.

# Returns

Depending upon `bsize`, the following are returned:

  - If the number of documents (second dimension of `integer_ids`) is atmost `bsize`, then the `integer_ids` and `integer_mask` are returned unchanged.
  - If the number of documents is larger than `bsize`, then the passages are first sorted by the number of attended tokens (figured out from the `integer_mask`), and then the sorted arrays `integer_ids`, `integer_mask` are returned, along with a list of `reverse_indices`, i.e a mapping from the documents to their indices in the original order.
"""
function _sort_by_length(
        integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)
    batch_size = size(integer_ids)[2]
    if batch_size <= bsize
        # if the number of passages fits the batch size, do nothing
        integer_ids, integer_mask, Vector(1:batch_size)
    end

    lengths = vec(sum(integer_mask; dims = 1))              # number of attended tokens in each passage
    indices = sortperm(lengths)                             # get the indices which will sort lengths
    reverse_indices = sortperm(indices)                     # invert the indices list

    integer_ids[:, indices], integer_mask[:, indices], reverse_indices
end

"""
    _split_into_batches(
        integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)

Split the given `integer_ids` and `integer_mask` into batches of size `bsize`.

# Arguments

  - `integer_ids`: The array of token IDs to batch.
  - `integer_mask`: The array of attention masks to batch.

# Returns

Batches of token IDs and attention masks, with each batch having size `bsize` (with the possibility of the last batch being smaller).
"""
function _split_into_batches(
        integer_ids::AbstractMatrix{Int32}, integer_mask::AbstractMatrix{Bool}, bsize::Int)
    batch_size = size(integer_ids)[2]
    batches = Vector{Tuple{AbstractMatrix{Int32}, AbstractMatrix{Bool}}}()
    for offset in 1:bsize:batch_size
        push!(batches,
            (integer_ids[:, offset:min(batch_size, offset + bsize - 1)],
                integer_mask[:, offset:min(batch_size, offset + bsize - 1)]))
    end
    batches
end
