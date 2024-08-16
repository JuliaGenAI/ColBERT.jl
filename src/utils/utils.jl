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
